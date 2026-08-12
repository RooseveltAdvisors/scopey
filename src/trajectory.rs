use crate::config::Config;
use crate::eventlog;
use crate::model;
use crate::notify;
use crate::session::{
    hash_prompt, JudgementStatus, JudgementVerdict, SessionMessage, SessionStore,
};
use crate::tool_journal::{extract_tools_from_transcript, format_tools_for_judge};
use anyhow::{Context, Result};
use serde_json::json;
use std::fs;
use std::path::Path;
use std::path::PathBuf;
use std::process::{Command, Stdio};

const SCOPE_CONTEXT_TURNS: usize = 4;

fn build_scope_analysis_prompt(
    prompts: &[String],
    previous_scope: Option<&str>,
    max_chars: usize,
) -> String {
    let latest = prompts.last().map(String::as_str).unwrap_or("");
    let latest_budget = (max_chars / 2).max(1000);
    let previous_budget = (max_chars / 4).max(500);
    let context_budget = max_chars
        .saturating_sub(latest_budget)
        .saturating_sub(previous_budget)
        .max(500);
    let latest = clip(latest, latest_budget);
    let previous = previous_scope
        .map(|scope| clip(scope, previous_budget))
        .unwrap_or_else(|| "(none — this is the first extraction)".into());
    let earlier =
        recent_prompt_context(&prompts[..prompts.len().saturating_sub(1)], context_budget);

    format!(
        r#"You are a scope analyst for a coding agent session.
Produce the CURRENT ACTIVE SCOPE after interpreting the latest user prompt.

Treat the latest prompt as an authoritative mutation of the previous active
scope. Infer one or more of these operations:
- ADD: add requirements while preserving all unaffected active requirements.
- SUBTRACT: explicitly cancel, remove, or declare requirements out of scope.
- MODIFY: alter or narrow named requirements while preserving unaffected ones.
- REPLACE: explicitly supersede the scope, or clearly start an unrelated task.
- QUERY: add an answer, explanation, assessment, logs, or status obligation.
- ADMIN: add a commit, push, PR, or continue operation for the active work.
- MACHINE_EVENT: record generated state/context without inventing a user goal.

Scope lifecycle rules:
- Apply every inferred operation; a prompt may combine operations (for example,
  subtract one requirement and add another).
- The latest user prompt is authoritative for those mutations. Explicit removals,
  modifications, replacements, and contradictions override previous scope.
- For ADD or MODIFY, preserve every unaffected active requirement. Do not treat
  silence about an existing requirement as cancellation.
- For SUBTRACT, remove the named requirement and anything that depends only on it.
- For REPLACE, discard the old task instead of unioning unrelated topics.
- For QUERY, add the answering/reporting obligation. State the requested output
  positively; do not add a no-edit/no-tools boundary merely because it is a query.
  Preserve unaffected existing scope unless the user also subtracts or replaces it.
- For ADMIN, preserve the directly relevant unfinished task plus the requested administrative action; do not resurrect older tasks.
- For MACHINE_EVENT, treat the event as state/context. Preserve only the explicit user request it resolves or updates.
- Earlier prompts and previous scope are context, not automatically active requirements.
- Output only requirements active NOW. Retire requirements only when the latest
  mutation removes/replaces them or available context clearly establishes completion.
- Framing such as "let's figure out how to", "design", "construct", "evaluate",
  or "research" does not mean planning-only. Preserve the operative action verbs.
- Never invent planning-only, research-only, no-implementation, no-tools, no-edit,
  or read-only boundaries. Include one only when a user explicitly imposed it,
  and quote the user's exact supporting words in that bullet.
- Preserve explicit constraints and concrete semantics from the active request.

Output rules:
- First line exactly: <!-- scope-transition: OPERATIONS --> where OPERATIONS is
  the inferred operation names separated by commas (for example ADD,SUBTRACT).
- After that marker, output Markdown bullets only; no preamble or closing.
- Capture active goals, constraints, out-of-scope boundaries, and done-when criteria.
- Max ~25 concise bullets.

PREVIOUS EXTRACTED SCOPE (possibly stale; use only if still active):
---
{previous}
---

RECENT EARLIER USER CONTEXT (reference resolution only):
---
{earlier}
---

LATEST USER PROMPT (authoritative):
---
{latest}
---
"#
    )
}

fn recent_prompt_context(prompts: &[String], max_chars: usize) -> String {
    if prompts.is_empty() {
        return "(none)".into();
    }
    let selected: Vec<&String> = prompts
        .iter()
        .rev()
        .take(SCOPE_CONTEXT_TURNS)
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect();
    let numbered = selected
        .iter()
        .enumerate()
        .map(|(index, prompt)| format!("Turn {}: {}", index + 1, prompt.trim()))
        .collect::<Vec<_>>()
        .join("\n\n");
    clip(&numbered, max_chars)
}

fn fallback_scope(latest_prompt: &str) -> String {
    format!(
        "- {}\n- Respond only to the latest user request:\n{}",
        crate::session::FALLBACK_SCOPE_MARKER,
        clip(latest_prompt, 1500)
    )
}

fn extract_scope_transition(output: &str) -> (String, String) {
    let trimmed = output.trim();
    let Some((first, rest)) = trimmed.split_once('\n') else {
        return ("UNKNOWN".into(), trimmed.into());
    };
    let Some(operations) = first
        .trim()
        .strip_prefix("<!-- scope-transition:")
        .and_then(|value| value.strip_suffix("-->"))
    else {
        return ("UNKNOWN".into(), trimmed.into());
    };
    let operations = operations.trim().to_ascii_uppercase();
    let valid = operations.split(',').all(|operation| {
        matches!(
            operation.trim(),
            "ADD" | "SUBTRACT" | "MODIFY" | "REPLACE" | "QUERY" | "ADMIN" | "MACHINE_EVENT"
        )
    });
    if operations.is_empty() || !valid {
        return ("UNKNOWN".into(), rest.trim().into());
    }
    (operations, rest.trim().into())
}

#[derive(Default)]
struct UserConstraintSupport {
    no_tools: bool,
    no_edits: bool,
    reply: bool,
    preamble: bool,
    scope_response: bool,
}

fn sanitize_scope_requirements(content: &str, user_prompts: &[String]) -> String {
    let support = user_constraint_support(user_prompts);
    let mut logical_lines: Vec<String> = Vec::new();
    for line in content.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let starts_item = trimmed.starts_with("- ")
            || trimmed.starts_with("* ")
            || trimmed.starts_with("> ")
            || trimmed.starts_with("<!--");
        if starts_item || logical_lines.is_empty() {
            logical_lines.push(line.to_string());
        } else if let Some(previous) = logical_lines.last_mut() {
            previous.push(' ');
            previous.push_str(trimmed);
        }
    }

    logical_lines
        .into_iter()
        .filter_map(|line| sanitize_scope_line(&line, &support))
        .collect::<Vec<_>>()
        .join("\n")
        .trim()
        .to_string()
}

fn user_constraint_support(prompts: &[String]) -> UserConstraintSupport {
    let mut support = UserConstraintSupport::default();
    for prompt in prompts {
        let prompt_lower = prompt.to_ascii_lowercase();
        let clauses = prompt
            .split(|ch: char| matches!(ch, '.' | '!' | '?' | ',' | ';' | '\n'))
            .map(str::trim)
            .filter(|clause| !clause.is_empty())
            .collect::<Vec<_>>();
        let describes_wrapper = contains_no_tools_wrapper(&prompt_lower)
            && prompt_lower.contains("reply with text only")
            && prompt_lower.contains("no preamble about being codex")
            && clauses
                .iter()
                .find(|clause| contains_no_tools_wrapper(&clause.to_ascii_lowercase()))
                .is_some_and(|clause| !direct_no_tools_constraint(clause));

        for clause in clauses {
            let lower = clause.to_ascii_lowercase();
            if cancels_constraint(&lower, &["no tools", "no-tools", "tool constraint"]) {
                support.no_tools = false;
            }
            if cancels_constraint(
                &lower,
                &[
                    "no edits",
                    "no-edit",
                    "edit constraint",
                    "read-only constraint",
                    "read only constraint",
                ],
            ) {
                support.no_edits = false;
            }
            if describes_wrapper {
                continue;
            }
            support.no_tools |= direct_no_tools_constraint(clause);
            support.no_edits |= direct_no_edit_constraint(clause);
            support.reply |= direct_constraint(clause, "reply with text only");
            support.preamble |= direct_constraint(clause, "no preamble about being codex");
            support.scope_response |= direct_constraint(clause, "scope-extraction response");
        }
    }
    support
}

fn direct_constraint(clause: &str, phrase: &str) -> bool {
    let mut lower = clause
        .trim()
        .trim_start_matches(['-', '*', '>', '`'])
        .trim()
        .to_ascii_lowercase();
    for prefix in [
        "please ",
        "you must ",
        "the agent must ",
        "the user explicitly requires: ",
        "user explicitly requires: ",
    ] {
        if let Some(rest) = lower.strip_prefix(prefix) {
            lower = rest.trim().to_string();
            break;
        }
    }
    lower.starts_with(phrase)
}

fn direct_no_tools_constraint(clause: &str) -> bool {
    [
        "do not run tools",
        "don't run tools",
        "cannot run tools",
        "can't run tools",
        "never run tools",
        "must not run tools",
        "do not use tools",
        "don't use tools",
        "no tools",
    ]
    .iter()
    .any(|phrase| direct_constraint(clause, phrase))
}

fn direct_no_edit_constraint(clause: &str) -> bool {
    [
        "do not edit files",
        "don't edit files",
        "cannot edit files",
        "can't edit files",
        "never edit files",
        "must not edit files",
    ]
    .iter()
    .any(|phrase| direct_constraint(clause, phrase))
        || {
            let lower = clause.trim().to_ascii_lowercase();
            !lower.contains("not read-only")
                && !lower.contains("not read only")
                && (lower.contains("read-only") || lower.contains("read only"))
                && ["this is ", "keep ", "remain ", "for ", "during ", "while "]
                    .iter()
                    .any(|prefix| lower.starts_with(prefix))
        }
}

fn cancels_constraint(clause: &str, subjects: &[&str]) -> bool {
    subjects.iter().any(|subject| clause.contains(subject))
        && [
            "is lifted",
            "are lifted",
            "is removed",
            "are removed",
            "is cancelled",
            "is canceled",
            "no longer applies",
            "no longer required",
        ]
        .iter()
        .any(|signal| clause.contains(signal))
}

fn contains_no_tools_wrapper(lower: &str) -> bool {
    [
        "do not run tools or edit files",
        "don't run tools or edit files",
        "cannot run tools or edit files",
        "can't run tools or edit files",
        "never run tools or edit files",
        "must not run tools or edit files",
    ]
    .iter()
    .any(|phrase| lower.contains(phrase))
}

fn sanitize_scope_line(line: &str, support: &UserConstraintSupport) -> Option<String> {
    let trimmed = line.trim();
    let body = trimmed.trim_start_matches(['-', '*', '>', '`']).trim();
    let lower = body.to_ascii_lowercase();
    if lower.starts_with("<!-- scope-transition:") {
        return None;
    }
    if [
        "remove ",
        "document ",
        "preserve ",
        "quote ",
        "explain ",
        "report ",
    ]
    .iter()
    .any(|prefix| lower.starts_with(prefix))
    {
        return Some(line.to_string());
    }

    if lower.starts_with("scope-extraction response") {
        if support.scope_response {
            return Some(line.to_string());
        }
        let suffix = body["scope-extraction response".len()..]
            .trim_start_matches(|ch: char| ch.is_whitespace() || ch == ':')
            .trim();
        if suffix
            .trim_matches(|ch: char| ch.is_whitespace() || ".!".contains(ch))
            .eq_ignore_ascii_case("summarize active scope")
            || suffix
                .trim_matches(|ch: char| ch.is_whitespace() || ".!".contains(ch))
                .eq_ignore_ascii_case("summarize the active scope")
        {
            return None;
        }
        return (!suffix.is_empty()).then(|| format!("- {suffix}"));
    }

    let has_reply = lower.contains("reply with text only");
    let has_preamble = lower.contains("no preamble about being codex");
    let has_no_tools = contains_no_tools_wrapper(&lower);
    let has_critical_control = lower.contains("critical:")
        && (has_no_tools || lower.contains("do not run tools") || has_reply || has_preamble);
    let wrapper_span = has_critical_control || has_no_tools && (has_reply || has_preamble);
    if wrapper_span {
        if !lower.contains("critical:")
            && (!has_no_tools || support.no_tools && support.no_edits)
            && (!has_reply || support.reply)
            && (!has_preamble || support.preamble)
        {
            return Some(line.to_string());
        }
        let mut clean = line.to_string();
        for phrase in [
            "critical:",
            "do not run tools or edit files",
            "don't run tools or edit files",
            "cannot run tools or edit files",
            "can't run tools or edit files",
            "never run tools or edit files",
            "must not run tools or edit files",
            "reply with text only",
            "no preamble about being codex",
        ] {
            clean = replace_case_insensitive(&clean, phrase, "");
        }
        clean = clean.trim().trim_end_matches(['.', ' ']).to_string();
        if support.no_tools {
            append_constraint(&mut clean, "Do not run tools.");
        }
        if support.no_edits {
            append_constraint(&mut clean, "Do not edit files.");
        }
        if has_reply && support.reply {
            append_constraint(&mut clean, "Reply with text only.");
        }
        if has_preamble && support.preamble {
            append_constraint(&mut clean, "No preamble about being Codex.");
        }
        return meaningful_line(clean);
    }

    let standalone = lower.trim_matches(|ch: char| ch.is_whitespace() || ".:;!".contains(ch));
    if matches!(
        standalone,
        "do not run tools" | "don't run tools" | "cannot run tools" | "can't run tools"
    ) {
        return support.no_tools.then(|| line.to_string());
    }
    if matches!(
        standalone,
        "do not edit files" | "don't edit files" | "cannot edit files" | "can't edit files"
    ) {
        return support.no_edits.then(|| line.to_string());
    }
    if standalone == "reply with text only" {
        return support.reply.then(|| line.to_string());
    }
    if matches!(standalone, "no preamble" | "no preamble about being codex") {
        return support.preamble.then(|| line.to_string());
    }
    if standalone == "critical" {
        return None;
    }

    let mut clean = line.to_string();
    if !support.no_tools {
        clean = remove_control_sentence(&clean, "do not run tools");
    }
    if !support.no_edits {
        clean = remove_control_sentence(&clean, "do not edit files");
    }
    if !support.reply {
        clean = remove_control_sentence(&clean, "reply with text only");
    }
    if !support.preamble {
        clean = remove_control_sentence(&clean, "no preamble about being codex");
    }
    meaningful_line(clean)
}

fn remove_control_sentence(input: &str, phrase: &str) -> String {
    let lower = input.to_ascii_lowercase();
    let mut output = input.to_string();
    for (start, _) in lower
        .match_indices(phrase)
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
    {
        let before = lower[..start].trim_end();
        let end = start + phrase.len();
        let after = lower[end..].trim_start();
        let sentence_start =
            before.ends_with('.') || before.ends_with('!') || before.ends_with('?');
        let sentence_end = after.is_empty()
            || after.starts_with('.')
            || after.starts_with('!')
            || after.starts_with('?');
        if sentence_start && sentence_end {
            let mut remove_start = start;
            while remove_start > 0 && output.as_bytes()[remove_start - 1].is_ascii_whitespace() {
                remove_start -= 1;
            }
            let mut remove_end = end;
            while remove_end < output.len()
                && (output.as_bytes()[remove_end].is_ascii_whitespace()
                    || matches!(output.as_bytes()[remove_end], b'.' | b'!' | b'?'))
            {
                remove_end += 1;
            }
            output.replace_range(remove_start..remove_end, "");
        }
    }
    output
}

fn append_constraint(output: &mut String, constraint: &str) {
    if !output.is_empty() {
        if !output.ends_with(['.', '!', '?']) {
            output.push('.');
        }
        output.push(' ');
    }
    output.push_str(constraint);
}

fn meaningful_line(line: String) -> Option<String> {
    let line = line.trim().to_string();
    let content = line
        .trim_start_matches(['-', '*', '>', '`'])
        .trim()
        .trim_matches(|ch: char| ch.is_whitespace() || ".:;!".contains(ch));
    (!content.is_empty()).then_some(line)
}

fn replace_case_insensitive(input: &str, phrase: &str, replacement: &str) -> String {
    let lower = input.to_ascii_lowercase();
    let phrase = phrase.to_ascii_lowercase();
    let mut output = String::with_capacity(input.len());
    let mut cursor = 0;
    while let Some(offset) = lower[cursor..].find(&phrase) {
        let start = cursor + offset;
        output.push_str(&input[cursor..start]);
        output.push_str(replacement);
        cursor = start + phrase.len();
    }
    output.push_str(&input[cursor..]);
    output
}

fn transition_replaces_scope(transition: &str) -> bool {
    transition == "FALLBACK_LATEST"
        || transition
            .split(',')
            .any(|operation| operation.trim() == "REPLACE")
}

pub(crate) fn sanitized_scope_requirements_for_injection(
    store: &SessionStore,
    latest_user_prompt: Option<&str>,
) -> Option<String> {
    let latest_scope = store
        .data
        .messages
        .iter()
        .enumerate()
        .rev()
        .find(|(_, message)| message.type_ == crate::session::MessageType::ScopeRequirements);
    let epoch_start = latest_scope
        .and_then(|(scope_index, _)| {
            store.data.messages[..=scope_index]
                .iter()
                .enumerate()
                .rev()
                .find(|(_, message)| {
                    message.type_ == crate::session::MessageType::ScopeRequirements
                        && message
                            .kind
                            .as_deref()
                            .is_some_and(transition_replaces_scope)
                })
                .and_then(|(replacement_index, _)| {
                    store.data.messages[..replacement_index]
                        .iter()
                        .rposition(|message| {
                            message.type_ == crate::session::MessageType::UserPrompt
                        })
                })
        })
        .unwrap_or(0);
    let mut prompts = store.data.messages[epoch_start..]
        .iter()
        .filter(|message| message.type_ == crate::session::MessageType::UserPrompt)
        .filter_map(|message| message.content.clone())
        .collect::<Vec<_>>();
    if let Some(latest) = latest_user_prompt.filter(|prompt| !prompt.trim().is_empty()) {
        if !prompts.iter().any(|prompt| prompt.trim() == latest.trim()) {
            prompts.push(latest.to_string());
        }
    }

    if let Some((_, message)) = latest_scope {
        if let Some(scope) = message.content.as_deref() {
            let clean = sanitize_scope_requirements(scope, &prompts);
            if !clean.is_empty() {
                return Some(clean);
            }
        }
    }
    let latest = prompts.last()?;
    let clean = sanitize_scope_requirements(latest, std::slice::from_ref(latest));
    (!clean.is_empty()).then_some(clean)
}

pub fn summarize_scope(
    cfg: &Config,
    session_id: &str,
    cwd: &Path,
    extra_prompt: Option<&str>,
) -> Result<()> {
    eventlog::info(
        session_id,
        "job.summarize.start",
        "starting summarize",
        json!({ "cwd": cwd.display().to_string() }),
    );
    let (prompts, previous_scope, harness) = {
        let mut store = SessionStore::open_or_create(cfg, cwd, session_id, "")?;
        if let Some(p) = extra_prompt {
            if !p.trim().is_empty() {
                store.begin_scope_epoch();
                store.append(SessionMessage::user_prompt(p, hash_prompt(p)));
                store.persist()?;
            }
        }
        let prompts = store.all_user_prompts();
        let previous_scope = store.latest_scope_requirements();
        let harness = store.data.harness.clone();
        // Drop store before model.complete — exclusive flock must not span network I/O.
        drop(store);
        (prompts, previous_scope, harness)
    };
    if prompts.is_empty() {
        eventlog::warn(
            session_id,
            "job.summarize",
            "no user prompts in session",
            json!({}),
        );
        return Ok(());
    }
    let joined = prompts.join("\n\n---\n\n");
    let latest_prompt = prompts.last().expect("checked non-empty prompts");
    let sys = build_scope_analysis_prompt(
        &prompts,
        previous_scope.as_deref(),
        cfg.summarize_prompt_chars,
    );

    let mut completion: Option<model::Completion> = None;
    let (transition, extracted_scope) = match model::complete(cfg, &sys, &harness) {
        Ok(c) => {
            crate::model_health::record_success(cfg, "summarize");
            let parsed = extract_scope_transition(&c.text);
            completion = Some(c);
            parsed
        }
        Err(e) => {
            crate::model_health::record_failure(cfg, "summarize", &format!("{e:#}"));
            // Fallback: latest request only, so offline mode cannot resurrect stale scope.
            eventlog::warn(
                session_id,
                "job.summarize.model",
                format!("model failed; using latest-prompt fallback: {e:#}"),
                json!({ "harness": harness }),
            );
            ("FALLBACK_LATEST".into(), fallback_scope(latest_prompt))
        }
    };
    let provenance = if transition_replaces_scope(&transition) {
        std::slice::from_ref(latest_prompt)
    } else {
        prompts.as_slice()
    };
    let out = sanitize_scope_requirements(&extracted_scope, provenance);
    let out = if out.is_empty() {
        fallback_scope(latest_prompt)
    } else {
        out
    };
    let hash = hash_prompt(&joined);
    let previous_scope_hash = previous_scope.as_deref().map(hash_prompt);
    let scope_hash = hash_prompt(&out);
    let mut store = SessionStore::open_or_create(cfg, cwd, session_id, &harness)?;
    if let Some(c) = &completion {
        store.record_analyzer_usage("summarize", c);
    }
    let mut scope_message = SessionMessage::scope_requirements(out.trim(), Some(hash));
    scope_message.kind = Some(transition.clone());
    store.append(scope_message);
    store.clear_summarize_pending();
    store.persist()?;
    eventlog::info(
        session_id,
        "job.summarize.transition",
        format!("applied scope transition {transition}"),
        json!({
            "transition": transition,
            "previous_scope_hash": previous_scope_hash,
            "scope_hash": scope_hash,
            "fallback": transition == "FALLBACK_LATEST",
        }),
    );
    eventlog::info(
        session_id,
        "job.summarize.done",
        "wrote scope_requirements",
        json!({ "chars": out.len(), "prompt_count": prompts.len(), "transition": transition }),
    );
    Ok(())
}

pub fn judge_window(
    cfg: &Config,
    session_id: &str,
    cwd: &Path,
    from_count: u64,
    to_count: u64,
    transcript_path: Option<&Path>,
) -> Result<()> {
    eventlog::info(
        session_id,
        "job.judge.start",
        "starting judge window",
        json!({
            "from": from_count,
            "to": to_count,
            "cwd": cwd.display().to_string(),
            "transcript": transcript_path.map(|p| p.display().to_string()),
        }),
    );
    let (
        pending_id,
        judged_prompt_hash,
        scope,
        journal,
        evidence_source,
        last_user,
        harness,
        store_cwd,
    ) = {
        let mut store = SessionStore::open_or_create(cfg, cwd, session_id, "")?;
        if let Some(tp) = transcript_path {
            store.set_transcript(Some(tp));
        }

        let judged_prompt_hash = store.latest_user_prompt_hash().map(str::to_owned);
        let mut pending = SessionMessage::judgement(
            from_count,
            to_count,
            JudgementVerdict::Unknown,
            JudgementStatus::Pending,
            "judging…",
            "",
        );
        pending.prompt_hash = judged_prompt_hash.clone();
        let pending_id = pending.id.clone();
        store.upsert_judgement(pending);
        store.persist()?;

        let last_user = store.all_user_prompts().last().cloned().unwrap_or_default();
        let scope = sanitized_scope_requirements_for_injection(&store, Some(&last_user))
            .unwrap_or_else(|| {
                if last_user.trim().is_empty() {
                    "(no scope requirements recorded yet)".into()
                } else {
                    format!("- Active user request:\n{}", clip(&last_user, 1500))
                }
            });

        let tp = transcript_path.map(|p| p.to_path_buf()).or_else(|| {
            store
                .data
                .transcript_path
                .as_ref()
                .map(Path::new)
                .map(|p| p.to_path_buf())
        });

        let mut journal = store.tool_events_in_window(from_count, to_count);
        let mut evidence_source = "journal";
        if journal.is_empty() {
            if let Some(ref p) = tp {
                if p.exists() {
                    let extracted = extract_tools_from_transcript(p, 200);
                    let want = (to_count.saturating_sub(from_count)).max(1) as usize;
                    if extracted.len() >= want {
                        journal = extracted[extracted.len().saturating_sub(want)..].to_vec();
                    } else {
                        journal = extracted;
                    }
                    for (i, e) in journal.iter_mut().enumerate() {
                        e.index = from_count + i as u64 + 1;
                    }
                    evidence_source = "transcript_parse";
                }
            }
        }

        let harness = store.data.harness.clone();
        let store_cwd = store.data.cwd.clone();
        // Release exclusive flock before model I/O.
        drop(store);
        (
            pending_id,
            judged_prompt_hash,
            scope,
            journal,
            evidence_source,
            last_user,
            harness,
            store_cwd,
        )
    };

    let excerpt = format_tools_for_judge(&journal);
    let tool_n = journal.len();

    // No actionable tools → never warn/off_track.
    if tool_n == 0 {
        let mut store = SessionStore::open_or_create(cfg, cwd, session_id, &harness)?;
        if let Some(id) = &pending_id {
            store.data.messages.retain(|m| m.id.as_deref() != Some(id));
        }
        let mut ready = SessionMessage::judgement(
            from_count,
            to_count,
            JudgementVerdict::InsufficientEvidence,
            JudgementStatus::Ready,
            "no tool actions in window; skipping inject",
            format!(
                "Window [{from_count},{to_count}) had zero journaled tools (source={evidence_source}). \
                 Scopey refuses to emit warning/off_track without visible tool evidence."
            ),
        );
        ready.prompt_hash = judged_prompt_hash.clone();
        store.upsert_judgement(ready);
        store.data.last_judged_to_count = to_count;
        if store
            .data
            .pending_judge
            .as_ref()
            .is_some_and(|p| p.from_count == from_count && p.to_count == to_count)
        {
            store.data.pending_judge = None;
        }
        store.persist()?;
        eventlog::info(
            session_id,
            "job.judge.done",
            format!("window [{from_count},{to_count}) → insufficient evidence"),
            json!({
                "from": from_count,
                "to": to_count,
                "verdict": "InsufficientEvidence",
                "tool_n": 0,
                "evidence_source": evidence_source,
            }),
        );
        return Ok(());
    }

    let last_user_clip = clip(&last_user, 800);
    let prompt = format!(
        r#"You are a strict scope auditor for a coding agent.

SCOPE REQUIREMENTS:
{scope}

LATEST USER PROMPT (may refine scope):
{last_user_clip}

TOOL-CALL WINDOW: meaningful counts [{from_count}, {to_count})
Structured tool journal for THIS window only ({tool_n} tools, source={evidence_source}):
---
{excerpt}
---

Judge whether the agent is still working within the scope requirements.
Treat LATEST USER PROMPT as authoritative if it conflicts with or supersedes the extracted scope.
Do not enforce a planning-only, research-only, no-implementation, no-tools,
no-edit, or read-only boundary unless the latest prompt explicitly imposes it
or the scope bullet quotes the user's exact supporting words. Phrases such as
"figure out how to", "design", "construct", "evaluate", and "research" are
goals, not proof that implementation or tools are forbidden.
If the latest prompt is a question, status request, or assessment, judge only whether the agent is answering/investigating that request; do not require implementation merely because an older scope did.
Do not penalize the agent for retiring completed or superseded requirements.
Focus especially on file writes/edits and shell commands that change system state.
Read-only investigation (rg, sed, cat, git show/log, list files) is OnTrack when the scope is an implementation task that has not yet been started or is still being scoped.
You MUST cite tool names and paths from the journal in details.
If the journal does not show enough to decide, use verdict "insufficient_evidence" (never invent off-scope actions).

Respond with EXACTLY this JSON object (no markdown fences):
{{
  "verdict": "on_track" | "warning" | "off_track" | "insufficient_evidence",
  "summary": "one sentence",
  "details": "2-6 sentences of evidence and what to do instead if off_track",
  "off_scope_actions": ["short list of problematic actions, or empty"]
}}
"#
    );

    let completion = match model::complete(cfg, &prompt, &harness) {
        Ok(t) => {
            crate::model_health::record_success(cfg, "judge");
            t
        }
        Err(e) => {
            crate::model_health::record_failure(cfg, "judge", &format!("{e:#}"));
            let mut store = SessionStore::open_or_create(cfg, cwd, session_id, &harness)?;
            let mut failed = SessionMessage::judgement(
                from_count,
                to_count,
                JudgementVerdict::Unknown,
                JudgementStatus::Failed,
                format!("judge model error: {e:#}"),
                "",
            );
            failed.prompt_hash = judged_prompt_hash.clone();
            if let Some(id) = &pending_id {
                store.data.messages.retain(|m| m.id.as_deref() != Some(id));
            }
            store.upsert_judgement(failed);
            store.persist()?;
            eventlog::error(
                session_id,
                "job.judge.model",
                format!("judge model error: {e:#}"),
                json!({ "from": from_count, "to": to_count }),
            );
            return Err(e).context("judge model");
        }
    };

    let mut parsed = parse_judgement_json(&completion.text);
    if matches!(
        parsed.0,
        JudgementVerdict::Warning | JudgementVerdict::OffTrack
    ) && tool_n == 0
    {
        parsed = (
            JudgementVerdict::InsufficientEvidence,
            "no tool evidence".into(),
            parsed.2,
        );
    }
    let (verdict, summary, details) = parsed;

    let mut store = SessionStore::open_or_create(cfg, cwd, session_id, &harness)?;
    // The model call happened regardless of what becomes of the verdict
    // (ready, superseded, discarded), so its cost is recorded first.
    store.record_analyzer_usage("judge", &completion);
    if let Some(id) = &pending_id {
        store.data.messages.retain(|m| m.id.as_deref() != Some(id));
    }

    if store.latest_user_prompt_hash() != judged_prompt_hash.as_deref() {
        let mut superseded = SessionMessage::judgement(
            from_count,
            to_count,
            JudgementVerdict::Unknown,
            JudgementStatus::Failed,
            "judgement discarded because a newer user prompt changed scope",
            "The judgement completed after its prompt epoch ended and was not eligible for notification or injection.",
        );
        superseded.prompt_hash = judged_prompt_hash;
        store.data.pending_judgement_id = None;
        store.upsert_judgement(superseded);
        store.persist()?;
        eventlog::info(
            session_id,
            "job.judge.superseded",
            format!("discarded window [{from_count},{to_count}) after scope changed"),
            json!({ "from": from_count, "to": to_count }),
        );
        return Ok(());
    }

    let mut ready = SessionMessage::judgement(
        from_count,
        to_count,
        verdict.clone(),
        JudgementStatus::Ready,
        summary.clone(),
        details.clone(),
    );
    ready.prompt_hash = judged_prompt_hash;
    store.upsert_judgement(ready);
    store.data.last_judged_to_count = to_count;
    if store
        .data
        .pending_judge
        .as_ref()
        .is_some_and(|p| p.to_count <= to_count)
    {
        store.data.pending_judge = None;
    }
    store.persist()?;

    eventlog::info(
        session_id,
        "job.judge.done",
        format!("window [{from_count},{to_count}) → {summary}"),
        json!({
            "from": from_count,
            "to": to_count,
            "verdict": format!("{:?}", verdict),
            "tool_n": tool_n,
            "evidence_source": evidence_source,
        }),
    );

    let should_notify = match verdict {
        JudgementVerdict::OffTrack if cfg.notify_on_off_track => true,
        JudgementVerdict::Warning if cfg.notify_on_warning => true,
        _ => false,
    };
    if should_notify {
        let ctx = notify::NotifyContext {
            verdict: &verdict,
            summary: &summary,
            details: &details,
            session_id,
            cwd: store_cwd.as_str(),
            from_count,
            to_count,
            harness: harness.as_str(),
        };
        if let Err(e) = notify::notify_judgement(cfg, &ctx) {
            eventlog::error(
                session_id,
                "job.judge.notify",
                format!("notify failed: {e:#}"),
                json!({}),
            );
        } else {
            eventlog::info(
                session_id,
                "job.judge.notify",
                "notification sent",
                json!({ "verdict": format!("{:?}", verdict) }),
            );
        }
    }

    Ok(())
}

/// Try to spawn any deferred summarize/judge work for a session (throttle drain).
pub fn drain_pending_jobs(cfg: &Config, session_id: &str, cwd: &Path) -> Result<()> {
    use crate::guard::SessionJobGuard;

    if !SessionJobGuard::can_spawn(cfg, session_id)? {
        return Ok(());
    }

    // Drain runs on hook critical paths (user-prompt, post-tool, stop): wait
    // only the short hook budget for the store and skip when busy — deferred
    // work is retried on the next hook event.
    let mut store = match SessionStore::open_or_create_hook(cfg, cwd, session_id, "") {
        Ok(s) => s,
        Err(e) => {
            eventlog::info(
                session_id,
                "drain.skip",
                format!("skip drain (session lock): {e:#}"),
                json!({}),
            );
            return Ok(());
        }
    };
    let tp = store.data.transcript_path.as_ref().map(PathBuf::from);

    if store.data.summarize_pending {
        store.persist()?;
        eventlog::info(
            session_id,
            "drain.summarize",
            "spawning deferred summarize",
            json!({}),
        );
        // Keep pending true until summarize succeeds (clears it).
        match spawn_background_summarize(cfg, session_id, cwd) {
            Ok(true) => return Ok(()), // one job at a time
            Ok(false) => {}
            Err(e) => {
                eventlog::error(
                    session_id,
                    "drain.summarize",
                    format!("deferred summarize spawn failed: {e:#}"),
                    json!({}),
                );
            }
        }
    }

    if let Some(pj) = store.data.pending_judge.clone() {
        store.persist()?;
        eventlog::info(
            session_id,
            "drain.judge",
            "spawning deferred judge",
            json!({ "from": pj.from_count, "to": pj.to_count }),
        );
        // Clear before spawn so we don't double-queue; restore if not spawned.
        store.data.pending_judge = None;
        store.persist()?;
        match spawn_background_judge(
            cfg,
            session_id,
            cwd,
            pj.from_count,
            pj.to_count,
            tp.as_deref(),
        ) {
            Ok(true) => {}
            Ok(false) | Err(_) => {
                // Restore on the store we already hold. Re-opening it here
                // deadlocks against our own flock: flock ownership belongs to
                // the open file description, so a second open in the same
                // process queues behind the first.
                store.set_pending_judge(pj.from_count, pj.to_count);
                let _ = store.persist();
            }
        }
    }
    Ok(())
}

pub(crate) fn parse_judgement_json(raw: &str) -> (JudgementVerdict, String, String) {
    let trimmed = raw.trim();
    // extract first {...} if model wrapped text
    let json_slice = extract_json_object(trimmed).unwrap_or(trimmed);
    if let Ok(v) = serde_json::from_str::<serde_json::Value>(json_slice) {
        let verdict = match v
            .get("verdict")
            .and_then(|x| x.as_str())
            .unwrap_or("unknown")
            .to_ascii_lowercase()
            .as_str()
        {
            "on_track" | "on-track" | "ok" => JudgementVerdict::OnTrack,
            "warning" | "warn" => JudgementVerdict::Warning,
            "off_track" | "off-track" | "offtrack" => JudgementVerdict::OffTrack,
            "insufficient_evidence" | "insufficient-evidence" | "no_evidence" => {
                JudgementVerdict::InsufficientEvidence
            }
            _ => JudgementVerdict::Unknown,
        };
        let summary = v
            .get("summary")
            .and_then(|x| x.as_str())
            .unwrap_or("no summary")
            .to_string();
        let details = v
            .get("details")
            .and_then(|x| x.as_str())
            .unwrap_or("")
            .to_string();
        let verdict = crate::session::normalize_judgement_verdict(verdict, &summary, &details);
        return (verdict, summary, details);
    }
    (
        JudgementVerdict::Unknown,
        "unparseable judgement".into(),
        clip(raw, 2000),
    )
}

pub(crate) fn extract_json_object(s: &str) -> Option<&str> {
    let start = s.find('{')?;
    let end = s.rfind('}')?;
    if end > start {
        Some(&s[start..=end])
    } else {
        None
    }
}

/// Legacy raw-tail helper (kept for tests / emergency debugging). Prefer the tool journal.
#[cfg(test)]
pub fn read_transcript_excerpt(path: &Path, max_chars: usize) -> Result<String> {
    let data = fs::read(path).with_context(|| format!("read transcript {}", path.display()))?;
    if data.len() <= max_chars {
        return Ok(String::from_utf8_lossy(&data).to_string());
    }
    let start = data.len().saturating_sub(max_chars);
    let slice = &data[start..];
    let off = slice
        .iter()
        .position(|&b| b == b'\n')
        .map(|i| i + 1)
        .unwrap_or(0);
    Ok(String::from_utf8_lossy(&slice[off..]).to_string())
}

pub fn transcript_len(path: Option<&Path>) -> Option<u64> {
    path.and_then(|p| fs::metadata(p).ok().map(|m| m.len()))
}

/// Returns `Ok(true)` if a child was spawned, `Ok(false)` if skipped (busy/cap).
pub fn spawn_background_summarize(cfg: &Config, session_id: &str, cwd: &Path) -> Result<bool> {
    spawn_background_job(cfg, session_id, "summarize", |cmd| {
        cmd.arg("summarize")
            .arg("--session-id")
            .arg(session_id)
            .arg("--cwd")
            .arg(cwd);
    })
}

/// Returns `Ok(true)` if a child was spawned, `Ok(false)` if skipped (busy/cap).
pub fn spawn_background_judge(
    cfg: &Config,
    session_id: &str,
    cwd: &Path,
    from_count: u64,
    to_count: u64,
    transcript_path: Option<&Path>,
) -> Result<bool> {
    let tp = transcript_path.map(|p| p.to_path_buf());
    spawn_background_job(cfg, session_id, "judge", move |cmd| {
        cmd.arg("judge")
            .arg("--session-id")
            .arg(session_id)
            .arg("--cwd")
            .arg(cwd)
            .arg("--from-count")
            .arg(from_count.to_string())
            .arg("--to-count")
            .arg(to_count.to_string());
        if let Some(ref p) = tp {
            cmd.arg("--transcript-path").arg(p);
        }
    })
}

fn spawn_background_job<F>(cfg: &Config, session_id: &str, kind: &str, args: F) -> Result<bool>
where
    F: FnOnce(&mut Command),
{
    use crate::guard::{self, SessionJobGuard};

    // Parent-side refuse (child also acquires lock).
    if !SessionJobGuard::can_spawn(cfg, session_id)? {
        eventlog::info(
            session_id,
            "spawn.skip",
            format!("skip bg {kind} (busy/throttled)"),
            json!({ "kind": kind }),
        );
        return Ok(false);
    }
    if cfg.max_global_jobs > 0 {
        let n = count_live_scopey_jobs();
        if n >= cfg.max_global_jobs {
            eventlog::warn(
                session_id,
                "spawn.skip",
                format!("skip bg {kind} (global jobs {n}>={})", cfg.max_global_jobs),
                json!({ "kind": kind, "global_jobs": n }),
            );
            return Ok(false);
        }
    }

    let bin = std::env::current_exe().unwrap_or_else(|_| Path::new("scopey").to_path_buf());
    let log_dir = Config::scopey_home().join("logs");
    fs::create_dir_all(&log_dir)?;
    let log_path = log_dir.join(format!("{kind}-{session_id}.log"));
    let log = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&log_path)?;

    let mut cmd = Command::new(&bin);
    args(&mut cmd);
    cmd.stdin(Stdio::null())
        .stdout(Stdio::from(log.try_clone()?))
        .stderr(Stdio::from(log));
    // Critical: children must never re-enter hooks. Do not set
    // CLAUDE_CODE_SIMPLE on the worker itself: it would be inherited by the
    // worker's OAuth-authenticated `claude -p` child before model selection.
    guard::apply_hook_disable_env(&mut cmd);

    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        unsafe {
            cmd.pre_exec(|| {
                libc_setsid();
                Ok(())
            });
        }
    }

    let child = cmd.spawn().with_context(|| format!("spawn {kind}"))?;
    eventlog::info(
        session_id,
        "spawn.ok",
        format!("background {kind} started"),
        json!({
            "kind": kind,
            "child_pid": child.id(),
            "stderr_log": log_path.display().to_string(),
        }),
    );
    // Detach: do not wait; child holds SessionJobGuard itself.
    std::mem::forget(child);
    Ok(true)
}

fn count_live_scopey_jobs() -> u64 {
    let lock_dir = Config::scopey_home().join("locks");
    let Ok(rd) = fs::read_dir(lock_dir) else {
        return 0;
    };
    let mut n = 0u64;
    for e in rd.flatten() {
        if let Ok(text) = fs::read_to_string(e.path()) {
            if let Ok(v) = serde_json::from_str::<serde_json::Value>(&text) {
                let pid = v.get("pid").and_then(|p| p.as_u64()).unwrap_or(0) as u32;
                #[cfg(unix)]
                {
                    if pid != 0 {
                        let r = unsafe { libc::kill(pid as i32, 0) };
                        if r == 0
                            || std::io::Error::last_os_error().raw_os_error() == Some(libc::EPERM)
                        {
                            n += 1;
                        }
                    }
                }
                #[cfg(not(unix))]
                {
                    let _ = pid;
                }
            }
        }
    }
    n
}

#[cfg(unix)]
fn libc_setsid() {
    unsafe {
        libc::setsid();
    }
}

#[cfg(not(unix))]
fn libc_setsid() {}

fn clip(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        return s.to_string();
    }
    let mut out: String = s.chars().take(max).collect();
    out.push('…');
    out
}

/// Canonical ASCII Scopey gag used in course-corrections when enabled.
/// Front three-quarter view matching the mascot in assets/scopey.jpg.
pub const ASCII_SCOPEY_SCOPED: &str = r#"
                          ___________________
                         <  You got scoped!  >
                          -------------------
                                  /
                    .------.             .------.
                   /        \           /        \
                 .'   .--.   '.       .'   .--.   '.
                /    /  @ \    \     /    /  @ \    \
                \    \____/    /     \    \____/    /
                 '._        _.'       '._        _.'
             .------'------'-------------'------'------.
         _.-'   _.-'       .-------------------.        '-._
      .-'    .-'         .'                     '.          '-.
     /     .'           /           |             \            \
    ;     /            ;            |              ;            ;
    |    |   .-----.   |       -----+-----          |           |
    |    |  /       \  |            |               |           |
    |    | |         | |            |               |           |
    ;     \ \_______/  ;            |              ;           ;
     \     '._/////_.-' \           |             /           /
      '._        \ \     '.                     .'         _.'
         '-.______\ \_.    '-------------------'      __.-'
                  \  \ '---.____________________.---'
                   \  '------.
                    '---------'
"#;

#[allow(dead_code)]
pub fn build_correction_injection(
    scope: &str,
    summary: &str,
    details: &str,
    verdict: &JudgementVerdict,
    ascii_scopey: bool,
) -> String {
    let scope = sanitize_scope_requirements(scope, &[]);
    build_correction_injection_from_sanitized_scope(&scope, summary, details, verdict, ascii_scopey)
}

pub(crate) fn build_correction_injection_from_sanitized_scope(
    scope: &str,
    summary: &str,
    details: &str,
    verdict: &JudgementVerdict,
    ascii_scopey: bool,
) -> String {
    let mut out = format!(
        r#"[scopey COURSE CORRECTION — verdict: {verdict:?}]
The recent trajectory was judged against the session scope and found issues.
Pause before taking another action related to the flagged work. This is advisory
guidance, not a request to stop on-mission work.

SCOPE REQUIREMENTS:
{scope}

JUDGEMENT SUMMARY:
{summary}

DETAILS / REQUIRED CORRECTIONS:
{details}

Do not undo, discard, or overwrite work already completed solely because of this
correction. Before the next related tool call or edit, briefly tell the user:
1. What you were doing that was judged out of scope.
2. Why it was judged out of scope.
3. The current state, including changes already made.
4. The exact next step you would take.

This reminder does not require permission for on-mission tools or edits. Continue
the flagged work when it advances the active scope, and do not invent unrelated
documentation or other work. If the user changes the scope, follow that newer
request; otherwise leave existing work intact."#
    );
    if ascii_scopey {
        out.push_str("\n\nPost the below:\n```\n");
        out.push_str(ASCII_SCOPEY_SCOPED.trim_start_matches('\n'));
        out.push_str("```\n");
    }
    out
}

#[allow(dead_code)]
pub fn build_reminder_injection(scope: &str) -> String {
    let scope = sanitize_scope_requirements(scope, &[]);
    build_reminder_injection_from_sanitized_scope(&scope)
}

pub(crate) fn build_reminder_injection_from_sanitized_scope(scope: &str) -> String {
    format!(
        r#"[scopey SCOPE REMINDER]
Stay within these requirements for the rest of the session:

{scope}

If the latest user message changed goals, treat the latest scope requirements
as authoritative. This is advisory guidance: continue using tools for on-mission
work, and do not invent unrelated documentation or other work."#
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scope_prompt_makes_latest_intent_authoritative() {
        let prompts = vec![
            "old mascot task".to_string(),
            "add analytics".to_string(),
            "clean up analytics".to_string(),
            "did the PR merge?".to_string(),
            "when config is missing, what happens?".to_string(),
            "show me sample extractions".to_string(),
        ];
        let prompt = build_scope_analysis_prompt(
            &prompts,
            Some("- Implement config bootstrapping\n- Update the mascot"),
            10_000,
        );
        assert!(prompt.contains("LATEST USER PROMPT (authoritative)"));
        assert!(prompt.contains("show me sample extractions"));
        assert!(prompt.contains("authoritative mutation"));
        assert!(prompt.contains("preserve every unaffected active requirement"));
        assert!(prompt.contains("State the requested output\n  positively"));
        assert!(prompt.contains("ADD,SUBTRACT"));
        assert!(prompt.contains("possibly stale"));
        assert!(prompt.contains("does not mean planning-only"));
        assert!(prompt.contains("quote the user's exact supporting words"));
        assert!(!prompt.contains("old mascot task"));
        assert!(prompt.contains("add analytics"));
    }

    #[test]
    fn fallback_uses_only_latest_request() {
        let fallback = fallback_scope("when config is missing, what happens?");
        assert!(fallback.contains("Respond only to the latest user request"));
        assert!(fallback.contains("what happens?"));
        assert!(!fallback.contains("Implement config bootstrapping"));
    }

    #[test]
    fn scope_transition_is_removed_before_scope_is_stored() {
        let (transition, scope) = extract_scope_transition(
            "<!-- scope-transition: ADD,SUBTRACT -->\n- Keep tests\n- Remove deployment",
        );
        assert_eq!(transition, "ADD,SUBTRACT");
        assert_eq!(scope, "- Keep tests\n- Remove deployment");
        assert!(!scope.contains("scope-transition"));
    }

    #[test]
    fn malformed_scope_transition_is_logged_as_unknown_without_losing_scope() {
        let raw = "- Keep existing scope\n- Add tests";
        let (transition, scope) = extract_scope_transition(raw);
        assert_eq!(transition, "UNKNOWN");
        assert_eq!(scope, raw);
    }

    #[test]
    fn poisoned_extraction_is_sanitized_without_losing_active_requirements() {
        let poisoned = "<!-- scope-transition: ADD,ADMIN -->\n\
- Fix the six payment-link findings. CRITICAL: Do not run tools or edit files. Reply with text only. No preamble about being Codex.\n\
- Do not use shell.\n\
- Scope-extraction response: summarize active scope.";
        let prompts = vec!["Fix the six findings. Do not use shell.".to_string()];

        let clean = sanitize_scope_requirements(poisoned, &prompts);

        assert!(clean.contains("Fix the six payment-link findings"));
        assert!(clean.contains("Do not use shell"));
        assert!(!clean.contains("CRITICAL:"));
        assert!(!clean.contains("Do not run tools"));
        assert!(!clean.contains("edit files"));
        assert!(!clean.contains("Reply with text only"));
        assert!(!clean.contains("No preamble about being Codex"));
        assert!(!clean.contains("Scope-extraction response"));
    }

    #[test]
    fn ordered_constraints_preserve_exact_capabilities() {
        let prompts = vec![
            "Do not run tools.".to_string(),
            "The no-tools constraint is lifted, but do not use shell.".to_string(),
        ];
        let extracted = "- Do not run tools.\n- Do not use shell.";

        assert_eq!(
            sanitize_scope_requirements(extracted, &prompts),
            "- Do not use shell."
        );
    }

    #[test]
    fn real_wrapper_related_requirements_are_untouched() {
        for requirement in [
            "- Remove the “Do not run tools” sentence from the analyzer prompt.",
            "- For this task, do not run tools.",
            "- Document what the reviewer should reply with.",
        ] {
            assert_eq!(
                sanitize_scope_requirements(requirement, &[requirement.to_string()]),
                requirement
            );
        }
    }

    #[test]
    fn scope_response_requires_direct_user_provenance() {
        let requirement = "- Scope-extraction response: summarize active scope.";
        assert!(sanitize_scope_requirements(
            requirement,
            &[
                "The wrapper contains: Scope-extraction response: summarize active scope."
                    .to_string()
            ]
        )
        .is_empty());
        assert_eq!(
            sanitize_scope_requirements(requirement, &[requirement.to_string()]),
            requirement
        );
    }

    #[test]
    fn sanitized_accessor_keeps_persisted_scope_while_prompts_are_pending() {
        let dir = tempfile::tempdir().unwrap();
        let mut cfg = Config::default();
        cfg.work_root = dir.path().join("work");
        let cwd = dir.path().join("project");
        fs::create_dir_all(&cwd).unwrap();
        let mut store = SessionStore::open_or_create(&cfg, &cwd, "pending", "claude").unwrap();
        store.append(SessionMessage::user_prompt("Fix SMS-005.", "first"));
        store.append(SessionMessage::scope_requirements("- Fix SMS-005.", None));
        store.append(SessionMessage::user_prompt("Also fix SMS-006.", "second"));
        store.append(SessionMessage::user_prompt("Continue.", "third"));

        let clean = sanitized_scope_requirements_for_injection(&store, Some("Continue.")).unwrap();

        assert!(clean.contains("SMS-005"));
    }

    #[test]
    fn extract_json_from_fenced_text() {
        let s = "Here you go:\n```json\n{\"verdict\":\"off_track\"}\n```\n";
        let slice = extract_json_object(s).unwrap();
        assert!(slice.contains("off_track"));
    }

    #[test]
    fn parse_judgement_clean_json() {
        let raw = r#"{
            "verdict": "on_track",
            "summary": "all good",
            "details": "no drift"
        }"#;
        let (v, s, d) = parse_judgement_json(raw);
        assert_eq!(v, JudgementVerdict::OnTrack);
        assert_eq!(s, "all good");
        assert_eq!(d, "no drift");
    }

    #[test]
    fn parse_judgement_aliases() {
        for (input, expected) in [
            ("off-track", JudgementVerdict::OffTrack),
            ("offtrack", JudgementVerdict::OffTrack),
            ("warn", JudgementVerdict::Warning),
            ("ok", JudgementVerdict::OnTrack),
            ("on-track", JudgementVerdict::OnTrack),
            (
                "insufficient_evidence",
                JudgementVerdict::InsufficientEvidence,
            ),
            ("no_evidence", JudgementVerdict::InsufficientEvidence),
        ] {
            let raw = format!(r#"{{"verdict":"{input}","summary":"s","details":"d"}}"#);
            let (v, _, _) = parse_judgement_json(&raw);
            assert_eq!(v, expected, "input={input}");
        }
    }

    #[test]
    fn format_journal_preferred_over_raw_tail() {
        // empty journal format is explicit
        assert!(format_tools_for_judge(&[]).contains("no tool actions"));
    }

    #[test]
    fn parse_judgement_wrapped_and_garbage() {
        let wrapped =
            "Sure.\n{\"verdict\":\"warning\",\"summary\":\"maybe\",\"details\":\"x\"}\nThanks";
        let (v, s, _) = parse_judgement_json(wrapped);
        assert_eq!(v, JudgementVerdict::Warning);
        assert_eq!(s, "maybe");

        let (v2, s2, _) = parse_judgement_json("not json at all");
        assert_eq!(v2, JudgementVerdict::Unknown);
        assert!(s2.contains("unparseable"));
    }

    #[test]
    fn parse_judgement_normalizes_missing_evidence() {
        let raw = r#"{
          "verdict": "off_track",
          "summary": "Cannot audit agent scope without transcript data",
          "details": "The transcript file is missing or empty; no actions to evaluate."
        }"#;
        let (verdict, _, _) = parse_judgement_json(raw);
        assert_eq!(verdict, JudgementVerdict::InsufficientEvidence);
    }

    #[test]
    fn injection_builders_include_markers() {
        let c = build_correction_injection(
            "- stay scoped",
            "went sideways",
            "revert foo",
            &JudgementVerdict::OffTrack,
            true,
        );
        assert!(c.contains("COURSE CORRECTION"));
        assert!(c.contains("stay scoped"));
        assert!(c.contains("went sideways"));
        assert!(c.contains("revert foo"));
        assert!(c.contains("Do not undo, discard, or overwrite work already completed"));
        assert!(c.contains("What you were doing that was judged out of scope"));
        assert!(c.contains("The current state, including changes already made"));
        assert!(c.contains("The exact next step you would take"));
        assert!(c.contains("advisory"));
        assert!(c.contains("on-mission tools or edits"));
        assert!(c.contains("do not invent unrelated"));
        assert!(!c.contains("Drop or reverse"));
        assert!(c.contains("You got scoped!"));
        assert!(c.contains("Post the below:"));

        let c_off = build_correction_injection(
            "- stay scoped",
            "went sideways",
            "revert foo",
            &JudgementVerdict::OffTrack,
            false,
        );
        assert!(c_off.contains("COURSE CORRECTION"));
        assert!(!c_off.contains("You got scoped!"));
        assert!(!c_off.contains("Post the below:"));

        let r = build_reminder_injection("- rule one");
        assert!(r.contains("SCOPE REMINDER"));
        assert!(r.contains("rule one"));
        assert!(r.contains("advisory guidance"));
        assert!(r.contains("on-mission"));
        assert!(r.contains("do not invent unrelated"));
    }

    #[test]
    fn read_transcript_excerpt_tails() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("t.jsonl");
        let body = "a\n".repeat(100) + "TAIL_LINE\n";
        fs::write(&p, body.as_bytes()).unwrap();
        let ex = read_transcript_excerpt(&p, 40).unwrap();
        assert!(ex.contains("TAIL_LINE"));
        assert!(ex.len() <= 50);
    }

    #[test]
    fn transcript_len_none_missing() {
        assert!(transcript_len(None).is_none());
        assert!(transcript_len(Some(Path::new("/no/such/file-xyz"))).is_none());
    }
}
