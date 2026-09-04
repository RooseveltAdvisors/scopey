# Vision

This repository is the Roosevelt Advisors fork of ArchAstro/scopey, a Rust CLI that keeps coding-agent sessions on scope with requirement summaries, trajectory judging, course corrections, and drift insights.
Upstream's product purpose is described in the README and is not restated here; this vision states why the fork exists, what it carries beyond upstream, and where it must never diverge.

## Why the fork exists

The fleet runs every coding agent under firstmate management, and scopey rides inside those same sessions through its installed hooks.
Living that close together surfaced two collisions that upstream had no reason to feel.

First, scopey's own injected corrections re-enter the session as prompts, and a phrase filter guessing at "looks like scopey" is too brittle a boundary for text that must never become user requirements.
Second, scopey's background summarize and judge helpers inherit the environment of the session they serve, so a helper launched inside a firstmate-managed session could present firstmate's identity and even take firstmate's session lock away from the captain's real session.

The fork is where scopey learns to be a safe tenant of the fleet without waiting on upstream.

## What the fork carries beyond upstream

A structured source provenance field on hook events replaces prompt-text guessing: an event that originates from scopey is generated context at the user-prompt boundary and is never persisted as a user requirement, while explicit user-authored constraints such as no-tool or read-only stay intact.
Background summarize and judge helpers drop inherited firstmate identity markers and refuse `/opt/ra/firstmate` as a working directory, so scopey's own model work can never be mistaken for a firstmate session or seize its lock; the configured `model_runner` is unchanged.
Both changes carry their own end-to-end coverage in `tests/cli_hooks.rs` and are documented in the README and CHANGELOG like any upstream change.

## What the fork must never diverge on

It stays a faithful superset of upstream: config and session formats remain compatible, and the fork's delta stays minimal, tested, and shaped so it can be proposed upstream rather than drifting into a private dialect.
Pull requests go to origin, RooseveltAdvisors/scopey, and nowhere else; upstream receives nothing from this fleet's automation.
Scopey's machinery never acquires firstmate identity or holds a firstmate lock, and machine-generated text never becomes a user requirement, no matter how plausible the phrasing.
The subagent suppression, runtime safety, and hook responsiveness upstream promises are inherited obligations here, not optional features.

## Non-goals

- No firstmate-specific features, fleet UX, or captain preferences in this codebase.
- No rewrites of upstream internals for taste; the delta exists only where the fleet's way of working demands it.
- No independent release line or divergent versioning story.

## Done well, one year out

The fork's delta is still small, still tested, and either merged upstream or trivially rebased onto upstream releases.
Scopey runs unattended inside firstmate-managed sessions with zero lock collisions and zero machine-generated requirements, and nobody has to remember the fork is there.

A change belongs in this fork when it fixes a concrete collision between scopey and the fleet's way of working.
A change is resisted when it is a general improvement upstream should carry, or when it widens the delta beyond what the collision required.
