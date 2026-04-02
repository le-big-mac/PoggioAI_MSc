# Docker-Publish Parity Repair Plan

Canonical reference: `codex/math-active`

Target branch to repair: `docker-publish`

## Goal

Preserve the logical flow, artifact contracts, and information handoff behavior of
`codex/math-active`, while replacing API-driven specialist execution with local CLI
agent execution.

This plan treats `codex/math-active` as the behavioral spec. Any difference in
`docker-publish` must be classified as one of:

- intended CLI-runtime substitution
- harmless implementation difference
- regression that must be fixed

## What Must Stay Canonical

### Pipeline Flow

- Stage roster and order from `build_pipeline_stages_v2()`
- Theory and experiment routing semantics
- Resume and `--start-from-stage` behavior
- Milestone and validation loop behavior

### Artifact Contracts

- `paper_workspace/` outputs
- `math_workspace/` outputs
- proof, check, review, and handoff file locations
- required artifact gates and validation inputs

### Information Flow

- persona council -> literature review -> brainstorm -> goals -> plan
- theory-track handoffs across proposer/prover/verifiers/transcription
- experiment-track handoffs across literature/design/run/verify/transcription
- post-track synthesis into formalized results, writeup, proofreading, review

### Agent Contracts

- each agent reads the same upstream artifacts as canonical
- each agent writes the same mandatory outputs as canonical
- prompt instructions still reference valid tools/commands for the new runtime

## Allowed Differences

- model backend selection is CLI-based instead of API-based
- budgeting is invocation/wall-clock based instead of token/cost based
- tool invocation is via stable shell commands instead of LangChain tool calls
- optional quick-pass features may exist, but must not alter standard pipeline parity

## Repair Workstreams

### 1. Flow Parity Audit

Acceptance criteria:

- `build_pipeline_stages_v2()` matches canonical stage order
- graph routing still preserves canonical theory/experiment/post-track sequencing
- CLI-only additions do not change default full-research behavior

Checklist:

- compare `runner.py` startup and resume semantics
- compare `graph.py` stage rosters and routing helpers
- compare graph config fields used by build functions
- isolate non-parity additions such as quick-pass behind explicit flags only

### 2. Artifact Parity Audit

Acceptance criteria:

- all canonical artifact paths remain unchanged
- all validators and downstream readers point to the same paths
- CLI wrappers operate on canonical subdirectories

Checklist:

- audit `math_workspace/claim_graph.json`
- audit `math_workspace/proofs/`
- audit `math_workspace/checks/`
- audit `paper_workspace/*.tex|*.json|*.md`
- compare required-artifact gates in `runner.py` and supervision code

### 3. Tool Contract Parity

Acceptance criteria:

- every canonical tool concept still has a valid runtime equivalent
- prompts reference only valid runtime commands
- no prompt instructs agents to use removed callable tools

Checklist:

- map canonical tool names to CLI equivalents
- keep stable wrappers for:
  - paper search
  - arXiv search
  - citation search
  - claim graph
  - proof rigor check
  - latex compile
- decide which removed tools need lightweight CLI restoration

### 4. Prompt Parity

Acceptance criteria:

- prompts preserve canonical task logic and output expectations
- old tool references are translated consistently, not partially
- examples use commands that actually work in this environment

Checklist:

- audit all prompts changed in `docker-publish`
- rewrite old callable syntax to CLI-command syntax where needed
- remove stale references to unavailable tools
- ensure command examples use the correct interpreter/runtime assumptions

### 5. Broken Path Repairs

Fix before deeper parity work:

- math CLI wrappers must use canonical `math_workspace/` paths
- adversarial verifier nodes must instantiate successfully
- proof-rigor instructions must match actual CLI checker semantics
- backend preflight must validate all configured backends, not only the default

### 6. Regression Test Layer

Acceptance criteria:

- tests fail on current regressions and pass after repair
- tests encode canonical behavior, not implementation details

Suggested tests:

- claim graph CLI writes to `math_workspace/claim_graph.json`
- proof rigor CLI works with prompt-recommended usage
- adversarial math verifier node builds
- adversarial experiment verifier node builds
- prompt adapter emits runnable canonical commands
- per-agent backend overrides are preflight-validated

## Execution Order

1. Fix confirmed breakages that block parity work.
2. Repair canonical path and artifact mismatches.
3. Repair prompt/tool contract drift.
4. Audit and restore any missing lightweight tool wrappers required for parity.
5. Add regression tests for repaired parity surfaces.
6. Run a final branch-to-branch parity review against `codex/math-active`.

## Current Confirmed Regressions

- math CLI wrappers operate in workspace root instead of `math_workspace/`
- adversarial verifier node construction is broken
- rigorous verifier prompt references nonexistent or mismatched tool behavior
- prompt migration is incomplete and still contains old callable-tool syntax
- CLI command examples assume `python` even though this environment exposes `python3`
- runner preflight validates only the default backend

## Decision Rule

If a behavior in `docker-publish` differs from `codex/math-active` and does not
directly follow from the API->CLI substitution, it should be treated as a regression
until explicitly justified otherwise.
