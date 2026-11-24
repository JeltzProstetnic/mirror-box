# Backlog — mirror-box

Task IDs use the `MIR-NN` prefix. States: `[ ]` open, `[>]` in-progress, `[x]` done.
Convention: `~/.claude/reference/backlog-convention.md`.

## Open

### Inbox intake 2026-08-24 — delivered by cfg-agent-fleet

2 item(s) moved verbatim from `cross-project/inbox.md` while this project had no active session. Each preamble names the original source and date.

- [ ] [P2] `MIR-1` **MG: "aIware and all its subprojects must be aware of this knowledge file, not only when I mention it explicitly.".** (P2 — always-load the FMT didactic-patterns registry, MG-directed 2026-07-31, re-filed from aIware WSL 2026-08-04 S284): **MG: "aIware and all its subprojects must be aware of this knowledge file, not only when I mention it explicitly."** The file is `~/aIware/.claude/knowledge/didactic-patterns.md` — the durable catalog of FMT teaching devices plus the corrections not to re-introduce (currently 30 patterns; the load-bearing ones for implementation work are **#18** *closure is an advantage in a fit window, never a necessity* and **#26** *the advantage scales with redeployed-model richness, NOT recursion depth*). **Asked:** add it to an unconditional "Always loaded" section in `~/mirror-box/CLAUDE.md` — the always-load set, not the triggered table. aIware made the same edit to its own manifest in S284; that is the reference pattern. Rationale: patterns get reused and gaps (aIware `AIW-95`) stay visible only if every aIware-family session carries the registry without a manual trigger. Source: aIware WSL 2026-08-04.  **[DONE simbook WSL 2026-08-06 — `Always loaded` section added to `~/simbook/CLAUDE.md` above the shared-knowledge block, unconditional, with the #15/#16 book-vs-paper routing spelled out. Also fixed a second manifest defect found while doing it: the shared-knowledge list cited `publication-workflow.md` at a path that does not exist.]**

- [ ] [P3] `MIR-2` **Replace ASCII architecture diagram in README.md with Mermaid.** Replace ASCII architecture diagram in README.md with Mermaid.

### Inbox intake 2026-11-25 — delivered by cfg-agent-fleet

- [ ] [P2] `MIR-3` **Fix the FMT defects in the copied crucible spec — or replace the copy with a pointer.** (from aIware S342, AIW-303) `papers/fmt-implementation-spec.md` is an identical copy of the crucible spec (same ~450 lines, differs only in whitespace), which carries claim-changing FMT defects — criticality treated as a condition, explicit = phenomenal without the loop, and others. Every crucible hit applies here at the same line numbers; `docs/design-spec.md:535` is also affected. Hits and replacement text: `~/aIware/docs/reviews/2026-10-19-crosscheck-outside-hits.md` sections 1 and 5. Coordinate with crucible: fix both copies, or replace this copy with a pointer to crucible's.

## Done
