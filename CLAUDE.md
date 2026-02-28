# Mirror Box — Design 16 Pseudo-AC Implementation

Public repo: `JeltzProstetnic/mirror-box`

## Key Files

| File | Purpose |
|------|---------|
| `session-context.md` | Current session state — read first |
| `papers/fmt-implementation-spec.md` | FMT implementation reference (derived from published theory) |
| `docs/design-spec.md` | Design 16 specification |
| `docs/dev-plan.md` | Development plan |

## Project Facts

- **Phase**: Active development (Design 16 — pseudo-AC quick-win)
- **Tech stack**: Python 3.11+, pytest, TDD
- **Coverage target**: 80%+ on core components
- **License**: MIT
- **Key concepts**: Four Models (IWM->EWM, ISM->ESM), virtual qualia, approximated criticality
- **Related**: `~/crucible/` (Design 15, the real AC implementation), `~/aIware/` (theory papers)

## Knowledge Loading

| Domain | File | Load when... |
|--------|------|-------------|
| Software Development | `~/.claude/domains/software-development/tdd-protocol.md` | Writing code or tests |

## Rules

- TDD: write tests before implementation
- **Git**: Public repo. Push to `origin main`.
- The `aiware` package in `packages/aiware/` is the shared base library. Changes here may need to be coordinated with `~/crucible/` when that project starts.
