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

## Environment Setup

One command: `bash setup.sh` — creates venv, installs everything including web UI deps.

If venv exists but deps are missing: `source .venv/bin/activate && pip install -e "packages/mirror_box[web,dashboard,cuda,dev]" -e "packages/aiware"`

## Launch Commands

| What | Command |
|------|---------|
| Web UI (full 3-panel) | `python -m mirror_box.scripts.web_chat --gpu` → port 8420 |
| Demo + Gradio dashboard | `mirror-box-demo --gpu --dashboard` |
| Demo (CLI only) | `mirror-box-demo --gpu --fast` |

**AI-first launch rule:** When user asks to "start", "launch", or "run" mirror-box, ALWAYS open the browser automatically after the server is listening. On WSL: `powershell.exe -Command "Start-Process 'http://localhost:8420'"`. Never just print a URL and wait.

## Rules

- TDD: write tests before implementation
- **Git**: Public repo. Push to `origin main`.
- The `aiware` package in `packages/aiware/` is the shared base library. Changes here may need to be coordinated with `~/crucible/` when that project starts.
