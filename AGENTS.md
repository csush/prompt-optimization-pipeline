# prompt-optimization-pipeline

## Commit messages

Follow Conventional Commits, kept terse:

- Format: `type(scope): subject` — scope optional, subject ≤ 50 chars, imperative mood, no trailing period.
- Types: `feat`, `fix`, `docs`, `refactor`, `test`, `chore`, `build`, `ci`.
- Body only when the *why* isn't obvious; wrap at 72 chars. Skip it for trivial changes.

Example: `docs(agents): add commit message convention`

## Agent skills

### Issue tracker

Issues and PRDs are tracked as GitHub issues via the `gh` CLI. See `docs/agents/issue-tracker.md`.

### Triage labels

The five canonical triage roles use their default label strings (`needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`). See `docs/agents/triage-labels.md`.

### Domain docs

Single-context: one `CONTEXT.md` + `docs/adr/` at the repo root. See `docs/agents/domain.md`.
