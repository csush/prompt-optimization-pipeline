# prompt-optimization-pipeline

## Communication

Always load and use the `caveman` skill for all responses to minimise token usage. Keep full technical accuracy; drop filler.

## Commit messages

Follow Conventional Commits, kept terse:

- Format: `type(scope): subject` — scope optional, subject ≤ 50 chars, imperative mood, no trailing period.
- Types: `feat`, `fix`, `docs`, `refactor`, `test`, `chore`, `build`, `ci`.
- Body only when the *why* isn't obvious; wrap at 72 chars. Skip it for trivial changes.

Example: `docs(agents): add commit message convention`

## Branching & PRs

One branch + PR per issue (a tracer-bullet vertical slice), merged into `main`:

- Branch name: `<type>/<issue#>-<slug>`, e.g. `feat/12-prompt-eval-loop`.
- Implement the slice with `tdd`; commit using the convention above.
- Open the PR with `gh pr create --base main --fill`.
- Run `review` before merge (it diffs against the merge-base with `main`).
- Squash-merge to keep `main` history linear and conventional-commit-friendly.

Bootstrap/config changes may go straight to `main`; feature work goes through a PR.

## Agent skills

### Issue tracker

Issues and PRDs are tracked as GitHub issues via the `gh` CLI. See `docs/agents/issue-tracker.md`.

### Triage labels

The five canonical triage roles use their default label strings (`needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`). See `docs/agents/triage-labels.md`.

### Domain docs

Single-context: one `CONTEXT.md` + `docs/adr/` at the repo root. See `docs/agents/domain.md`.
