# AGENTS.md

## Repository
- Primary repo: jayden0701/nntrainer
- Primary working branch: t5gemma2

## Branching
- Always start from `t5gemma2`
- Create a new branch for each task
- Branch name format: `codex/<short-task-name>`

## Scope rules
- Do not modify unrelated files
- Avoid formatting-only changes
- Do not rename files unless necessary
- Do not change public APIs unless explicitly requested
- Keep diffs minimal and review-friendly

## Validation
- Prefer validating only the smallest relevant target first
- If build/test commands are needed, prefer:
  - `git submodule sync && git submodule update --init --depth 1`
  - `meson build`
  - `ninja -C build`
  - `cd build && ninja test`
- If a full build is too expensive, explain what was not run

## Testing
- If fixing a bug, add or update a regression test when practical
- If changing behavior, explain expected before/after behavior in the PR

## Output format
- Summarize:
  1. root cause
  2. files changed
  3. validation run
  4. remaining risks
