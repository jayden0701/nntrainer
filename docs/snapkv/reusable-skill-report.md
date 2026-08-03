# Reusable paper-to-feature skill report

Date: 2026-08-03 KST

## Installed skill

The reusable personal skill is installed at:

```text
C:\Users\jayde\.codex\skills\implement-research-paper
```

Its workflow covers source identity/provenance, full paper reading, pinned
reference implementation research, evidence-versus-adaptation separation,
normative specification, repository integration mapping, risk-first planning,
pure-policy extraction, layered verification, independent review, stop rules,
and evidence-backed handoff.

The package contains:

- `SKILL.md`: trigger and end-to-end workflow;
- `references/workflow-checklists.md`: detailed extraction, integration,
  verification, and review checklists;
- `scripts/init_research_implementation_docs.py`: non-destructive scaffold for
  eight persistent analysis/decision/verification documents;
- `agents/openai.yaml`: user-facing metadata and default invocation prompt.

## Validation

- The scaffold script was run in clean scratch directories. First execution
  created an identity manifest plus all eight documents; the second skipped
  every existing file without overwriting it. A different feature/paper pair
  failed with exit code 2 before mutation. A bounded exclusive initialization
  lock serialized manifest and document publication; two concurrent
  same-identity processes both returned zero and produced exactly one
  non-empty copy of all nine files with no residual lock.
- The official `skill-creator` `quick_validate.py` check passed after the final
  edit: `Skill is valid!`.
- No extra README was placed inside the skill package.
- A fresh package re-audit approved the bounded-lock revision with no remaining
  actionable issue; the installed package contains exactly its four intended
  files and no `__pycache__` or `.pyc` artifact.

## Blind forward test

A fresh-context agent used the skill for an analysis-only Group Normalization
paper-to-nntrainer task. It wrote only under:

```text
C:\nntrainer\tmp\implement-research-paper-forward-test-20260803
```

The test intentionally supplied the wrong arXiv ID. The workflow independently
resolved that `1810.08459` is an unrelated optics paper, identified Group
Normalization as `1803.08494`, read/rendered the full 17-page ECCV publication,
pinned author and maintained references, mapped nntrainer integration seams,
and produced a normative 31-requirement specification, risk-first plan,
decision log, and verification ledger without editing production code.

The forward test exposed two instruction gaps. The skill was then updated to:

1. distinguish an unambiguous, explicitly recorded paper-ID correction from a
   conflict that must pause for user direction;
2. prescribe isolated role-based audit passes and honest labeling when
   independent subagents are unavailable.

A final package audit found three additional reuse risks, which were fixed:

3. source analysis may proceed under an unambiguous correction, but production
   edits now require confirmation;
4. an identity manifest prevents mixed feature/paper provenance, while a
   bounded initialization lock and exclusive writes prevent partial-file races;
5. every pinned reference must record its license and an explicit clean-room,
   conceptual-derivation, or copy-compatible decision.

These edits were revalidated with `quick_validate.py` and the independent
package re-audit.
