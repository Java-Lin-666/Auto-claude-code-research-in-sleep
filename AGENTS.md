# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## Modal

- gpu: modal # tells run-experiment to use

## What This Repo Is

ARIS (Auto-Codex-research-in-sleep) is a **skill-based research harness** — a collection of plain Markdown `SKILL.md` files that orchestrate the ML research lifecycle. There is no build system, no compiled code, and no runtime dependencies. The "code" is Markdown skills invoked via Codex slash commands.

## Skill Structure

Each skill lives at `skills/<name>/SKILL.md`. To understand any skill's behavior, read that file directly — `AGENT_GUIDE.md` is a routing index, not the specification.

Shared rules that apply across all skills:

- `skills/shared-references/reviewer-independence.md`
- `skills/shared-references/experiment-integrity.md`
- `skills/shared-references/effort-contract.md`
- `skills/shared-references/citation-discipline.md`
- `skills/shared-references/writing-principles.md`
- `skills/shared-references/venue-checklists.md`

## Invoking Skills

```
/skill-name "arguments" — key: value, key2: value2
```

Universal parameters accepted by every skill:

- `— effort: lite | balanced | max | beast` (default: `balanced`)
- `— human checkpoint: true | false` (default: `false`)
- `— AUTO_PROCEED: true | false` (default: `true`)

## Workflow Pipelines

Full pipeline: `/research-pipeline "direction"` chains W1 → W1.5 → W2 → W3.

Individual entry points:
| Workflow | Command | Input needed |
|----------|---------|-------------|
| W1: Idea Discovery | `/idea-discovery` | research direction |
| W1.5: Experiment Bridge | `/experiment-bridge` | `EXPERIMENT_PLAN.md` |
| W2: Auto Review | `/auto-review-loop` | paper + results |
| W3: Paper Writing | `/paper-writing` | `NARRATIVE_REPORT.md` |
| W4: Rebuttal | `/rebuttal` | paper + reviews |

## Artifact Contracts

Skills communicate through plain-text files in the working project directory:

| File                      | Written by        | Read by                           |
| ------------------------- | ----------------- | --------------------------------- |
| `IDEA_REPORT.md`          | idea-discovery    | experiment-bridge                 |
| `EXPERIMENT_PLAN.md`      | experiment-plan   | experiment-bridge                 |
| `EXPERIMENT_LOG.md`       | experiment-bridge | auto-review-loop, result-to-claim |
| `NARRATIVE_REPORT.md`     | auto-review-loop  | paper-writing                     |
| `paper/main.tex`          | paper-write       | paper-compile                     |
| `research-wiki/`          | research-wiki     | idea-creator, research-lit        |
| `.aris/meta/events.jsonl` | hooks (passive)   | meta-optimize                     |

## Cross-Model Protocol

- **Executor** (Codex): writes code, runs experiments, drafts papers
- **Reviewer** (GPT-5.4 / Gemini / GLM): critiques and scores — must be a different model family
- Pass file paths to the reviewer, never summaries or interpretations
- The executor must NOT judge its own eval code — reviewer audits directly

## Adding or Modifying Skills

1. Create or edit `skills/<name>/SKILL.md`
2. If the skill needs shared behavior, reference the appropriate file in `skills/shared-references/`
3. For Codex CLI variants, mirror changes in `skills/skills-codex/<name>/SKILL.md`
4. Update `AGENT_GUIDE.md` workflow table if the skill is a new entry point

## Tests

```bash
# Run the test suite
python -m pytest tests/
```

Check `tests/` for existing test patterns before adding new ones.
