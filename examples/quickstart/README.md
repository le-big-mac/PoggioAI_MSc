# Quickstart Example: Batch Normalization & Spectral Regularization

This is the recommended first run for new users. It demonstrates the full consortium pipeline on a focused, well-scoped research question in mathematical ML theory.

## What This Example Does

The task asks consortium to:
1. Search ArXiv for literature on batch normalization and spectral norms
2. Synthesize a literature review
3. Plan a research investigation
4. Propose a minimal experiment design
5. Write a 4-page paper draft in Markdown

It is designed to be **fast and self-contained** — no LaTeX installation, no GPU, no web crawl needed.

## Prerequisites

- Python environment set up (`./scripts/bootstrap.sh researchlab minimal`)
- At least one CLI agent tool installed (`claude`, `codex`, or `gemini`) — verify with `which claude`

## Running It

From the repo root:

```bash
# Validate your setup first (free)
python launch_multiagent.py --task "test" --dry-run

# Run the quickstart
python launch_multiagent.py \
  --task "$(cat examples/quickstart/task.txt)" \
  --output-format markdown \
  --no-counsel \
  --no-log-to-files
```

## Expected Cost & Time

All LLM costs are **included in your CLI tool subscription** (Claude Max, ChatGPT Pro, Gemini Advanced, etc.) — no per-token charges.

| CLI Tool | Estimated Time |
|---|---|
| claude (default) | 20–35 min |
| codex | 25–45 min |
| gemini | 15–25 min |

## Expected Outputs

After the run, look in `results/consortium_<timestamp>/`:

```
results/consortium_<timestamp>/
├── final_paper.md              ← Main output: the generated paper draft
├── paper_workspace/
│   ├── literature_review.pdf   ← Synthesized literature review
│   ├── research_plan.pdf       ← Research plan
│   └── references.bib          ← ArXiv citations
├── run_summary.json            ← Invocation and timing summary
└── cli_budget_state.json       ← Invocation and wall-clock tracking
```

The `final_paper.md` should contain:
- Abstract with clear hypothesis
- Introduction with motivation
- Related work section citing ~5–15 relevant ArXiv papers
- Theoretical background on batch normalization and spectral norms
- Proposed experiment outline
- Conclusion and future directions

## Reference Outputs

The `expected_outputs/` directory contains annotated samples showing what typical outputs look like. Your outputs will differ (different papers, different phrasing) but should have the same structure.

## Interpreting the Results

**Good signs:**
- `final_paper.md` is 1,500–4,000 words
- Literature review cites real ArXiv papers (check the DOIs/URLs)
- Research plan mentions concrete experimental steps
- `cli_budget_state.json` shows invocations completed normally

**Signs something went wrong:**
- `final_paper.md` is very short (<500 words) → the pipeline may have hit the budget cap or stalled
- No citations → ArXiv search may have failed (check the log for errors)
- `STATUS.txt` says `INCOMPLETE` → run `python launch_multiagent.py --resume results/consortium_<timestamp>/` to continue

## Next Steps

Once you're comfortable with the quickstart:

1. **Try your own research question**: Replace `task.txt` content with your topic
2. **Enable LaTeX output**: Drop `--output-format markdown` after installing `./scripts/bootstrap.sh researchlab latex`
3. **Enable math agents**: Add `--enable-math-agents` for theorem formalization
4. **Run a multi-stage campaign**: See `OpenClaw_Use_Guide.md`
5. **Browse past runs**: `python launch_multiagent.py --list-runs`
