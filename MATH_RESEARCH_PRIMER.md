# Mathematical Deep Learning Research with consortium

A primer for using this multi-agent system to turn vague mathematical deep learning
intuitions into rigorous theory papers.

## What This System Does

You give it a rough idea -- something like "I think SGD implicitly compresses neural
network representations, and that's why it generalizes" -- and it:

1. Searches the mathematical DL theory literature
2. Formalizes your intuition into precise definitions, assumptions, and theorem statements
3. Builds a structured dependency graph of mathematical claims
4. Writes proof drafts for each claim
5. Audits the proofs for symbolic rigor
6. Stress-tests the claims with numerical counterexample searches
7. Writes a LaTeX paper with the accepted results

The system is **human-on-the-loop**: you launch it, optionally steer it mid-run, and
inspect the outputs. You are not in the loop for every decision.

---

## Architecture Overview

```
You (vague idea)
  |
  v
PersonaCouncil -------> evaluates research direction (3-persona debate)
  |
  |--- BrainstormAgent         generates research approaches
  |--- FormalizeGoalsAgent     structures goals for execution
  |--- MathProposerAgent      builds the formal claim graph (defs, lemmas, theorems)
  |--- MathProverAgent        writes structured proof drafts
  |--- MathRigorousVerifier   audits proofs for symbolic completeness
  |--- MathEmpiricalVerifier  numerical sanity checks + counterexample search
  |--- ExperimentationAgent   (optional) runs computational experiments
  |--- ResourcePrepAgent      organizes results for paper writing
  |--- WriteupAgent           produces LaTeX paper
  |--- ReviewerAgent          peer-reviews the paper
  |--- ProofreadingAgent      final quality pass
```

### The Two Pipelines

The system has two pipelines that work together:

**Math Pipeline** (enabled by `--enable-math-agents`):
```
IdeationAgent -> MathProposer -> MathProver -> RigorousVerifier -> EmpiricalVerifier
```

**Paper Pipeline** (always active):
```
(Experimentation) -> ResourcePreparation -> WriteupAgent -> ReviewerAgent -> ProofreadingAgent
```

The pipeline orchestrates both tracks via a direct-wired LangGraph workflow, feeding accepted math claims into the paper pipeline.

---

## The Math Pipeline in Detail

### 1. Discovery Phase -- From Vague Idea to Mathematical Program

The persona council and brainstorm agents take your informal intuition and produce a structured mathematical research program.
They search ArXiv for DL theory papers, identify the right mathematical framework
(functional analysis, measure theory, optimization theory, etc.), and draft:

- Precise definitions with explicit quantifiers and domains
- Labeled assumptions (A1, A2, ...) that are falsifiable and motivated
- Theorem statements with clear hypothesis-conclusion structure
- Proof strategy outlines identifying key techniques needed

**Output**: A structured description that the MathProposerAgent can turn into a
formal claim graph.

### 2. MathProposerAgent -- The Claim Graph

Transforms the ideation output into `math_workspace/claim_graph.json`, a
dependency-structured DAG of mathematical claims. Each claim has:

- **id**: e.g., `D_effective_dim_1`, `L_concentration_1`, `T_generalization_main`
- **type**: definition / lemma / proposition / theorem / corollary
- **statement**: precise mathematical statement
- **assumptions**: labeled list
- **depends_on**: list of claim IDs this claim requires
- **status**: tracks progress through the verification pipeline
- **must_accept**: whether this claim is required for the main result
- **tags**: area and type tags

The claim graph is the **single source of truth** for the mathematical content.
All downstream agents read from and write back to it.

**Naming convention**:
- `D_<slug>` for definitions
- `L_<slug>_<k>` for lemmas
- `T_<slug>` for theorems
- `C_<slug>` for corollaries

### 3. MathProverAgent -- Proof Construction

Picks up claims with status `proposed` and writes structured proof drafts in
`math_workspace/proofs/<claim_id>.md`. Each proof file has mandatory sections:

- Claim (restated)
- Assumptions
- Dependencies
- Definitions / Notation
- Proof Plan
- Detailed Steps (at least 6 for core claims, 4 for simpler lemmas)
- Edge Cases / Domain Checks
- Conclusion
- Open Issues

Sets claim status to `proved_draft` when done. Does not certify rigor -- that's
the verifier's job.

**Standard lemma fast path**: For well-known results (Cauchy-Schwarz, Jensen's
inequality, etc.), the prover can reference `math_workspace/lemma_library.md`
instead of re-deriving everything.

### 4. MathRigorousVerifierAgent -- Symbolic Audit

Audits `proved_draft` claims using a strict checklist:

1. Statement precision (quantifiers, domains, constants)
2. Assumptions explicit and actually used
3. Dependencies referenced by claim ID, dependency gate satisfied
4. Dimensions/shapes consistent
5. Inequalities justified, constants tracked
6. Probability/calculus conditions explicit
7. Edge/domain constraints explicit
8. No hidden nontrivial jumps
9. No placeholders or open issues

Also runs `math_proof_rigor_checker_tool` in strict mode.

Writes structured audit logs to `math_workspace/checks/<claim_id>.jsonl`.
Upgrades status to `verified_symbolic` only on full pass.

### 5. MathEmpiricalVerifierAgent -- Numerical Stress Testing

Takes `verified_symbolic` claims and stress-tests them numerically:

- Encodes claims as testable expressions (equalities, inequalities)
- Runs randomized checks across at least 3 regimes:
  - Typical/central values
  - Small/edge cases
  - Large/edge cases
- Each regime uses 64+ trials
- Any counterexample is taken seriously

Upgrades status to `verified_numeric` on pass, or demotes back to `proved_draft`
with counterexample evidence.

### 6. Acceptance Gate

A claim is set to `accepted` only when ALL of:
- Status is `verified_numeric`
- Proof file exists in `math_workspace/proofs/`
- Symbolic audit record shows `pass`
- Numeric evidence shows `pass` or explicit `waived`
- All dependency claims are also `accepted`

Only `accepted` claims appear as derived results in the final paper.
Non-accepted claims are labeled as conjectures.

### Claim Status Progression

```
proposed
  |
  v  (MathProver)
proved_draft
  |
  v  (RigorousVerifier)
verified_symbolic
  |
  v  (EmpiricalVerifier)
verified_numeric
  |
  v  (Manager acceptance gate)
accepted
```

A claim can also be demoted back to `proved_draft` at any stage if issues are found,
or `rejected` with concrete evidence.

---

## The Math Tools

Four specialized tools power the math agents:

| Tool | Purpose |
|------|---------|
| `math_claim_graph_tool` | CRUD operations on the claim graph: init, add/update/get claims, manage dependencies, validate DAG structure, manage lemma library |
| `math_proof_workspace_tool` | Create proof templates, read/write proofs, append check logs in `math_workspace/` |
| `math_proof_rigor_checker_tool` | Automated symbolic rigor checking (strict mode for formal audit) |
| `math_numerical_claim_verifier_tool` | Randomized numerical testing of mathematical claims with multi-regime support |

---

## How to Run It

### Prerequisites

Everything is already installed in the `researchlab` conda environment.

```bash
conda activate researchlab
cd consortium
```

### Basic Math Research Run

```bash
python launch_multiagent.py \
  --enable-math-agents \
  --reasoning-effort high \
  --no-log-to-files \
  --task "YOUR IDEA HERE"
```

### Key Flags

| Flag | What it does |
|------|-------------|
| `--enable-math-agents` | Activates the math pipeline (MathLiterature, Proposer, Prover, RigorousVerifier, EmpiricalVerifier, ProofTranscription) |
| `--reasoning-effort high` | Maximum thinking depth for GPT-5 (use `high` for math) |
| `--no-log-to-files` | Print output to terminal instead of log files |
| `--enforce-paper-artifacts` | Require `final_paper.tex` before termination |
| `--require-pdf` | Also require compiled `final_paper.pdf` |
| `--enforce-editorial-artifacts` | Full editorial quality gate (style guide, review verdict, etc.) |
| `--manager-max-steps 40` | Give the manager more steps for complex proofs |
| `--resume <path>` | Continue from an existing workspace |

### Example: Full Quality Run

```bash
python launch_multiagent.py \
  --enable-math-agents \
  --enforce-paper-artifacts \
  --enforce-editorial-artifacts \
  --require-pdf \
  --reasoning-effort high \
  --manager-max-steps 40 \
  --no-log-to-files \
  --task "I suspect that for two-layer ReLU networks trained with gradient descent
on isotropic Gaussian data, the learned features converge to a low-rank subspace
whose dimension scales as O(sqrt(n/d)) where n is sample size and d is input
dimension. Formalize this, prove it under reasonable assumptions on the target
function, and verify numerically."
```

### Providing Context Documents

Put your notes, reference papers, or partial drafts in the workspace:

```bash
mkdir -p results/my_project/inputs
cp my_notes.md results/my_project/inputs/
cp reference_paper.pdf results/my_project/inputs/

python launch_multiagent.py \
  --enable-math-agents \
  --resume results/my_project \
  --task "Read inputs/*.md and inputs/*.pdf. Use them as context to formalize
  the mathematical ideas described there. Build a claim graph and prove the
  main theorems."
```

---

## Crafting Good Task Descriptions

The `--task` string is the single most important input. A good task for math
DL research should include:

1. **The informal intuition** (1-2 paragraphs of what you believe is true and why)
2. **The type of result you want** (bound, convergence theorem, approximation result,
   characterization, impossibility result)
3. **Mathematical framework preference** (if you have one)
4. **Scope** (what to focus on, what to defer)

### Good Example

> "I have an intuition that the information bottleneck principle explains why deep
> networks generalize: intermediate layers compress the input while preserving
> task-relevant information, and this compression is driven by the noise in SGD.
> Formalize this for two-layer networks with ReLU activations trained on
> classification tasks. I want a bound on the mutual information between hidden
> representations and the input that decreases during training, and a theorem
> connecting this compression rate to generalization gap. Use information-theoretic
> tools (data processing inequality, rate-distortion theory). Start with the
> Gaussian input case."

### Weak Example

> "Prove something about neural networks and generalization."

The more mathematical context you provide, the better the formalization will be.

---

## Steering a Running Job

The system listens on `127.0.0.1:5001` for mid-run steering. From another terminal:

```bash
nc 127.0.0.1 5001
```

Then type:
1. `interrupt` (first line)
2. Your instruction (e.g., "The proof of Lemma 2 is using a bound that's too loose.
   Try using Bernstein's inequality instead of Hoeffding.")
3. Press Enter twice (two empty lines)
4. `m` for modification or `n` for new task

This pauses at the next step boundary, injects your instruction, and resumes.
It does **not** restart the run.

---

## Inspecting Outputs

After a run, the workspace (in `results/consortium_<timestamp>/`) contains:

```
results/consortium_YYYYMMDD_HHMMSS/
  math_workspace/
    claim_graph.json          <-- the full claim DAG
    proofs/
      T_main_theorem.md       <-- proof drafts
      L_concentration_1.md
      ...
    checks/
      T_main_theorem.jsonl    <-- audit + numeric check logs
      L_concentration_1.jsonl
      ...
    lemma_library.md          <-- standard results index
  paper_workspace/
    final_paper.tex           <-- the paper
    final_paper.pdf
    references.bib
    *.tex                     <-- individual sections
  working_idea.json           <-- the formalized research idea
  past_ideas_and_results.md   <-- iteration history
```

### What to Review First

1. **`claim_graph.json`** -- Check the theorem statements, assumptions, and which
   claims are accepted vs. still proposed
2. **`proofs/<claim_id>.md`** -- Read the actual proof drafts, especially the
   "Open Issues" sections
3. **`checks/<claim_id>.jsonl`** -- See what the verifiers found (any failures,
   counterexamples, or waived checks)
4. **`final_paper.pdf`** -- The assembled paper

---

## Configuration

### Model Selection (`.llm_config.yaml`)

Currently configured to use Claude Code CLI (`claude`) as the default backend.
For math-heavy work, `claude-opus-4-6` is recommended. Alternatives:

- `codex` backend with `gpt-5.4` -- good for multi-step deduction
- `gemini` backend with `gemini-3-pro-preview` -- large context for long proofs

### Budget (`.llm_config.yaml`)

Math research with proofs requires many CLI agent invocations. The default
limits are 100 invocations and 4 hours of wall-clock time. For a full paper
with proofs, consider raising:

```yaml
budget:
  max_invocations: 200
  max_wall_clock_seconds: 28800   # 8 hours
```

### CLI Tools

Requires at least one CLI agent tool installed: `claude`, `codex`, or `gemini`.
No API keys needed — agents run locally via subscription plans.

---

## Tips for Mathematical Research

1. **Start with ideation only**: Run with a low `--manager-max-steps` (e.g., 10)
   to get the claim graph, then review it yourself before committing to full proofs.

2. **Provide reference papers**: Put key papers in `inputs/` so the system can
   extract notation conventions and proof techniques from them.

3. **Use steering for proof guidance**: If you know the right proof technique
   (e.g., "use a PAC-Bayes bound with a data-dependent prior"), inject it via
   the steering mechanism rather than hoping the system discovers it.

4. **Iterate**: Run ideation + claim construction, review, then resume with
   `--resume` and a refined task. Multi-session refinement often beats a single
   long run.

5. **Check the claim graph early**: The claim graph is the skeleton of your paper.
   If the definitions or theorem statements are wrong, everything downstream
   will be wrong too.

6. **Leverage the lemma library**: For standard results, the system can cite
   them from `lemma_library.md` instead of re-proving them. You can pre-populate
   this file with results you know you'll need.

---

## Quick Reference: Common Commands

```bash
# Activate the environment
conda activate researchlab

# Simple math research run
python launch_multiagent.py \
  --enable-math-agents \
  --reasoning-effort high \
  --no-log-to-files \
  --task "YOUR IDEA"

# Resume from a previous run
python launch_multiagent.py \
  --enable-math-agents \
  --resume results/consortium_YYYYMMDD_HHMMSS/ \
  --task "Continue proving the remaining lemmas"

# Full quality pipeline with paper output
python launch_multiagent.py \
  --enable-math-agents \
  --enforce-paper-artifacts \
  --require-pdf \
  --reasoning-effort high \
  --manager-max-steps 40 \
  --no-log-to-files \
  --task "YOUR IDEA"

# Steer a running job (from another terminal)
nc 127.0.0.1 5001

# Check preflight
python scripts/preflight_check.py --with-docs --with-web --with-experiment
```
