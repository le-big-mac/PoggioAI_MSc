# Contributing to consortium

This guide explains how to extend consortium: adding new agents, new tools, modifying prompts, updating CLI backend support, and running tests.

---

## Table of Contents

- [Development Setup](#development-setup)
- [Running Tests](#running-tests)
- [Adding a New Specialist Agent](#adding-a-new-specialist-agent)
- [Adding a New Tool to an Existing Toolkit](#adding-a-new-tool-to-an-existing-toolkit)
- [Modifying Agent Prompts](#modifying-agent-prompts)
- [Adding CLI Backend Support](#adding-cli-backend-support)
- [Code Style](#code-style)

---

## Development Setup

```bash
./scripts/bootstrap.sh researchlab minimal
conda activate researchlab
pip install -e . pytest pytest-cov mypy
# Verify at least one CLI agent tool is installed
which claude || which codex || which gemini
```

Verify everything is working:

```bash
pytest tests/ -v
python launch_multiagent.py --task "test" --dry-run
```

---

## Running Tests

```bash
# All tests
pytest tests/ -v

# Specific test file
pytest tests/test_budget.py -v

# With coverage
pytest tests/ --cov=consortium --cov-report=term-missing
```

Tests use `unittest.mock` to mock CLI agent subprocess calls — no real CLI invocations are made.

---

## Adding a New Specialist Agent

### Step 1: Create the agent file

Copy an existing simple agent (e.g., `consortium/agents/proofreading_agent.py`) as a template:

```python
# consortium/agents/my_new_agent.py
from langchain_core.tools import BaseTool
from .base_agent import create_specialist_agent

def get_tools(workspace_dir: str, ...) -> list[BaseTool]:
    """Return the tools this agent can use."""
    return [...]

def build_node(model, workspace_dir: str, ...):
    """Return a LangGraph node (callable) for this agent."""
    tools = get_tools(workspace_dir)
    instructions = "..."  # or import from prompts/
    return create_specialist_agent(model, tools, instructions)
```

### Step 2: Create a prompt instructions file

Add `consortium/prompts/my_new_instructions.py` with the agent's system prompt:

```python
MY_NEW_INSTRUCTIONS = """
You are a specialist agent responsible for...
"""
```

### Step 3: Register the agent in `graph.py`

In `consortium/graph.py`, import your agent and add it to `build_research_graph()`:

```python
from .agents.my_new_agent import build_node as build_my_new_node

# Inside build_research_graph(), add the node:
graph.add_node("my_new_agent", build_my_new_node(model, workspace_dir, ...))
```

### Step 4: Add to the appropriate stage group constant in `graph.py`

The pipeline is built from constant lists in `graph.py`. Add your agent to the correct group:

```python
# graph.py — stage group constants
DISCOVERY_STAGES = [...]          # ideation, literature_review, research_planner
EXPERIMENT_PIPELINE_STAGES = [...]  # experiment_literature, design, experimentation, verification, transcription
MATH_PIPELINE_STAGES = [...]       # math_literature, proposer, prover, verifiers, transcription
POST_TRACK_STAGES = [...]          # synthesis_lit_review, results_analysis, resource_prep, writeup, proofread, reviewer

# Add your stage name to the appropriate list, e.g.:
EXPERIMENT_PIPELINE_STAGES = [
    ...
    "my_new_agent",   # add at the right position within the group
    ...
]
```

### Step 5: Add stage aliases to `runner.py`

In `consortium/runner.py`, add entries to `_STAGE_ALIASES`:

```python
_STAGE_ALIASES = {
    ...
    "my_new": "my_new_agent",
    "my_new_agent": "my_new_agent",
}
```

### Step 6: Wire edges in `build_research_graph()` in `graph.py`

In the `build_research_graph()` function, add edges connecting your new node to the preceding and following nodes in the pipeline.

### Step 7: Write tests

Add tests in `tests/test_graph.py` verifying the new stage appears in `build_pipeline_stages()`.

---

## Adding a New Tool to an Existing Toolkit

### Step 1: Create the tool file

Place it in the appropriate toolkit directory:

```
consortium/toolkits/
  search/          ← search tools (ArXiv, web, text inspector)
  ideation/        ← idea generation and refinement
  experimentation/ ← experiment execution
  math/            ← math verification tools
  writeup/         ← LaTeX, citations, figures
  filesystem/      ← file I/O
```

Follow the LangChain `BaseTool` interface:

```python
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

class MyToolInput(BaseModel):
    query: str = Field(description="The input query")

class MyNewTool(BaseTool):
    name: str = "my_new_tool"
    description: str = "Does X given Y."
    args_schema: type[BaseModel] = MyToolInput

    def _run(self, query: str) -> str:
        # implementation
        return result
```

### Step 2: Register the tool in the relevant agent's `get_tools()`

```python
# In the agent that should use this tool:
from ..toolkits.my_toolkit.my_new_tool import MyNewTool

def get_tools(workspace_dir, ...):
    return [
        ...,
        MyNewTool(...),
    ]
```

---

## Modifying Agent Prompts

All agent system prompts live in `consortium/prompts/`:

| File | Agent |
|------|-------|
| `literature_review_instructions.py` | LiteratureReviewAgent |
| `persona_council_instructions.py` | PersonaCouncil |
| `brainstorm_instructions.py` | BrainstormAgent |
| `formalize_goals_instructions.py` | FormalizeGoalsAgent |
| `experimentation_instructions.py` | ExperimentationAgent |
| `results_analysis_instructions.py` | ResultsAnalysisAgent |
| `writeup_instructions.py` | WriteupAgent |
| `proofreading_instructions.py` | ProofreadingAgent |
| `reviewer_instructions.py` | ReviewerAgent |
| `resource_preparation_instructions.py` | ResourcePreparationAgent |
| `math_proposer_instructions.py` | MathProposerAgent |
| `math_prover_instructions.py` | MathProverAgent |
| `math_rigorous_verifier_instructions.py` | MathRigorousVerifierAgent |
| `math_empirical_verifier_instructions.py` | MathEmpiricalVerifierAgent |
| `math_literature_instructions.py` | MathLiteratureAgent |
| `proof_transcription_instructions.py` | ProofTranscriptionAgent |

**Tips for prompt modifications:**
- Keep the artifact contract section intact (agents must write specific files for validation gates to pass)
- Test prompt changes with `--no-counsel` and a short task to iterate quickly
- Large prompt changes may require adjusting the manager's stage-routing heuristics

---

## Adding CLI Backend Support

To add support for a new CLI agent tool (e.g., a new LLM provider's CLI):

### Step 1: Add a `CLIBackendSpec` entry

In `consortium/cli_completion.py` (or the CLI backend registry), register the new backend via `create_cli_backend_registry()`:

```python
# Add a new CLIBackendSpec for the provider
CLI_BACKENDS = {
    ...
    "my-new-cli": CLIBackendSpec(
        binary="my-new-cli",           # CLI executable name
        model_flag="--model",          # flag to specify model
        prompt_flag="--prompt",        # flag to pass prompt text
        default_model="my-model-v1",   # default model identifier
    ),
}
```

### Step 2: Add context limit to `base_agent.py`

In `consortium/agents/base_agent.py`:

```python
MODEL_CONTEXT_LIMITS = {
    ...
    "my-new-model": 128_000,
}
```

### Step 3: Add parameter filtering if needed

If the new CLI tool has non-standard flags, add a branch in `consortium/config.py`'s `filter_model_params()`.

### Step 4: Add CLI tool validation to `runner.py`

In `consortium/runner.py`'s CLI tool validation, add a check for the new binary:

```python
_CLI_TOOL_MAP = {
    ...
    "my-provider-prefix": "my-new-cli",  # maps model prefix to CLI binary name
}
```

### Step 5: Verify the tool is available

The new CLI tool must be installed and on PATH. No API keys are needed — CLI tools authenticate through their own subscriptions (e.g., Claude Max, ChatGPT Pro, Gemini Advanced).

---

## Code Style

- **Type hints**: Add type annotations to all new public functions
- **Docstrings**: Add docstrings to agent `build_node()` functions and tool classes
- **No real CLI calls in tests**: Mock all CLI agent subprocess calls with `unittest.mock.patch`
- **Imports**: Standard library → third-party → local (relative imports for intra-package)
- **Configuration**: New runtime options go in `args.py` (CLI) + `config.py` (YAML) with CLI overriding YAML
