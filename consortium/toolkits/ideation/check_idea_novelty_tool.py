from __future__ import annotations

import json
import os
from typing import Any, Optional, Type

from langchain_core.tools import BaseTool
from pydantic import BaseModel, ConfigDict, Field

from consortium.cli_completion import cli_completion

from .paper_search_tool import _search_semantic_scholar


class CheckIdeaNoveltyToolInput(BaseModel):
    idea_json: str = Field(description="JSON string containing the idea to check")
    task_description: Optional[str] = Field(default=None, description="Description of the research task or domain")
    code: Optional[str] = Field(default=None, description="Experiment code for context")
    max_num_iterations: Optional[int] = Field(default=None, description="Maximum number of search iterations")
    base_dir: Optional[str] = Field(default=None, description="Base directory for loading context files")


class CheckIdeaNoveltyTool(BaseTool):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    name: str = "CheckIdeaNoveltyTool"
    description: str = "Assesses the novelty of a research idea by calling LLM and searching literature. Returns the idea with novelty field (True/False) and justification."
    args_schema: Type[BaseModel] = CheckIdeaNoveltyToolInput
    model_id: str = ""

    def __init__(self, model=None, **kwargs: Any):
        """
        Initialize the tool with access to the LLM model.

        Args:
            model: The LLM model to use for novelty checking
        """
        model_id = model if isinstance(model, str) else getattr(model, 'model', str(model)) if model else ""
        super().__init__(model_id=model_id, **kwargs)

    def _run(self, idea_json: str, task_description: str = "", code: str = "",
                max_num_iterations: int = 3, base_dir: str = "") -> str:
        """
        Check the novelty of a research idea by performing literature search and LLM analysis.

        Args:
            idea_json: JSON string containing the idea to check
            task_description: Task description for context
            code: Experiment code for context
            max_num_iterations: Maximum number of search iterations (default: 3)
            base_dir: Base directory for loading context files

        Returns:
            JSON string containing the idea with novelty assessment
        """
        try:
            # Load additional context if base_dir is provided
            if base_dir:
                import os
                try:
                    if not code and os.path.exists(os.path.join(base_dir, "experiment.py")):
                        with open(os.path.join(base_dir, "experiment.py"), "r") as f:
                            code = f.read()

                    if not task_description and os.path.exists(os.path.join(base_dir, "prompt.json")):
                        with open(os.path.join(base_dir, "prompt.json"), "r") as f:
                            prompt_data = json.load(f)
                            task_description = prompt_data.get("task_description", "")
                except Exception as e:
                    # Continue with provided parameters if file loading fails
                    pass

            # Parse the idea
            idea = json.loads(idea_json)

            if not self.model_id:
                return json.dumps({"error": "No model provided to CheckIdeaNoveltyTool"})

            # Create the novelty system message
            system_msg = f"""You are an ambitious AI PhD student who is looking to publish a paper that will contribute significantly to the field.
You have an idea and you want to check if it is novel or not. I.e., not overlapping significantly with existing literature or already well explored.
Be a harsh critic for novelty, ensure there is a sufficient contribution in the idea for a new conference or workshop paper.
You will be given access to the Semantic Scholar API, which you may use to survey the literature and find relevant papers to help you make your decision.
The top 10 results for any search query will be presented to you with the abstracts.

You will be given {max_num_iterations} to decide on the paper, but you do not need to use them all.
At any round, you may exit early and decide on the novelty of the idea.
Decide a paper idea is novel if after sufficient searching, you have not found a paper that significantly overlaps with your idea.
Decide a paper idea is not novel, if you have found a paper that significantly overlaps with your idea.

{task_description}
<experiment.py>
{code}
</experiment.py>
"""

            last_query_results = ""
            current_round = 1
            decision = "undecided"
            novelty_justification = ""

            # Iterative novelty checking process
            while current_round <= max_num_iterations and decision == "undecided":
                prompt = f'''Round {current_round}/{max_num_iterations}.
You have this idea:

"""
{json.dumps(idea, indent=2)}
"""

The results of the last query are (empty on first round):
"""
{last_query_results}
"""

Respond in the following format:

THOUGHT:
<THOUGHT>

RESPONSE:
```json
<JSON>
```

In <THOUGHT>, first briefly reason over the idea and identify any query that could help you make your decision.
If you have made your decision, add "Decision made: novel." or "Decision made: not novel." to your thoughts.

In <JSON>, respond in JSON format with the following fields:
- "Query": An optional search query to search the literature (e.g. attention is all you need). You must make a query if you have not decided this round.
- "Decision": A decision on the novelty of the idea. Either "decision made: novel", "decision made: not novel", or "undecided".

A query will work best if you are able to recall the exact name of the paper you are looking for, or the authors.
This JSON will be automatically parsed, so ensure the format is precise.
'''

                # Get LLM response via cli_completion
                response_content = cli_completion(prompt, system_prompt=system_msg)

                # Extract JSON from response
                if "```json" in response_content:
                    json_start = response_content.find("```json") + 7
                    json_end = response_content.find("```", json_start)
                    if json_end != -1:
                        response_json = response_content[json_start:json_end].strip()
                        response_data = json.loads(response_json)

                        decision = response_data.get("Decision", "undecided")
                        query = response_data.get("Query", "")

                        # Extract novelty justification from thoughts
                        if "THOUGHT:" in response_content:
                            thought_start = response_content.find("THOUGHT:") + 8
                            thought_end = response_content.find("RESPONSE:", thought_start)
                            if thought_end != -1:
                                novelty_justification += f"Round {current_round}: {response_content[thought_start:thought_end].strip()}\n\n"

                        # If undecided and query provided, search Semantic Scholar
                        if decision == "undecided" and query:
                            s2_key = os.environ.get("S2_API_KEY", "")
                            if s2_key:
                                try:
                                    papers = _search_semantic_scholar(
                                        query=query,
                                        result_limit=10,
                                        s2_api_key=s2_key,
                                    )
                                    if papers:
                                        summaries = []
                                        for p in papers:
                                            authors = ", ".join(
                                                a.get("name", "") for a in p.get("authors", [])[:3]
                                            )
                                            abstract = (p.get("abstract") or "")[:300]
                                            summaries.append(
                                                f"- {p.get('title', 'Unknown')} "
                                                f"({p.get('year', '?')}) by {authors}\n"
                                                f"  {abstract}"
                                            )
                                        last_query_results = (
                                            f"Searched for: '{query}'\n\n"
                                            f"Found {len(papers)} papers:\n\n"
                                            + "\n\n".join(summaries)
                                        )
                                    else:
                                        last_query_results = (
                                            f"Searched for: '{query}' — "
                                            "No papers found on Semantic Scholar."
                                        )
                                except Exception as search_err:
                                    last_query_results = (
                                        f"Searched for: '{query}' — "
                                        f"Search failed: {search_err}"
                                    )
                            else:
                                last_query_results = (
                                    f"Searched for: '{query}' — "
                                    "S2_API_KEY not set; Semantic Scholar search unavailable."
                                )

                        if decision.startswith("decision made:"):
                            break

                current_round += 1

            # Determine final novelty assessment
            is_novel = "novel" in decision.lower()

            # Add novelty assessment to the idea
            idea["novel"] = is_novel
            idea["novelty_decision"] = decision
            idea["novelty_justification"] = novelty_justification.strip()
            idea["novelty_checked_rounds"] = current_round - 1

            return json.dumps(idea, indent=2)

        except json.JSONDecodeError as e:
            return json.dumps({"error": f"Invalid JSON in idea_json: {str(e)}"})
        except Exception as e:
            return json.dumps({"error": f"Error checking novelty: {str(e)}"})

