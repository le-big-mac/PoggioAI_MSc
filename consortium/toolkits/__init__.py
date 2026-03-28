"""
AI research toolkits for the consortium pipeline.

Toolkit groups:
  ideation/       - Idea generation, refinement, novelty checking, paper search
  experimentation/ - Experiment execution and idea standardization
  search/         - arXiv, web search, text inspection, visual QA
  filesystem/     - File editing, knowledge-base / repo management
  communication/  - User-facing tools (talk_to_user)
  writeup/        - LaTeX writing, citation, plotting, document analysis
  math/           - Claim graph, proof workspace, numerical verification
"""

from .ideation.paper_search_tool import PaperSearchTool
from .ideation.check_idea_novelty_tool import CheckIdeaNoveltyTool
from .experimentation.run_experiment_tool import RunExperimentTool
from .model_utils import get_raw_model

__all__ = [
    "PaperSearchTool",
    "CheckIdeaNoveltyTool",
    "RunExperimentTool",
    "get_raw_model",
]
