"""
AI research toolkits for the consortium pipeline.

Toolkit groups:
  ideation/       - Paper search, novelty checking
  experimentation/ - Experiment execution
  search/         - arXiv paper fetching
  writeup/        - Citation search, LaTeX tools, document analysis
  math/           - Claim graph, proof workspace, numerical verification
"""

from .model_utils import get_raw_model

__all__ = [
    "get_raw_model",
]
