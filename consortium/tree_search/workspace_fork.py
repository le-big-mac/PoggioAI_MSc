"""Copy-on-write workspace forking for tree search branches.

Provides the same sandbox primitives used by counsel mode (populate, merge,
retarget tools) but generalised for tree-search branch isolation.  Both
counsel and tree-search share these helpers.
"""

from __future__ import annotations

import os
import shutil
import time
from typing import Any, List


# Patterns to skip when copying a workspace into a branch.
SKIP_PATTERNS = ("counsel_sandboxes", "tree_branches", "*.db", "*.lock")


# ---------------------------------------------------------------------------
# Populate / merge
# ---------------------------------------------------------------------------

def populate_branch(
    workspace_dir: str,
    branch_dir: str,
    *,
    extra_skip: tuple[str, ...] = (),
    max_attempts: int = 3,
) -> None:
    """Copy *workspace_dir* into *branch_dir*, skipping heavy/transient artifacts.

    Semantics are identical to ``counsel._populate_sandbox`` but also skip
    ``tree_branches/`` to avoid recursive nesting.
    """
    if os.path.exists(branch_dir):
        shutil.rmtree(branch_dir)

    skip = SKIP_PATTERNS + extra_skip

    for attempt in range(max_attempts):
        try:
            shutil.copytree(
                workspace_dir,
                branch_dir,
                ignore=shutil.ignore_patterns(*skip),
            )
            return
        except InterruptedError:
            if attempt == max_attempts - 1:
                raise
            if os.path.exists(branch_dir):
                shutil.rmtree(branch_dir, ignore_errors=True)
            time.sleep(0.2 * (2 ** attempt))


def merge_branch(
    branch_dir: str,
    workspace_dir: str,
    *,
    skip_dirs: tuple[str, ...] = ("counsel_sandboxes", "tree_branches"),
) -> None:
    """Copy all files from *branch_dir* into *workspace_dir*.

    Last writer wins on conflicts — identical semantics to
    ``counsel._merge_sandbox``.
    """
    for root, dirs, files in os.walk(branch_dir):
        dirs[:] = [d for d in dirs if d not in skip_dirs]
        for fname in files:
            src = os.path.join(root, fname)
            rel = os.path.relpath(src, branch_dir)
            dst = os.path.join(workspace_dir, rel)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            try:
                shutil.copy2(src, dst)
            except Exception as exc:
                print(f"[tree_search] merge warning: failed to copy {rel}: {exc}")


# ---------------------------------------------------------------------------
# Tool retargeting
# ---------------------------------------------------------------------------

def retarget_tools(
    original_tools: List[Any],
    new_workspace: str,
) -> List[Any]:
    """Clone each tool so that workspace/working-dir paths point to *new_workspace*.

    Falls back to the original tool when re-instantiation fails.
    Semantics match ``counsel._sandbox_tools``.
    """
    result: list[Any] = []
    for tool in original_tools:
        try:
            cls = type(tool)
            kwargs: dict[str, Any] = {}
            for attr in (
                "working_dir",
                "workspace_dir",
                "model",
                "model_name",
                "authorized_imports",
                "allow_accepted_transition",
            ):
                if hasattr(tool, attr):
                    if attr in ("working_dir", "workspace_dir"):
                        kwargs[attr] = new_workspace
                    else:
                        kwargs[attr] = getattr(tool, attr)
            result.append(cls(**kwargs) if kwargs else tool)
        except Exception:
            result.append(tool)
    return result


# ---------------------------------------------------------------------------
# Convenience: fork + id-based directory
# ---------------------------------------------------------------------------

def fork_workspace(
    workspace_dir: str,
    branch_id: str,
    *,
    extra_skip: tuple[str, ...] = (),
) -> str:
    """Create a branch workspace under ``<workspace>/tree_branches/<branch_id>/``.

    Returns the absolute path of the new branch workspace.
    """
    branch_dir = os.path.join(workspace_dir, "tree_branches", branch_id)
    populate_branch(workspace_dir, branch_dir, extra_skip=extra_skip)
    return branch_dir


def promote_branch(branch_dir: str, workspace_dir: str) -> None:
    """Merge the winning branch back into the canonical workspace."""
    merge_branch(branch_dir, workspace_dir)
