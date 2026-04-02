"""CLI entry point for deterministic LaTeX/paper validation checks."""

from __future__ import annotations

import argparse
import json
import os
import sys

from ...supervision.paper_quality_validation import validate_paper_quality


def _find_existing(paths: list[str]) -> str | None:
    for path in paths:
        if os.path.exists(path):
            return path
    return None


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate final paper sources and compiled artifacts."
    )
    parser.add_argument(
        "--workspace",
        default=".",
        help="Workspace root containing paper_workspace/ (default: current directory)",
    )
    parser.add_argument(
        "--require-pdf",
        action="store_true",
        help="Fail validation when no final_paper.pdf is present.",
    )
    parser.add_argument(
        "--require-bib",
        action="store_true",
        help="Fail validation when no references.bib is present.",
    )
    args = parser.parse_args()

    workspace_dir = os.path.abspath(args.workspace)
    result = validate_paper_quality(workspace_dir)

    pdf_path = _find_existing([
        os.path.join(workspace_dir, "final_paper.pdf"),
        os.path.join(workspace_dir, "paper_workspace", "final_paper.pdf"),
    ])
    bib_path = _find_existing([
        os.path.join(workspace_dir, "references.bib"),
        os.path.join(workspace_dir, "paper_workspace", "references.bib"),
    ])

    errors = list(result.get("errors", []))
    warnings = list(result.get("warnings", []))

    if args.require_pdf and not pdf_path:
        errors.append("missing final_paper.pdf (root or paper_workspace/)")
    if args.require_bib and not bib_path:
        errors.append("missing references.bib (root or paper_workspace/)")

    payload = {
        **result,
        "pdf_found": pdf_path is not None,
        "pdf_path": pdf_path,
        "references_found": bib_path is not None,
        "references_path": bib_path,
        "is_valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
    }
    json.dump(payload, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0 if payload["is_valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
