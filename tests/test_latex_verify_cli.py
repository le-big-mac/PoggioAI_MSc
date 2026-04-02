from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def _run_cli(workspace: Path, *extra: str) -> subprocess.CompletedProcess[str]:
    cmd = [
        sys.executable,
        "-m",
        "consortium.toolkits.writeup.latex_verify_cli",
        "--workspace",
        str(workspace),
        *extra,
    ]
    return subprocess.run(cmd, capture_output=True, text=True)


def test_latex_verify_cli_passes_with_valid_tex_pdf_and_bib(tmp_path: Path) -> None:
    paper_ws = tmp_path / "paper_workspace"
    paper_ws.mkdir()
    (paper_ws / "final_paper.tex").write_text(
        "\\documentclass{article}\n\\begin{document}\nHello \\cite{smith2024}.\n\\end{document}\n",
        encoding="utf-8",
    )
    (paper_ws / "final_paper.pdf").write_bytes(b"%PDF-1.4\n")
    (paper_ws / "references.bib").write_text(
        "@article{smith2024,title={Example}}\n",
        encoding="utf-8",
    )

    result = _run_cli(tmp_path, "--require-pdf", "--require-bib")
    assert result.returncode == 0
    payload = json.loads(result.stdout)
    assert payload["is_valid"] is True
    assert payload["pdf_found"] is True
    assert payload["references_found"] is True


def test_latex_verify_cli_fails_on_placeholder_and_missing_pdf(tmp_path: Path) -> None:
    paper_ws = tmp_path / "paper_workspace"
    paper_ws.mkdir()
    (paper_ws / "final_paper.tex").write_text(
        "\\documentclass{article}\n\\begin{document}\n[cite: missing]\n\\end{document}\n",
        encoding="utf-8",
    )

    result = _run_cli(tmp_path, "--require-pdf", "--require-bib")
    assert result.returncode != 0
    payload = json.loads(result.stdout)
    assert payload["is_valid"] is False
    assert any("missing final_paper.pdf" in err for err in payload["errors"])
    assert any("missing references.bib" in err for err in payload["errors"])
    assert any("placeholder" in err for err in payload["errors"])
