import importlib.util
import sys
from pathlib import Path


_PUBLISHER_PATH = Path(__file__).resolve().parents[1] / "scripts" / "publish_results.py"
_SPEC = importlib.util.spec_from_file_location("publish_results_test_module", _PUBLISHER_PATH)
publish_results = importlib.util.module_from_spec(_SPEC)
assert _SPEC and _SPEC.loader
sys.modules.setdefault("publish_results_test_module", publish_results)
_SPEC.loader.exec_module(publish_results)


def test_find_existing_issue_post_returns_matching_filename(tmp_path):
    research_dir = tmp_path / "_research"
    research_dir.mkdir()
    (research_dir / "2026-04-01-some-idea.md").write_text(
        "---\nissue_number: 17\n---\n"
    )
    (research_dir / "2026-04-01-other-idea.md").write_text(
        "---\nissue_number: 18\n---\n"
    )

    found = publish_results._find_existing_issue_post(tmp_path, 17)

    assert found == "2026-04-01-some-idea.md"


def test_build_post_reuses_existing_issue_filename(tmp_path):
    workspace = tmp_path / "consortium_20260402_123456"
    workspace.mkdir()
    (workspace / "experiment_metadata.json").write_text(
        '{"task_preview": "Stable issue title", "cli_model": "gpt-5.4"}'
    )
    (workspace / "run_summary.json").write_text("{}")

    site_repo = tmp_path / "site"
    research_dir = site_repo / "_research"
    research_dir.mkdir(parents=True)
    existing = research_dir / "2026-04-01-stable-issue-title.md"
    existing.write_text("---\nissue_number: 42\n---\n")

    filename, content, pdf_source = publish_results.build_post(workspace, site_repo, issue_number=42)

    assert filename == existing.name
    assert 'issue_number: 42' in content
    assert pdf_source is None
