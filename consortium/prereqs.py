"""
LaTeX prerequisite detection helpers.

Compatibility surface preserved for tests and utility scripts. The main runner
uses an inline tectonic check, but these helpers remain useful for preflight
and environment diagnostics.
"""

import os
import shutil


def resolve_executable(tool_name, env_var, extra_candidates=None):
    """Resolve an executable from env override, candidate paths, or PATH."""
    override = os.getenv(env_var, "").strip()
    if override:
        candidate = os.path.expanduser(override)
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate, None
        return None, f"{env_var} points to a missing/non-executable file: {override}"

    for candidate in extra_candidates or []:
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate, None

    found = shutil.which(tool_name)
    if found:
        return found, None
    return None, f"{tool_name} not found on PATH"


def check_latex_prereqs():
    """Detect pdflatex and bibtex for compatibility with legacy callers."""
    pdflatex_path, pdflatex_err = resolve_executable(
        tool_name="pdflatex",
        env_var="CONSORTIUM_PDFLATEX_PATH",
        extra_candidates=["/Library/TeX/texbin/pdflatex"],
    )
    bibtex_path, bibtex_err = resolve_executable(
        tool_name="bibtex",
        env_var="CONSORTIUM_BIBTEX_PATH",
        extra_candidates=["/Library/TeX/texbin/bibtex"],
    )
    if pdflatex_err or bibtex_err:
        issues = []
        if pdflatex_err:
            issues.append(f"- pdflatex: {pdflatex_err}")
        if bibtex_err:
            issues.append(f"- bibtex: {bibtex_err}")
        return None, None, "LaTeX prerequisites are required.\n" + "\n".join(issues)
    return pdflatex_path, bibtex_path, None

