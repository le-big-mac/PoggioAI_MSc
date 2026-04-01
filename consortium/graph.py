"""
LangGraph research pipeline — directly wired multi-phase workflow.

This graph replaces the manager hub-and-spoke loop with:
1. Discovery: ideation -> literature review -> research planner
2. Parallel execution: theory and experiment tracks in parallel
3. Synthesis loop: merge -> synthesis literature review -> results analysis
4. Paper production: resource preparation -> writeup -> proofreading -> reviewer
"""

from __future__ import annotations

import json
import os
from typing import Any, Callable, List, Optional

from langgraph.graph import END, StateGraph
from langgraph.types import Send

from .agents import (
    build_experiment_design_node,
    build_experiment_literature_node,
    build_experiment_transcription_node,
    build_experiment_verification_node,
    build_experimentation_node,
    build_literature_review_node,
    build_math_empirical_verifier_node,
    build_math_literature_node,
    build_math_proposer_node,
    build_math_prover_node,
    build_math_rigorous_verifier_node,
    build_proof_transcription_node,
    build_proofreading_node,
    build_resource_preparation_node,
    build_results_analysis_node,
    build_reviewer_node,
    build_track_merge_node,
    build_writeup_node,
    build_brainstorm_node,
    build_formalize_goals_node,
    build_formalize_results_node,
    build_research_plan_writeup_node,
)
from .milestone_report import (
    generate_milestone_report,
    wait_for_human_response,
)
from .pdf_summary import with_pdf_summary
from .state import ResearchState
from .workflow_utils import (
    classify_review_fixes,
    followup_decision_requires_loop,
    run_intermediate_validation,
    run_validation_gates,
    safe_int,
)

MATH_PIPELINE_STAGES = [
    "math_literature_agent",
    "math_proposer_agent",
    "math_prover_agent",
    "math_rigorous_verifier_agent",
    "math_empirical_verifier_agent",
    "proof_transcription_agent",
]

EXPERIMENT_PIPELINE_STAGES = [
    "experiment_literature_agent",
    "experiment_design_agent",
    "experimentation_agent",
    "experiment_verification_agent",
    "experiment_transcription_agent",
]

# ---------------------------------------------------------------------------
# V2 pipeline stage rosters
# ---------------------------------------------------------------------------

V2_PRE_TRACK_STAGES = [
    "persona_council",
    "literature_review_agent",
    "brainstorm_agent",
    "formalize_goals_entry",
    "formalize_goals_agent",
    "research_plan_writeup_agent",
]

V2_POST_TRACK_STAGES = [
    "formalize_results_agent",
    "resource_preparation_agent",
    "writeup_agent",
    "proofreading_agent",
    "reviewer_agent",
]


def build_pipeline_stages_v2(enable_math_agents: bool) -> list[str]:
    stages = list(V2_PRE_TRACK_STAGES)
    if enable_math_agents:
        stages.extend(MATH_PIPELINE_STAGES)
    stages.extend(EXPERIMENT_PIPELINE_STAGES)
    stages.extend(V2_POST_TRACK_STAGES)
    return stages


# ---------------------------------------------------------------------------
# Quick-pass pipeline (--quick-pass)
# ---------------------------------------------------------------------------

QUICK_PIPELINE_STAGES = [
    "persona_council",
    "literature_review_agent",
    "brainstorm_agent",
    "formalize_goals_entry",
    "formalize_goals_agent",
    "research_plan_writeup_agent",
]


def build_pipeline_stages_quick() -> list[str]:
    return list(QUICK_PIPELINE_STAGES)


def _read_file_for_verdict(path: str, max_chars: int = 50000) -> str:
    """Read a file for the quick verdict assembler, returning '' on failure."""
    try:
        with open(path) as f:
            return f.read(max_chars)
    except (OSError, UnicodeDecodeError):
        return ""


def _latex_to_markdown(tex: str) -> str:
    """Best-effort LaTeX → Markdown conversion for display on a website."""
    import re as _re

    # Strip preamble (everything before \begin{document}) and \end{document}
    m = _re.search(r"\\begin\{document\}(.*?)(?:\\end\{document\})?$", tex, _re.DOTALL)
    if m:
        tex = m.group(1)

    # Remove \maketitle, \tableofcontents, \newpage
    tex = _re.sub(r"\\(?:maketitle|tableofcontents|newpage|clearpage)\b", "", tex)

    # Sections → Markdown headings
    tex = _re.sub(r"\\section\*?\{([^}]*)\}", r"## \1", tex)
    tex = _re.sub(r"\\subsection\*?\{([^}]*)\}", r"### \1", tex)
    tex = _re.sub(r"\\subsubsection\*?\{([^}]*)\}", r"#### \1", tex)
    tex = _re.sub(r"\\paragraph\*?\{([^}]*)\}", r"**\1**", tex)

    # Text formatting
    tex = _re.sub(r"\\textbf\{([^}]*)\}", r"**\1**", tex)
    tex = _re.sub(r"\\textit\{([^}]*)\}", r"*\1*", tex)
    tex = _re.sub(r"\\emph\{([^}]*)\}", r"*\1*", tex)
    tex = _re.sub(r"\\texttt\{([^}]*)\}", r"`\1`", tex)
    tex = _re.sub(r"\\underline\{([^}]*)\}", r"\1", tex)

    # Citations → (Author, Year) style
    tex = _re.sub(r"\\citep?\{([^}]*)\}", r"(\1)", tex)
    tex = _re.sub(r"\\citet\{([^}]*)\}", r"\1", tex)

    # Lists: \begin{itemize}...\end{itemize}, \begin{description}...\end{description}
    tex = _re.sub(r"\\begin\{(?:itemize|enumerate|description)\}(?:\[.*?\])?", "", tex)
    tex = _re.sub(r"\\end\{(?:itemize|enumerate|description)\}", "", tex)
    tex = _re.sub(r"\\item\s*\[([^\]]*)\]\s*", r"- **\1** ", tex)
    tex = _re.sub(r"\\item\s*", "- ", tex)

    # Math environments: keep $...$ and $$...$$ as-is (MathJax renders them)
    # Convert \[ \] to $$ $$
    tex = tex.replace("\\[", "$$").replace("\\]", "$$")

    # Theorem-like environments → bold header + content
    for env in ("theorem", "lemma", "proposition", "definition", "remark", "corollary", "conjecture"):
        tex = _re.sub(
            rf"\\begin\{{{env}\}}(?:\[([^\]]*)\])?",
            lambda m, e=env: f"\n**{e.title()}" + (f" ({m.group(1)})" if m.group(1) else "") + ".**",
            tex,
        )
        tex = _re.sub(rf"\\end\{{{env}\}}", "", tex)

    # Proof environment
    tex = _re.sub(r"\\begin\{proof\}(?:\[([^\]]*)\])?",
                  lambda m: "\n*Proof" + (f" ({m.group(1)})" if m.group(1) else "") + ".*",
                  tex)
    tex = _re.sub(r"\\end\{proof\}", "$$\\\\square$$\n", tex)

    # Tables: simplify (just strip the environment, keep content)
    tex = _re.sub(r"\\begin\{(?:table|tabular|longtable|tabularx)\}(?:\[.*?\])?\{[^}]*\}", "", tex)
    tex = _re.sub(r"\\end\{(?:table|tabular|longtable|tabularx)\}", "", tex)
    tex = _re.sub(r"\\(?:hline|toprule|midrule|bottomrule|cline\{[^}]*\})", "---", tex)
    tex = _re.sub(r"\\caption\{([^}]*)\}", r"*\1*", tex)

    # Figures
    tex = _re.sub(r"\\begin\{figure\}(?:\[.*?\])?", "", tex)
    tex = _re.sub(r"\\end\{figure\}", "", tex)
    tex = _re.sub(r"\\includegraphics(?:\[.*?\])?\{([^}]*)\}", r"![figure](\1)", tex)

    # URLs and hrefs
    tex = _re.sub(r"\\href\{([^}]*)\}\{([^}]*)\}", r"[\2](\1)", tex)
    tex = _re.sub(r"\\url\{([^}]*)\}", r"[\1](\1)", tex)

    # Footnotes → parenthetical
    tex = _re.sub(r"\\footnote\{([^}]*)\}", r" (\1)", tex)

    # Strip remaining commands we don't handle
    tex = _re.sub(r"\\(?:label|ref|eqref|pageref|vspace|hspace|noindent|centering|small|large|Large|footnotesize|normalsize)\b\{?[^}]*\}?", "", tex)
    tex = _re.sub(r"\\(?:newcommand|renewcommand|DeclareMathOperator)\{[^}]*\}(?:\[[^\]]*\])?\{[^}]*\}", "", tex)

    # Clean up excessive blank lines
    tex = _re.sub(r"\n{3,}", "\n\n", tex)

    return tex.strip()


def build_quick_verdict_node(workspace_dir: str) -> Any:
    """Terminal node for the quick-pass pipeline.

    Assembles quick_pass_verdict.json and a combined final_paper.md
    (condensed lit review + research plan) so the publish step works.
    """

    def quick_verdict_node(state: dict) -> dict:
        paper_ws = os.path.join(workspace_dir, "paper_workspace")

        novelty_text = _read_file_for_verdict(os.path.join(paper_ws, "novelty_flags.json"))
        brainstorm_text = _read_file_for_verdict(os.path.join(paper_ws, "brainstorm.json"))
        lit_review_text = _read_file_for_verdict(os.path.join(paper_ws, "literature_review.tex"))
        plan_text = _read_file_for_verdict(os.path.join(paper_ws, "research_plan.md"))
        if not plan_text:
            plan_text = _read_file_for_verdict(os.path.join(paper_ws, "research_plan.tex"))

        feasibility = state.get("lit_review_feasibility") or {}
        feasible = feasibility.get("feasible", True)

        novelty_data = {}
        num_approaches = 0
        try:
            if novelty_text:
                novelty_data = json.loads(novelty_text)
        except (json.JSONDecodeError, TypeError):
            pass
        try:
            if brainstorm_text:
                bs = json.loads(brainstorm_text)
                num_approaches = len(bs) if isinstance(bs, list) else len(bs.get("approaches", []))
        except (json.JSONDecodeError, TypeError):
            pass

        claims = novelty_data.get("claims", [])
        open_claims = [c for c in claims if c.get("status") == "OPEN"]
        known_claims = [c for c in claims if c.get("status") in ("KNOWN", "EQUIVALENT_KNOWN")]

        verdict = {
            "feasible": feasible,
            "feasibility_reason": feasibility.get("reason", ""),
            "total_claims": len(claims),
            "open_claims": len(open_claims),
            "known_claims": len(known_claims),
            "num_approaches": num_approaches,
            "pipeline_mode": "quick",
        }

        verdict_path = os.path.join(workspace_dir, "quick_pass_verdict.json")
        with open(verdict_path, "w") as f:
            json.dump(verdict, f, indent=2)
        print(f"[quick_verdict] Feasible: {feasible}, "
              f"Open claims: {len(open_claims)}/{len(claims)}, "
              f"Approaches: {num_approaches}")

        # Assemble final_paper.tex (lit review + novelty + research plan) and compile PDF
        import subprocess as _sp

        # Build novelty assessment as LaTeX
        novelty_tex = ""
        if claims:
            items = []
            for c in claims:
                status = c.get("status", "?")
                claim_text = c.get("claim_text", c.get("claim_id", "?"))[:200]
                # Escape LaTeX special chars
                for ch in ("&", "%", "$", "#", "_", "{", "}"):
                    claim_text = claim_text.replace(ch, "\\" + ch)
                items.append(f"  \\item[\\textbf{{{status}}}] {claim_text}")
            novelty_tex = (
                "\\section{Novelty Assessment}\n"
                "\\begin{description}\n"
                + "\n".join(items)
                + "\n\\end{description}\n"
            )

        verdict_line = (
            f"NOT FEASIBLE --- {feasibility.get('reason', 'see details below')}"
            if not feasible
            else f"FEASIBLE --- {len(open_claims)} open claims, {num_approaches} approaches identified"
        )

        # If lit review and plan are already full LaTeX documents, extract their bodies
        def _extract_body(tex: str) -> str:
            import re as _re2
            m = _re2.search(r"\\begin\{document\}(.*?)(?:\\end\{document\})?$", tex, _re2.DOTALL)
            if m:
                body = m.group(1)
                # Remove \maketitle — we have our own
                body = body.replace("\\maketitle", "")
                return body.strip()
            return tex

        lit_body = _extract_body(lit_review_text) if lit_review_text else ""
        plan_body = _extract_body(plan_text) if plan_text else ""

        combined_tex = (
            "\\documentclass[11pt,a4paper]{article}\n"
            "\\usepackage[utf8]{inputenc}\n"
            "\\usepackage[T1]{fontenc}\n"
            "\\usepackage{lmodern}\n"
            "\\usepackage[margin=1.1in]{geometry}\n"
            "\\usepackage{amsmath,amssymb,amsthm}\n"
            "\\usepackage{booktabs}\n"
            "\\usepackage{enumitem}\n"
            "\\usepackage{hyperref}\n"
            "\\usepackage{natbib}\n"
            "\\usepackage{microtype}\n"
            "\\usepackage{xcolor}\n"
            "\\usepackage{graphicx}\n"
            "\\newtheorem{theorem}{Theorem}\n"
            "\\newtheorem{lemma}[theorem]{Lemma}\n"
            "\\newtheorem{proposition}[theorem]{Proposition}\n"
            "\\newtheorem{definition}{Definition}\n"
            "\\newtheorem{remark}{Remark}\n"
            "\\title{Quick Pass: Research Assessment}\n"
            "\\author{PoggioAI Research Consortium}\n"
            f"\\date{{\\today}}\n"
            "\\begin{document}\n"
            "\\maketitle\n"
            f"\\begin{{center}}\\textbf{{Verdict: {verdict_line}}}\\end{{center}}\n"
            "\\bigskip\n"
        )
        if lit_body:
            combined_tex += lit_body + "\n\n"
        if novelty_tex:
            combined_tex += novelty_tex + "\n\n"
        if plan_body:
            combined_tex += "\\section{Research Plan}\n" + plan_body + "\n\n"
        combined_tex += "\\end{document}\n"

        # Write .tex
        tex_path = os.path.join(workspace_dir, "final_paper.tex")
        with open(tex_path, "w") as f:
            f.write(combined_tex)
        print(f"[quick_verdict] LaTeX written: {tex_path}")

        # Compile PDF (best-effort, don't fail the pipeline if LaTeX is missing)
        try:
            for _ in range(2):  # two passes for references
                _sp.run(
                    ["pdflatex", "-interaction=nonstopmode", "-output-directory", workspace_dir, tex_path],
                    cwd=workspace_dir, capture_output=True, timeout=60,
                )
            pdf_path = os.path.join(workspace_dir, "final_paper.pdf")
            if os.path.isfile(pdf_path):
                print(f"[quick_verdict] PDF compiled: {pdf_path}")
            else:
                print("[quick_verdict] PDF compilation failed — .tex file still available")
        except (FileNotFoundError, _sp.TimeoutExpired) as e:
            print(f"[quick_verdict] PDF compilation skipped: {e}")

        return {"finished": True}

    quick_verdict_node.__name__ = "quick_verdict"
    return quick_verdict_node


def quick_lit_review_gate_router(state: "ResearchState") -> str:
    """Lit review gate router for quick-pass: infeasible → verdict, not retry."""
    target = state.get("current_agent") or "brainstorm_agent"
    if target == "persona_council":
        return "quick_verdict_infeasible"
    return "brainstorm_agent"


def build_research_graph_quick(config: "ResearchGraphConfig"):
    """Build the quick-pass pipeline: persona → lit review → brainstorm → goals → plan → verdict."""
    from .graph_config import ResearchGraphConfig  # noqa: F811
    from .persona_council import create_persona_council_node

    workspace_dir = config.workspace_dir
    authorized_imports = config.authorized_imports
    summary_model_id = config.summary_model_id
    checkpointer = config.checkpointer
    cli_backend_registry = config.cli_backend_registry
    lit_review_max_attempts = config.artifacts.lit_review_max_attempts

    persona_council_specs = config.persona_council.specs
    persona_debate_rounds = config.persona_council.debate_rounds
    persona_synthesis_model = config.persona_council.synthesis_model
    persona_max_post_vote_retries = config.persona_council.max_post_vote_retries

    counsel_kwargs: dict = {}

    def _m(agent_name: str) -> Any:
        return cli_backend_registry.get(agent_name)

    def _wrap(node, name):
        return with_pdf_summary(node, name, workspace_dir, summary_model_id)

    nodes: dict[str, Any] = {
        "persona_council": create_persona_council_node(
            workspace_dir=workspace_dir,
            persona_specs=persona_council_specs,
            max_debate_rounds=persona_debate_rounds,
            synthesis_model=persona_synthesis_model,
            max_post_vote_retries=persona_max_post_vote_retries,
        ),
        "literature_review_agent": _wrap(
            build_literature_review_node(_m("literature_review_agent"), workspace_dir, authorized_imports, **counsel_kwargs),
            "literature_review_agent",
        ),
        "lit_review_gate": build_lit_review_gate_node(workspace_dir, max_attempts=1),
        "brainstorm_agent": _wrap(
            build_brainstorm_node(_m("brainstorm_agent"), workspace_dir, authorized_imports, **counsel_kwargs),
            "brainstorm_agent",
        ),
        "formalize_goals_entry": build_formalize_goals_entry_node(workspace_dir),
        "formalize_goals_agent": _wrap(
            build_formalize_goals_node(_m("formalize_goals_agent"), workspace_dir, authorized_imports, **counsel_kwargs),
            "formalize_goals_agent",
        ),
        "research_plan_writeup_agent": _wrap(
            build_research_plan_writeup_node(_m("research_plan_writeup_agent"), workspace_dir, authorized_imports, **counsel_kwargs),
            "research_plan_writeup_agent",
        ),
        "quick_verdict": build_quick_verdict_node(workspace_dir),
        "quick_verdict_infeasible": build_quick_verdict_node(workspace_dir),
    }

    graph = StateGraph(ResearchState)
    for name, node in nodes.items():
        graph.add_node(name, node)

    graph.set_entry_point("persona_council")
    graph.add_edge("persona_council", "literature_review_agent")
    graph.add_edge("literature_review_agent", "lit_review_gate")
    graph.add_conditional_edges(
        "lit_review_gate",
        quick_lit_review_gate_router,
        {
            "brainstorm_agent": "brainstorm_agent",
            "quick_verdict_infeasible": "quick_verdict_infeasible",
        },
    )
    graph.add_edge("quick_verdict_infeasible", END)
    graph.add_edge("brainstorm_agent", "formalize_goals_entry")
    graph.add_edge("formalize_goals_entry", "formalize_goals_agent")
    graph.add_edge("formalize_goals_agent", "research_plan_writeup_agent")
    graph.add_edge("research_plan_writeup_agent", "quick_verdict")
    graph.add_edge("quick_verdict", END)

    compile_kwargs: dict = {}
    if checkpointer is not None:
        compile_kwargs["checkpointer"] = checkpointer

    return graph.compile(**compile_kwargs)


# ---------------------------------------------------------------------------
# Routing helpers
# ---------------------------------------------------------------------------

def _format_track_task(state: dict, track_name: str, questions: list[str]) -> str:
    cycle = safe_int(state.get("research_cycle", 0), 0)
    if not questions:
        return f"No {track_name} questions were identified for this cycle."

    question_lines = "\n".join(f"- {question}" for question in questions)
    base = (
        f"Research cycle: {cycle}\n"
        f"Execute the {track_name} track for the current research plan.\n"
        f"Questions assigned to this track:\n{question_lines}\n\n"
        "Read the latest planning artifacts from `paper_workspace/`, "
        "produce the mandatory artifacts for your track, and ground all work "
        "in workspace evidence and cited literature."
    )

    # Inject structured goal context for the theory track
    if track_name == "theory":
        try:
            research_goals = state.get("research_goals") or {}
            theory_goals = [
                g for g in research_goals.get("goals", [])
                if g.get("track") in ("theory", "both")
            ]
            if theory_goals:
                goal_lines = []
                for g in theory_goals:
                    reframed = g.get("novelty_reframed", False)
                    reframe_note = (
                        f" [REFRAMED from {g.get('reframed_from_claim', '?')} — "
                        f"strategy: {g.get('reframing_strategy', 'N/A')}]"
                        if reframed else ""
                    )
                    sc = g.get("success_criteria") or {}
                    goal_lines.append(
                        f"  Goal {g.get('id', '?')}{reframe_note}:\n"
                        f"    Hypothesis: {g.get('hypothesis_id', 'N/A')}\n"
                        f"    Description: {str(g.get('description', ''))[:300]}\n"
                        f"    Strong success: {sc.get('strong', 'N/A')}\n"
                        f"    Minimum viable: {sc.get('minimum_viable', 'N/A')}"
                    )
                base += (
                    "\n\nTHEORY GOAL CONTEXT (from research_goals.json):\n"
                    + "\n".join(goal_lines)
                )
        except Exception:
            pass  # silently skip if goals unavailable

    # Inject cross-track dependency context for the experiment track
    if track_name == "experiment":
        td = state.get("track_decomposition") or {}
        deps = td.get("cross_track_dependencies") or []
        if deps:
            dep_lines = []
            for dep in deps:
                eq_idx = dep.get("empirical_question_index", "?")
                tq_idx = dep.get("depends_on_theory_question_index", "?")
                dtype = dep.get("dependency_type", "assumes_result")
                fallback = dep.get(
                    "fallback_if_theory_fails",
                    "Proceed with relaxed assumptions."
                )
                dep_lines.append(
                    f"  - Empirical Q{eq_idx} depends on Theory Q{tq_idx} "
                    f"({dtype}). If theory result unavailable: {fallback}"
                )
            base += (
                "\n\nCROSS-TRACK DEPENDENCIES (theory results may not be "
                "available yet):\n" + "\n".join(dep_lines)
            )

    return base


def track_router(state: ResearchState) -> list[Send]:
    """Fan out to the theory and/or experiment tracks based on track decomposition."""
    track_decomposition = state.get("track_decomposition") or {}
    theory_questions = list(track_decomposition.get("theory_questions") or [])
    empirical_questions = list(track_decomposition.get("empirical_questions") or [])
    recommended_track = str(track_decomposition.get("recommended_track", "")).strip().lower()

    sends: list[Send] = []
    theory_allowed = state.get("math_enabled", False) and recommended_track in {"", "both", "theory"}
    experiment_allowed = recommended_track in {"", "both", "empirical"}

    if theory_allowed and theory_questions:
        sends.append(
            Send(
                "theory_track",
                {
                    **state,
                    "agent_task": _format_track_task(state, "theory", theory_questions),
                    "theory_track_status": "in_progress",
                },
            )
        )

    if experiment_allowed and empirical_questions:
        sends.append(
            Send(
                "experiment_track",
                {
                    **state,
                    "agent_task": _format_track_task(state, "experiment", empirical_questions),
                    "experiment_track_status": "in_progress",
                },
            )
        )

    if not sends:
        sends.append(
            Send(
                "track_merge",
                {
                    **state,
                    "agent_task": (
                        "No theory or empirical execution track was selected. "
                        "Proceed directly to synthesis and results analysis."
                    ),
                },
            )
        )
    return sends


def build_track_decomposition_gate_node(
    workspace_dir: str,
    enable_math_agents: bool = False,
) -> Any:
    """
    Validate track_decomposition.json before milestone_goals and track_router().

    Catches:
    - Empty theory_questions when theory track is requested
    - Empty empirical_questions when experiment track is requested
    - recommended_track requesting theory when enable_math_agents=False
    - Missing or malformed track_decomposition.json
    """
    def track_decomposition_gate_node(state: dict) -> dict:
        paper_ws = os.path.join(workspace_dir, "paper_workspace")
        td_path = os.path.join(paper_ws, "track_decomposition.json")

        default_td = {
            "theory_questions": [],
            "empirical_questions": [
                "Execute the research plan as specified in research_goals.json."
            ],
            "recommended_track": "empirical",
            "rationale": "",
        }

        if not os.path.exists(td_path):
            print(
                "[track_decomposition_gate] WARNING: track_decomposition.json "
                "missing — defaulting to empirical only."
            )
            default_td["rationale"] = (
                "track_decomposition.json was missing; defaulted to empirical track."
            )
            return {"track_decomposition": default_td, "agent_task": None}

        try:
            with open(td_path) as f:
                td = json.load(f)
        except Exception as e:
            print(
                f"[track_decomposition_gate] Failed to parse "
                f"track_decomposition.json: {e}"
            )
            default_td["rationale"] = (
                f"track_decomposition.json parse failed: {e}"
            )
            return {"track_decomposition": default_td, "agent_task": None}

        recommended = (
            td.get("recommended_track", "empirical").strip().lower()
        )
        theory_q = td.get("theory_questions") or []
        empirical_q = td.get("empirical_questions") or []
        issues: list[str] = []

        # Enforce math agents constraint
        if not enable_math_agents and recommended in ("theory", "both"):
            issues.append(
                f"recommended_track='{recommended}' but "
                "enable_math_agents=False. Downgrading to 'empirical'."
            )
            recommended = "empirical"
            td["recommended_track"] = "empirical"

        # Enforce non-empty questions for active tracks
        goals_path = os.path.join(paper_ws, "research_goals.json")

        if recommended in ("theory", "both") and not theory_q:
            issues.append(
                "theory_questions is empty despite theory track being requested."
            )
            try:
                with open(goals_path) as f:
                    goals_data = json.load(f)
                theory_q = [
                    f"Address goal {g['id']}: {g['description']}"
                    for g in goals_data.get("goals", [])
                    if g.get("track") in ("theory", "both")
                ]
                td["theory_questions"] = theory_q
                issues.append(
                    f"Recovered {len(theory_q)} theory questions "
                    "from research_goals.json."
                )
            except Exception:
                recommended = "empirical"
                td["recommended_track"] = "empirical"
                issues.append(
                    "Could not recover theory questions — "
                    "downgrading to empirical."
                )

        if recommended in ("empirical", "both") and not empirical_q:
            issues.append(
                "empirical_questions is empty despite empirical track "
                "being requested."
            )
            try:
                with open(goals_path) as f:
                    goals_data = json.load(f)
                empirical_q = [
                    f"Address goal {g['id']}: {g['description']}"
                    for g in goals_data.get("goals", [])
                    if g.get("track") in ("experiment", "both")
                ]
                td["empirical_questions"] = empirical_q
                issues.append(
                    f"Recovered {len(empirical_q)} empirical questions "
                    "from research_goals.json."
                )
            except Exception:
                issues.append("Could not recover empirical questions.")

        if issues:
            print(
                f"[track_decomposition_gate] Issues corrected: {issues}"
            )

        # Write corrected version back to disk
        with open(td_path, "w") as f:
            json.dump(td, f, indent=2)

        return {"track_decomposition": td, "agent_task": None}

    track_decomposition_gate_node.__name__ = "track_decomposition_gate"
    return track_decomposition_gate_node


def build_formalize_goals_entry_node(workspace_dir: str) -> Any:
    """Inject a first-entry agent_task for formalize_goals_agent.

    On the brainstorm → formalize_goals path, agent_task is typically None.
    This node sets a clear task prompt with brainstorm data quality signal.
    On verify_completion re-entry, this node is bypassed because
    verify_completion routes directly to formalize_goals_agent.
    """

    def formalize_goals_entry_node(state: dict) -> dict:
        paper_ws = os.path.join(workspace_dir, "paper_workspace")
        brainstorm_json_exists = os.path.exists(
            os.path.join(paper_ws, "brainstorm.json")
        )
        brainstorm_md_exists = os.path.exists(
            os.path.join(paper_ws, "brainstorm.md")
        )

        if brainstorm_json_exists:
            task = (
                "BEGIN GOAL FORMALIZATION.\n\n"
                "The brainstorm is complete. `brainstorm.json` and `brainstorm.md` "
                "are available in `paper_workspace/`. Read them along with "
                "`research_proposal.md` and formalize the research goals into "
                "`research_goals.json` and `track_decomposition.json`. "
                "Run all programmatic validations before returning."
            )
        elif brainstorm_md_exists:
            task = (
                "BEGIN GOAL FORMALIZATION (DEGRADED MODE).\n\n"
                "`brainstorm.json` is missing — only `brainstorm.md` is available. "
                "Proceed in degraded mode: parse approaches from the markdown, "
                "write `brainstorm_missing_warning.txt`, set "
                "`brainstorm_data_quality: \"degraded\"` in `research_goals.json`, "
                "and limit to 2 goals maximum."
            )
        else:
            task = (
                "BEGIN GOAL FORMALIZATION (MINIMAL MODE).\n\n"
                "Both `brainstorm.json` and `brainstorm.md` are missing. "
                "Derive goals directly from `research_proposal.md` only. "
                "Limit to 2 goals. Set `brainstorm_data_quality: \"minimal\"` "
                "and write `brainstorm_missing_warning.txt` documenting this condition."
            )

        return {"agent_task": task}

    formalize_goals_entry_node.__name__ = "formalize_goals_entry"
    return formalize_goals_entry_node


def build_proofreading_entry_node(workspace_dir: str) -> Any:
    """Inject a targeted agent_task for proofreading_agent.

    On first entry, sets a generic copy-edit task.  On re-entry after
    validation_gate failure, injects the specific validation errors so the
    proofreader can run a targeted pass rather than a full re-audit.
    """

    def proofreading_entry_node(state: dict) -> dict:
        validation_results = state.get("validation_results", {})
        _PROOFREADING_GATES = {"paper_quality", "artifact_gate"}
        failed_gates = {
            gate: result
            for gate, result in validation_results.items()
            if not result.get("is_valid") and gate in _PROOFREADING_GATES
        }
        paper_ws = os.path.join(workspace_dir, "paper_workspace")
        has_prior_report = os.path.exists(
            os.path.join(paper_ws, "copyedit_report.tex")
        )

        if failed_gates:
            error_lines = [
                f"- {g}: {'; '.join(r.get('errors', []))}"
                for g, r in failed_gates.items()
            ]
            task = (
                "TARGETED COPY-EDIT PASS (re-entry after validation failure).\n\n"
                "Prior validation failures:\n" + "\n".join(error_lines) + "\n\n"
                + (
                    "A prior copyedit_report.tex exists — read it first to "
                    "avoid re-auditing fixed issues.\n"
                    if has_prior_report
                    else ""
                )
                + "Focus your edits on addressing these specific failures."
            )
        else:
            task = (
                "BEGIN COPY-EDIT PASS.\n\n"
                "Perform a full proofreading and copy-editing pass on the paper."
            )

        cleared = {
            k: v for k, v in validation_results.items()
            if k not in _PROOFREADING_GATES
        }
        return {"agent_task": task, "validation_results": cleared}

    proofreading_entry_node.__name__ = "proofreading_entry"
    return proofreading_entry_node


def _formalize_results_state_mapper(inner_node: Callable) -> Callable:
    """Copy agent output into the top-level formalized_results state key."""
    def wrapped(state: dict) -> dict:
        result = inner_node(state)
        agent_output = (result or {}).get("agent_outputs", {}).get("formalize_results_agent")
        if agent_output is not None:
            result["formalized_results"] = agent_output
        else:
            import logging
            logging.getLogger(__name__).warning(
                "formalize_results_agent produced no output — "
                "writing sentinel to formalized_results"
            )
            result["formalized_results"] = "[formalize_results_agent produced no output]"
        return result
    wrapped.__name__ = getattr(inner_node, "__name__", "formalize_results_agent")
    return wrapped


def followup_router(state: ResearchState) -> str:
    # Routing decision was already made by followup_gate_node which sets current_agent
    # to "research_planner_agent" (loop) or "resource_preparation_agent" (continue).
    return state.get("current_agent") or "resource_preparation_agent"


def validation_router(state: ResearchState) -> str:
    if state.get("finished"):
        return END
    # Cap retries to prevent unbounded loops
    retry_count = safe_int(state.get("validation_retry_count", 0), 0)
    max_retries = max(1, safe_int(state.get("max_validation_retries", 3), 3))
    if retry_count >= max_retries:
        return END
    # Route based on review verdict fix type classification
    workspace = state.get("workspace_dir") or "."
    fix_type = classify_review_fixes(workspace)
    if fix_type == "experiment":
        return "experiment_track"
    if fix_type == "theory":
        return "theory_track"
    return "writeup_agent"


def build_followup_gate_node(workspace_dir: str) -> Any:
    def followup_gate_node(state: dict) -> dict:
        required, reason = followup_decision_requires_loop(workspace_dir)
        research_cycle = safe_int(state.get("research_cycle", 0), 0)
        max_cycles = max(0, safe_int(state.get("max_research_cycles", 3), 3))

        if required and research_cycle < max_cycles:
            return {
                "current_agent": "research_planner_agent",
                "research_cycle": research_cycle + 1,
                "followup_iteration": safe_int(state.get("followup_iteration", 0), 0) + 1,
                "finished": False,
                "agent_task": (
                    "Prepare a focused follow-up research plan based on "
                    f"results analysis. Reason: {reason}"
                ),
            }

        return {
            "current_agent": "resource_preparation_agent",
            "finished": False,
            "agent_task": None,
        }

    followup_gate_node.__name__ = "followup_gate"
    return followup_gate_node


def build_validation_gate_node() -> Any:
    def validation_gate_node(state: dict) -> dict:
        validation = run_validation_gates(state)
        retry_count = safe_int(state.get("validation_retry_count", 0), 0)
        max_retries = max(1, safe_int(state.get("max_validation_retries", 3), 3))

        if validation["gate_passed"]:
            return {
                "validation_results": validation["validation_results"],
                "finished": True,
                "agent_task": None,
            }

        # Force completion when retry cap is reached
        if retry_count >= max_retries:
            return {
                "validation_results": validation["validation_results"],
                "finished": True,
                "validation_retry_count": retry_count,
                "agent_task": None,
            }

        error_lines = [
            f"- {gate}: {'; '.join(result['errors'])}"
            for gate, result in validation["validation_results"].items()
            if not result.get("is_valid")
        ]
        # Only increment retry count when review_verdict gate has actually
        # run (i.e., the reviewer produced a verdict).  Artifact-missing
        # failures should not consume retry slots.
        has_review_verdict = "review_verdict" in validation["validation_results"]
        return {
            "validation_results": validation["validation_results"],
            "finished": False,
            "validation_retry_count": retry_count + (1 if has_review_verdict else 0),
            "agent_task": (
                "Revise the paper to satisfy validation gates before finalization.\n"
                "Validation failures:\n" + "\n".join(error_lines)
            ),
        }

    validation_gate_node.__name__ = "validation_gate"
    return validation_gate_node


def build_novelty_gate_node(workspace_dir: str) -> Any:
    """Gate between ideation and literature review that checks novelty assessment."""

    def novelty_gate_node(state: dict) -> dict:
        assessment_path = os.path.join(
            workspace_dir, "paper_workspace", "novelty_assessment.json"
        )
        max_attempts = 3
        attempts = safe_int(state.get("novelty_check_attempts", 0), 0)

        if not os.path.exists(assessment_path):
            # No assessment file -- pass through (backward compat)
            return {"current_agent": "literature_review_agent", "agent_task": None}

        try:
            with open(assessment_path) as f:
                assessment = json.load(f)
        except Exception:
            return {"current_agent": "literature_review_agent", "agent_task": None}

        if assessment.get("novel", True):
            return {"current_agent": "literature_review_agent", "agent_task": None}

        if attempts >= max_attempts:
            # After max attempts, proceed anyway with a warning
            print(
                f"Warning: novelty gate failed after {max_attempts} attempts, "
                "proceeding to literature review."
            )
            return {"current_agent": "literature_review_agent", "agent_task": None}

        justification = assessment.get("novelty_justification", "N/A")
        closest = assessment.get("closest_existing_work", "N/A")
        return {
            "current_agent": "ideation_agent",
            "novelty_check_attempts": attempts + 1,
            "agent_task": (
                f"NOVELTY GATE REJECTION (attempt {attempts + 1}/{max_attempts}): "
                f"Your previous idea was assessed as NOT NOVEL.\n"
                f"Justification: {justification}\n"
                f"Closest existing work: {closest}\n\n"
                "Generate a substantially different idea that addresses a genuine "
                "gap in the literature. Delete the old novelty_assessment.json and "
                "run CheckIdeaNoveltyTool again on your new idea."
            ),
        }

    novelty_gate_node.__name__ = "novelty_gate"
    return novelty_gate_node


def build_milestone_gate_node(phase_name: str, workspace_dir: str) -> Any:
    """Build a milestone gate node that generates a report and optionally pauses.

    When ``enable_milestone_gates`` is True in state, the node blocks until a
    human responds via ``POST /milestone_response`` or the timeout expires.
    When disabled, it generates the PDF report but does not pause.
    """

    # Map milestone phase names to intermediate validation checkpoint names
    _PHASE_TO_CHECKPOINT = {
        "research_plan": None,           # no validation yet at planning stage
        "track_results": "post_merge",   # cross-track consistency
        "analysis": "post_analysis",     # early claim traceability
        "review": None,                  # final validation handles this
    }

    def milestone_gate_node(state: dict) -> dict:
        # Run intermediate validation if applicable for this phase
        checkpoint = _PHASE_TO_CHECKPOINT.get(phase_name)
        validation_update = {}
        if checkpoint:
            validation_update = run_intermediate_validation(state, checkpoint)

        budget_status = None
        budget_path = os.path.join(workspace_dir, "budget_state.json")
        if os.path.exists(budget_path):
            try:
                with open(budget_path) as f:
                    import json as _json
                    bd = _json.load(f)
                total = bd.get("total_usd", bd.get("total_cost_usd"))
                limit = bd.get("usd_limit")
                if total is not None and limit is not None:
                    budget_status = f"Budget: ${total:.2f} / ${limit:.2f} ({total / limit * 100:.0f}%)"
                elif total is not None:
                    budget_status = f"Budget spent: ${total:.2f}"
            except Exception:
                pass

        # Merge validation log into state before generating report
        # so the milestone report can display intermediate results
        merged_state = {**state, **validation_update}

        report_path = generate_milestone_report(
            phase=phase_name,
            state=merged_state,
            workspace_dir=workspace_dir,
            budget_status=budget_status,
        )

        update: dict = {"milestone_reports": [report_path] if report_path else []}
        update.update(validation_update)

        if state.get("enable_milestone_gates") and not state.get("autonomous_mode"):
            timeout = state.get("milestone_timeout", 3600)
            feedback = wait_for_human_response(timeout=timeout)
            if feedback:
                from datetime import datetime, timezone as _tz
                update["human_feedback_history"] = [{
                    "phase": phase_name,
                    "action": feedback.get("action", "approve"),
                    "feedback": feedback.get("feedback", ""),
                    "timestamp": datetime.now(_tz.utc).isoformat(),
                }]
                action = feedback.get("action", "approve")
                if action == "modify":
                    update["agent_task"] = feedback.get("feedback", "")
                elif action == "abort":
                    update["finished"] = True

        return update

    milestone_gate_node.__name__ = f"milestone_{phase_name}"
    return milestone_gate_node


def novelty_router(state: ResearchState) -> str:
    """Route based on novelty gate decision."""
    return state.get("current_agent") or "literature_review_agent"


def build_theory_track_subgraph(
    model: Any,
    workspace_dir: str,
    authorized_imports: Optional[List[str]] = None,
    counsel_models: Optional[List[Any]] = None,
    summary_model_id: Optional[str] = "claude-sonnet-4-6",
    model_registry: Optional[Any] = None,
    adversarial_verification: bool = False,
):
    graph = StateGraph(ResearchState)
    counsel_kwargs = {"counsel_models": counsel_models} if counsel_models is not None else {}

    def _m(agent_name: str) -> Any:
        if model_registry is not None:
            return model_registry.get(agent_name)
        return model

    def _wrap(node, name):
        return with_pdf_summary(node, name, workspace_dir, summary_model_id)

    graph.add_node(
        "math_literature_agent",
        _wrap(build_math_literature_node(_m("math_literature_agent"), workspace_dir, authorized_imports, **counsel_kwargs), "math_literature_agent"),
    )
    graph.add_node(
        "math_proposer_agent",
        _wrap(build_math_proposer_node(_m("math_proposer_agent"), workspace_dir, authorized_imports, **counsel_kwargs), "math_proposer_agent"),
    )
    graph.add_node(
        "math_prover_agent",
        _wrap(build_math_prover_node(_m("math_prover_agent"), workspace_dir, authorized_imports, **counsel_kwargs), "math_prover_agent"),
    )
    graph.add_node(
        "math_rigorous_verifier_agent",
        _wrap(build_math_rigorous_verifier_node(_m("math_rigorous_verifier_agent"), workspace_dir, authorized_imports, adversarial=adversarial_verification, **counsel_kwargs), "math_rigorous_verifier_agent"),
    )
    graph.add_node(
        "math_empirical_verifier_agent",
        _wrap(build_math_empirical_verifier_node(_m("math_empirical_verifier_agent"), workspace_dir, authorized_imports, **counsel_kwargs), "math_empirical_verifier_agent"),
    )
    graph.add_node(
        "proof_transcription_agent",
        _wrap(build_proof_transcription_node(_m("proof_transcription_agent"), workspace_dir, authorized_imports, **counsel_kwargs), "proof_transcription_agent"),
    )
    # -- Issue 3: goal tag validation gate (warn-only, no LLM) --
    def goal_tag_validation_gate(state: dict) -> dict:
        """Check that all must_accept claims have goal:<id> tags."""
        math_ws = os.path.join(workspace_dir, "math_workspace")
        cg_path = os.path.join(math_ws, "claim_graph.json")
        warnings = []
        try:
            with open(cg_path) as f:
                cg = json.load(f)
            for claim in cg.get("claims", []):
                if claim.get("must_accept") and claim.get("status") != "rejected":
                    tags = [str(t) for t in claim.get("tags", [])]
                    has_goal_tag = any(t.startswith("goal:") for t in tags)
                    if not has_goal_tag:
                        warnings.append(
                            f"- {claim.get('id')}: must_accept=true but no goal:<id> tag"
                        )
        except (FileNotFoundError, json.JSONDecodeError):
            pass  # claim graph may not exist yet if proposer failed
        if warnings:
            warn_path = os.path.join(math_ws, "goal_tag_warnings.md")
            content = "# Goal Tag Warnings\n\nThe following must_accept claims are missing goal:<id> tags.\nThis may cause verify_completion to under-count goal progress.\n\n" + "\n".join(warnings) + "\n"
            os.makedirs(math_ws, exist_ok=True)
            with open(warn_path, "w") as f:
                f.write(content)
            print(f"[goal_tag_validation_gate] WARNING: {len(warnings)} must_accept claims missing goal tags. See {warn_path}")
        return {"agent_task": None}

    # -- Issue 6: human review gate (scan checks for human_review_needed) --
    def human_review_gate(state: dict) -> dict:
        """Scan checks/*.jsonl for human_review_needed flags and write summary."""
        math_ws = os.path.join(workspace_dir, "math_workspace")
        checks_dir = os.path.join(math_ws, "checks")
        flags = []
        if os.path.isdir(checks_dir):
            for fname in os.listdir(checks_dir):
                if fname.endswith(".jsonl"):
                    fpath = os.path.join(checks_dir, fname)
                    try:
                        with open(fpath) as f:
                            for line in f:
                                line = line.strip()
                                if not line:
                                    continue
                                try:
                                    entry = json.loads(line)
                                except json.JSONDecodeError:
                                    continue
                                if entry.get("next_action") == "human_review_needed":
                                    claim_id = fname.replace(".jsonl", "")
                                    reason = entry.get("reason", entry.get("message", "tool-vs-manual conflict"))
                                    flags.append(f"- {claim_id}: {reason}")
                    except OSError:
                        continue
        if flags:
            flag_path = os.path.join(math_ws, "human_review_flags.md")
            content = "# Human Review Flags\n\nThe following claims have unresolved tool-vs-manual verification conflicts.\n\n" + "\n".join(flags) + "\n"
            with open(flag_path, "w") as f:
                f.write(content)
            print(f"[human_review_gate] {len(flags)} claims flagged for human review. See {flag_path}")
        return {"agent_task": None}

    # -- Issue 10: intra-track repair gate (max 2 retries) --
    REPAIR_THRESHOLD = 0.7  # must_accept completion ratio below which we retry
    MAX_THEORY_REPAIRS = 2

    def theory_track_repair_gate(state: dict) -> dict:
        """Check must_accept claim completion; route back to prover if below threshold."""
        repair_count = state.get("theory_repair_count", 0)
        math_ws = os.path.join(workspace_dir, "math_workspace")
        summary_path = os.path.join(workspace_dir, "paper_workspace", "theory_track_summary.json")

        # Try to read the structured summary
        must_accept_total = 0
        must_accept_ok = 0
        try:
            with open(summary_path) as f:
                summary = json.load(f)
            verified = set(summary.get("verified_numeric_claims", []))
            verified |= set(summary.get("verified_symbolic_claims", []))
            # Count must_accept claims from the claim graph
            cg_path = os.path.join(math_ws, "claim_graph.json")
            with open(cg_path) as f:
                cg = json.load(f)
            for claim in cg.get("claims", []):
                if claim.get("must_accept"):
                    must_accept_total += 1
                    if claim.get("status") in ("verified_symbolic", "verified_numeric", "accepted"):
                        must_accept_ok += 1
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            # Can't assess — don't retry
            return {"agent_task": None}

        ratio = must_accept_ok / must_accept_total if must_accept_total > 0 else 1.0
        if ratio < REPAIR_THRESHOLD and repair_count < MAX_THEORY_REPAIRS:
            print(f"[theory_track_repair_gate] must_accept ratio {ratio:.2f} < {REPAIR_THRESHOLD}, "
                  f"retry {repair_count + 1}/{MAX_THEORY_REPAIRS} — routing back to prover")
            # Write a targeted repair task for the prover
            failed_claims = []
            try:
                with open(os.path.join(math_ws, "claim_graph.json")) as f:
                    cg = json.load(f)
                for claim in cg.get("claims", []):
                    if claim.get("must_accept") and claim.get("status") in ("proved_draft", "proposed"):
                        failed_claims.append(claim.get("id", "unknown"))
            except (FileNotFoundError, json.JSONDecodeError):
                pass
            repair_task = (
                f"REPAIR PASS {repair_count + 1}: Focus on these blocked must_accept claims: "
                + ", ".join(failed_claims[:10])
                + ". Read their checks/*.jsonl audit logs for specific failure reasons and fix the proofs."
            )
            return {
                "agent_task": repair_task,
                "theory_repair_count": repair_count + 1,
            }
        if ratio < REPAIR_THRESHOLD:
            print(f"[theory_track_repair_gate] must_accept ratio {ratio:.2f} still below threshold "
                  f"after {MAX_THEORY_REPAIRS} retries — proceeding to END")
        return {"agent_task": None}

    def theory_repair_router(state: dict) -> str:
        """Route to prover for retry or END."""
        task = state.get("agent_task")
        if task and task.startswith("REPAIR PASS"):
            return "math_prover_agent"
        return END

    graph.add_node("goal_tag_validation_gate", goal_tag_validation_gate)
    graph.add_node("human_review_gate", human_review_gate)
    graph.add_node("theory_track_repair_gate", theory_track_repair_gate)

    graph.set_entry_point("math_literature_agent")
    graph.add_edge("math_literature_agent", "math_proposer_agent")
    graph.add_edge("math_proposer_agent", "goal_tag_validation_gate")
    graph.add_edge("goal_tag_validation_gate", "math_prover_agent")
    graph.add_edge("math_prover_agent", "math_rigorous_verifier_agent")
    graph.add_edge("math_rigorous_verifier_agent", "human_review_gate")
    graph.add_edge("human_review_gate", "math_empirical_verifier_agent")
    graph.add_edge("math_empirical_verifier_agent", "proof_transcription_agent")
    graph.add_edge("proof_transcription_agent", "theory_track_repair_gate")
    graph.add_conditional_edges(
        "theory_track_repair_gate",
        theory_repair_router,
        {"math_prover_agent": "math_prover_agent", END: END},
    )
    return graph.compile()


def build_experiment_track_subgraph(
    model: Any,
    workspace_dir: str,
    authorized_imports: Optional[List[str]] = None,
    counsel_models: Optional[List[Any]] = None,
    summary_model_id: Optional[str] = "claude-sonnet-4-6",
    model_registry: Optional[Any] = None,
):
    graph = StateGraph(ResearchState)
    counsel_kwargs = {"counsel_models": counsel_models} if counsel_models is not None else {}

    def _m(agent_name: str) -> Any:
        if model_registry is not None:
            return model_registry.get(agent_name)
        return model

    def _wrap(node, name):
        return with_pdf_summary(node, name, workspace_dir, summary_model_id)

    graph.add_node(
        "experiment_literature_agent",
        _wrap(build_experiment_literature_node(_m("experiment_literature_agent"), workspace_dir, authorized_imports, **counsel_kwargs), "experiment_literature_agent"),
    )
    graph.add_node(
        "experiment_design_agent",
        _wrap(build_experiment_design_node(_m("experiment_design_agent"), workspace_dir, authorized_imports, **counsel_kwargs), "experiment_design_agent"),
    )
    graph.add_node(
        "experimentation_agent",
        _wrap(build_experimentation_node(_m("experimentation_agent"), workspace_dir, authorized_imports, **counsel_kwargs), "experimentation_agent"),
    )
    graph.add_node(
        "experiment_verification_agent",
        _wrap(build_experiment_verification_node(_m("experiment_verification_agent"), workspace_dir, authorized_imports, **counsel_kwargs), "experiment_verification_agent"),
    )
    graph.add_node(
        "experiment_transcription_agent",
        _wrap(build_experiment_transcription_node(_m("experiment_transcription_agent"), workspace_dir, authorized_imports, **counsel_kwargs), "experiment_transcription_agent"),
    )
    graph.set_entry_point("experiment_literature_agent")
    graph.add_edge("experiment_literature_agent", "experiment_design_agent")
    graph.add_edge("experiment_design_agent", "experimentation_agent")
    graph.add_edge("experimentation_agent", "experiment_verification_agent")
    graph.add_edge("experiment_verification_agent", "experiment_transcription_agent")
    graph.add_edge("experiment_transcription_agent", END)
    return graph.compile()


def build_track_subgraph_node(
    subgraph: Any,
    status_field: str,
    status_value: str = "completed",
    workspace_dir: Optional[str] = None,
) -> Any:
    def node(state: dict) -> dict:
        final_state = subgraph.invoke(state)
        result = {
            "agent_outputs": final_state.get("agent_outputs", {}),
            status_field: status_value,
            "agent_task": None,
        }
        # Forward structured track summaries if available on disk
        if workspace_dir:
            summary_path = os.path.join(workspace_dir, "paper_workspace", "theory_track_summary.json")
            if os.path.isfile(summary_path):
                try:
                    with open(summary_path) as f:
                        result["theory_track_summary"] = json.load(f)
                except (json.JSONDecodeError, OSError):
                    pass
        return result

    node.__name__ = status_field.removesuffix("_status")
    return node


def build_noop_track_node(status_field: str) -> Any:
    def node(state: dict) -> dict:
        return {
            status_field: state.get(status_field),
            "agent_task": None,
        }

    node.__name__ = status_field.removesuffix("_status")
    return node


def build_synthesis_literature_node(
    model: Any,
    workspace_dir: str,
    authorized_imports: Optional[List[str]] = None,
    counsel_models: Optional[List[Any]] = None,
) -> Any:
    base_node = build_literature_review_node(
        model,
        workspace_dir,
        authorized_imports,
        **({"counsel_models": counsel_models} if counsel_models is not None else {}),
    )

    def synthesis_node(state: dict) -> dict:
        previous_output = state.get("agent_outputs", {}).get("literature_review_agent")
        result = base_node(state)
        outputs = dict(result.get("agent_outputs", {}))
        new_output = outputs.pop("literature_review_agent", previous_output)
        if previous_output is not None:
            outputs["literature_review_agent"] = previous_output
        outputs["synthesis_literature_review_agent"] = new_output
        return {
            **result,
            "agent_outputs": outputs,
        }

    synthesis_node.__name__ = "synthesis_literature_review_agent"
    return synthesis_node


# ---------------------------------------------------------------------------
# Pipeline — gate nodes and routers
# ---------------------------------------------------------------------------


def _read_file_safe(path: str, max_chars: int = 20000) -> str:
    """Read a file, returning empty string on failure."""
    if not os.path.exists(path):
        return ""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read(max_chars)
    except Exception:
        return ""


def _build_brainstorm_novelty_directive(novelty_data: dict) -> str:
    """Build a structured novelty directive for the brainstorm agent.

    Extracts OPEN and PARTIAL claims from novelty_flags.json data and
    formats them into an actionable directive so the brainstorm agent
    starts with rich context rather than a single summary sentence.
    """
    lines = ["NOVELTY GATE PASSED — structured claim summary from literature review:\n"]

    claims = novelty_data.get("claims", [])
    open_claims = [c for c in claims if c.get("status") == "OPEN"]
    partial_claims = [c for c in claims if c.get("status") == "PARTIAL"]

    if open_claims:
        lines.append("CONFIRMED OPEN CLAIMS (prioritize these):")
        for c in open_claims:
            cid = c.get("claim_id", "?")
            text = c.get("claim_text", "")[:200]
            conf = c.get("confidence", "?")
            rec = c.get("recommendation", "?")
            lines.append(f"  [{cid}] {text} (confidence: {conf}, recommendation: {rec})")
        lines.append("")

    if partial_claims:
        lines.append("PARTIALLY RESOLVED CLAIMS (extend beyond existing partial results):")
        for c in partial_claims:
            cid = c.get("claim_id", "?")
            text = c.get("claim_text", "")[:200]
            conf = c.get("confidence", "?")
            rec = c.get("recommendation", "?")
            ev_count = len(c.get("evidence", []))
            lines.append(
                f"  [{cid}] {text} (confidence: {conf}, recommendation: {rec}, "
                f"{ev_count} evidence sources)"
            )
        lines.append("")

    overall = novelty_data.get("overall_novelty_assessment", "N/A")
    lines.append(f"OVERALL ASSESSMENT: {overall}")
    lines.append(
        "\nRead `paper_workspace/novelty_flags.json` for full evidence on each claim."
    )

    return "\n".join(lines)


def build_lit_review_gate_node(workspace_dir: str, max_attempts: int = 2) -> Any:
    """Gate after lit review: checks novelty flags then feasibility.

    Routes to:
      - ``persona_council`` if novelty-blocked or infeasible (retries remain)
      - ``brainstorm_agent`` if feasible or retries exhausted
    """

    def lit_review_gate_node(state: dict) -> dict:
        import re as _re
        attempts = safe_int(state.get("lit_review_attempts", 0), 0)

        paper_ws = os.path.join(workspace_dir, "paper_workspace")
        lit_text = _read_file_safe(os.path.join(paper_ws, "literature_review.tex"))
        sources_text = _read_file_safe(
            os.path.join(paper_ws, "literature_review_sources.json"), 10000
        )

        if not lit_text:
            # No lit review output — pass through
            return {
                "current_agent": "brainstorm_agent",
                "lit_review_feasibility": {"feasible": True, "reason": "no lit review found"},
                "agent_task": None,
            }

        # ------------------------------------------------------------------
        # Novelty check — read novelty_flags.json (produced by Step 3b)
        # ------------------------------------------------------------------
        novelty_path = os.path.join(paper_ws, "novelty_flags.json")
        novelty_text = _read_file_safe(novelty_path, 10000)
        novelty_data = None
        blocking_claims = []

        if novelty_text:
            try:
                novelty_data = json.loads(novelty_text)
                blocking_claims = [
                    c for c in novelty_data.get("claims", [])
                    if c.get("blocking", False)
                ]
            except (json.JSONDecodeError, TypeError) as e:
                print(f"[lit_review_gate] Failed to parse novelty_flags.json: {e}")

        # Gate on blocking novelty claims BEFORE the LLM feasibility call
        if blocking_claims:
            blocking_summary = "; ".join(
                f"[{c.get('claim_id', '?')}] "
                f"{c.get('claim_text', '')[:150]}... "
                f"(status: {c.get('status', '?')}, "
                f"evidence: {len(c.get('evidence', []))} sources)"
                for c in blocking_claims
            )

            if attempts >= max_attempts:
                print(
                    f"Warning: lit_review_gate — blocking novelty claims after "
                    f"{max_attempts} attempts, proceeding to brainstorm_agent."
                )
                return {
                    "current_agent": "brainstorm_agent",
                    "lit_review_feasibility": {
                        "feasible": False,
                        "reason": f"NOVELTY BLOCKED: {blocking_summary}",
                    },
                    "agent_task": (
                        f"NOVELTY WARNING (max retries reached): Some proposed claims "
                        f"may not be novel. Proceed but explicitly acknowledge and "
                        f"work around: {blocking_summary}"
                    ),
                }

            return {
                "current_agent": "persona_council",
                "lit_review_attempts": attempts + 1,
                "lit_review_feasibility": {
                    "feasible": False,
                    "reason": f"NOVELTY BLOCKED: {blocking_summary}",
                },
                "agent_task": (
                    f"NOVELTY GATE REJECTION (attempt {attempts + 1}/{max_attempts}):\n"
                    f"The literature review found that one or more core claims in your "
                    f"research proposal are NOT novel — they have already been proven or "
                    f"established in the existing literature.\n\n"
                    f"Blocking claims:\n{blocking_summary}\n\n"
                    f"You MUST substantially reformulate or replace the blocking claims. "
                    f"Options:\n"
                    f"1. Strengthen the claim (generalize, relax assumptions, extend to "
                    f"new settings)\n"
                    f"2. Replace the claim with a genuinely open related problem\n"
                    f"3. Reframe the contribution as a new proof technique for a known "
                    f"result (only if the technique itself is novel)\n"
                    f"4. Pivot the research direction entirely\n\n"
                    f"Read `paper_workspace/novelty_flags.json` for full evidence."
                ),
            }

        # ------------------------------------------------------------------
        # CLI-based feasibility assessment
        # ------------------------------------------------------------------

        # Build novelty context for the feasibility prompt
        novelty_context = ""
        if novelty_data:
            open_count = sum(
                1 for c in novelty_data.get("claims", [])
                if c.get("status") == "OPEN"
            )
            partial_count = sum(
                1 for c in novelty_data.get("claims", [])
                if c.get("status") == "PARTIAL"
            )
            novelty_context = (
                f"\n\nNOVELTY ASSESSMENT SUMMARY:\n"
                f"- Fully open claims: {open_count}\n"
                f"- Partially resolved claims: {partial_count}\n"
                f"- Blocking (known/equivalent) claims: {len(blocking_claims)}\n"
                f"Overall: {novelty_data.get('overall_novelty_assessment', 'N/A')}"
            )

        prompt = (
            "You are a research feasibility assessor. Given the literature review below, "
            "determine whether the proposed research direction is FEASIBLE.\n\n"
            "A direction is INFEASIBLE if:\n"
            "1. The core idea has already been thoroughly explored (no room for contribution)\n"
            "2. The required methods/data are fundamentally unavailable\n"
            "3. The theoretical foundations have known fatal flaws\n"
            "4. The scope is impossibly broad for a single research effort\n"
            "5. The core proposed claims are already proven results with no novel angle "
            "(check the NOVELTY ASSESSMENT SUMMARY below if available)\n\n"
            f"LITERATURE REVIEW:\n{lit_text[:15000]}\n\n"
            f"SOURCES:\n{sources_text[:5000]}"
            f"{novelty_context}\n\n"
            'Respond in JSON (no markdown fences): {"feasible": true/false, "reason": "one paragraph explanation"}'
        )
        try:
            from .cli_completion import cli_completion
            raw = cli_completion(prompt, backend="claude")
            raw = _re.sub(r"^```(?:json)?\s*", "", raw.strip())
            raw = _re.sub(r"\s*```$", "", raw)
            result = json.loads(raw)
        except Exception as e:
            print(f"[lit_review_gate] CLI assessment failed: {e}, passing through")
            return {
                "current_agent": "brainstorm_agent",
                "lit_review_feasibility": {"feasible": True, "reason": f"assessment failed: {e}"},
                "agent_task": None,
            }

        feasible = result.get("feasible", True)
        reason = result.get("reason", "N/A")

        if feasible:
            return {
                "current_agent": "brainstorm_agent",
                "lit_review_feasibility": {"feasible": True, "reason": reason},
                "agent_task": _build_brainstorm_novelty_directive(novelty_data)
                if novelty_data else None,
            }

        # Infeasible
        if attempts >= max_attempts:
            print(
                f"Warning: lit_review_gate failed after {max_attempts} attempts, "
                "proceeding to brainstorm_agent anyway."
            )
            return {
                "current_agent": "brainstorm_agent",
                "lit_review_feasibility": {"feasible": False, "reason": reason},
                "agent_task": None,
            }

        return {
            "current_agent": "persona_council",
            "lit_review_attempts": attempts + 1,
            "lit_review_feasibility": {"feasible": False, "reason": reason},
            "agent_task": (
                f"LIT REVIEW FEASIBILITY REJECTION (attempt {attempts + 1}/{max_attempts}): "
                f"The literature review found the research direction INFEASIBLE.\n"
                f"Reason: {reason}\n\n"
                "Re-evaluate the research direction. Consider pivoting the angle, "
                "narrowing scope, or identifying an unexplored niche."
            ),
        }

    lit_review_gate_node.__name__ = "lit_review_gate"
    return lit_review_gate_node


def build_verify_completion_node(workspace_dir: str) -> Any:
    """Verify whether formalized research goals have been met by track execution.

    Three-way routing with progress vetting on re-entry:
      - ``formalize_results_agent``: all/nearly all goals met (ratio >= 0.8)
      - ``formalize_goals_agent``: some goals not met (incomplete, ratio 0.5-0.8)
      - ``brainstorm_agent``: fundamental rethink needed (ratio < 0.5)

    On 2nd+ cycle, if goals_met delta <= 0 since last check, forces forward.
    """

    def verify_completion_node(state: dict) -> dict:
        import re as _re

        research_goals = state.get("research_goals") or {}
        goals = research_goals.get("goals", [])
        total_goals = len(goals)

        if total_goals == 0:
            return {
                "current_agent": "formalize_results_agent",
                "verify_completion_result": {
                    "goals_met": 0, "goals_total": 0, "ratio": 1.0,
                    "verdict": "complete", "goal_verdicts": [],
                },
                "agent_task": None,
            }

        # Collect evidence from workspace
        evidence_parts = []
        math_ws = os.path.join(workspace_dir, "math_workspace")
        paper_ws = os.path.join(workspace_dir, "paper_workspace")
        exp_ws = os.path.join(workspace_dir, "experiment_workspace")

        # Parse claim graph for tag-filtered goal matching
        all_claims = []
        cg_content = _read_file_safe(os.path.join(math_ws, "claim_graph.json"), 8000)
        if cg_content:
            try:
                cg = json.loads(cg_content)
                all_claims = cg.get("claims", [])
            except Exception:
                pass

        # Build per-goal claim map using "goal:<id>" tags
        goal_claim_map = {}
        for g in goals:
            gid = g.get("id", "")
            if g.get("track") in ("theory", "both") and gid:
                relevant = [
                    c for c in all_claims
                    if any(f"goal:{gid}" in str(t) for t in c.get("tags", []))
                ]
                goal_claim_map[gid] = relevant

        has_tagged_claims = any(bool(v) for v in goal_claim_map.values())

        if has_tagged_claims:
            # Use tag-filtered evidence: present each goal with only its tagged claims
            for g in goals:
                gid = g.get("id", "")
                if gid in goal_claim_map and goal_claim_map[gid]:
                    claims_summary = json.dumps(
                        [{"id": c.get("id"), "statement": str(c.get("statement", ""))[:200],
                          "status": c.get("status"), "must_accept": c.get("must_accept")}
                         for c in goal_claim_map[gid]],
                        indent=2
                    )
                    evidence_parts.append(
                        f"--- THEORY CLAIMS FOR GOAL {gid} ---\n{claims_summary}"
                    )
            # Include untagged claims as supplementary context
            tagged_ids = {c.get("id") for cs in goal_claim_map.values() for c in cs}
            untagged = [c for c in all_claims if c.get("id") not in tagged_ids]
            if untagged:
                untagged_summary = json.dumps(
                    [{"id": c.get("id"), "status": c.get("status")} for c in untagged],
                    indent=2
                )
                evidence_parts.append(
                    f"--- THEORY: untagged claims (supplementary) ---\n{untagged_summary}"
                )
        elif cg_content:
            # Fallback: no tags found, use raw claim graph as before
            evidence_parts.append(f"--- THEORY: claim_graph.json ---\n{cg_content}")

        for name in ["experiment_results.json", "experiment_design.json"]:
            content = _read_file_safe(os.path.join(paper_ws, name), 8000)
            if content:
                evidence_parts.append(f"--- EXPERIMENT: {name} ---\n{content}")
        for name in ["results_summary.json", "experiment_report.md"]:
            content = _read_file_safe(os.path.join(exp_ws, name), 5000)
            if content:
                evidence_parts.append(f"--- EXPERIMENT: {name} ---\n{content}")

        agent_outputs = state.get("agent_outputs", {})
        for agent_name in ["proof_transcription_agent", "experiment_transcription_agent",
                           "experiment_verification_agent", "track_merge"]:
            output = str(agent_outputs.get(agent_name, "")).strip()
            if output:
                evidence_parts.append(f"--- AGENT: {agent_name} ---\n{output[:3000]}")

        evidence = "\n\n".join(evidence_parts) if evidence_parts else "No execution evidence found."

        goal_descriptions = "\n".join(
            f"Goal {i+1} (ID: {g.get('id', i)}):\n"
            f"  Description: {g.get('description', 'N/A')}\n"
            f"  Success Criteria: {g.get('success_criteria', 'N/A')}\n"
            f"  Track: {g.get('track', 'both')}"
            for i, g in enumerate(goals)
        )

        prompt = (
            "You are a rigorous research goal completion assessor.\n\n"
            f"RESEARCH GOALS:\n{goal_descriptions}\n\n"
            f"EXECUTION EVIDENCE:\n{evidence}\n\n"
            "For EACH goal, assess whether it has been MET based on the evidence.\n"
            "A goal is MET if concrete evidence (data, proofs, results) supports completion.\n"
            "A goal is NOT MET if no evidence exists or evidence contradicts it.\n\n"
            "Respond in JSON (no markdown fences):\n"
            '{"goal_verdicts": [{"goal_id": "...", "met": true/false, "evidence_summary": "..."}], '
            '"overall_assessment": "one paragraph"}'
        )

        try:
            from .cli_completion import cli_completion
            raw = cli_completion(prompt, backend="claude")
            raw = _re.sub(r"^```(?:json)?\s*", "", raw.strip())
            raw = _re.sub(r"\s*```$", "", raw)
            result = json.loads(raw)
        except Exception as e:
            print(f"[verify_completion] CLI assessment failed: {e}, passing through")
            return {
                "current_agent": "formalize_results_agent",
                "verify_completion_result": {
                    "goals_met": total_goals, "goals_total": total_goals,
                    "ratio": 1.0, "verdict": "complete",
                    "goal_verdicts": [], "error": str(e),
                },
                "agent_task": None,
            }

        goal_verdicts = result.get("goal_verdicts", [])
        goals_met = sum(1 for v in goal_verdicts if v.get("met", False))
        ratio = goals_met / total_goals if total_goals > 0 else 1.0

        verify_rework = safe_int(state.get("verify_rework_attempts", 0), 0)
        brainstorm_cyc = safe_int(state.get("brainstorm_cycle", 0), 0)

        # Progress vetting: compare against previous result
        prev_result = state.get("verify_completion_result")
        prev_history = list(state.get("verify_completion_history") or [])
        if prev_result is not None:
            prev_history.append(prev_result)
            prev_goals_met = prev_result.get("goals_met", 0)
            delta = goals_met - prev_goals_met
            if delta <= 0:
                print(
                    f"[verify_completion] Progress stalled: {goals_met} goals met "
                    f"(prev {prev_goals_met}, delta {delta}). Forcing forward."
                )
                return {
                    "current_agent": "formalize_results_agent",
                    "verify_completion_result": {
                        "goals_met": goals_met, "goals_total": total_goals,
                        "ratio": ratio, "verdict": "complete_forced_stalled",
                        "goal_verdicts": goal_verdicts,
                    },
                    "verify_completion_history": prev_history,
                    "agent_task": None,
                }

        new_result = {
            "goals_met": goals_met, "goals_total": total_goals,
            "ratio": ratio, "goal_verdicts": goal_verdicts,
        }

        # Three-way decision
        if ratio >= 0.8:
            new_result["verdict"] = "complete"
            return {
                "current_agent": "formalize_results_agent",
                "verify_completion_result": new_result,
                "verify_completion_history": prev_history,
                "agent_task": None,
            }

        if ratio >= 0.5:
            # Incomplete — rework goals
            if verify_rework >= 3:
                print(
                    f"Warning: verify_completion 'incomplete' after {verify_rework} "
                    "rework attempts, forcing forward."
                )
                new_result["verdict"] = "complete_forced"
                return {
                    "current_agent": "formalize_results_agent",
                    "verify_completion_result": new_result,
                    "verify_completion_history": prev_history,
                    "agent_task": None,
                }

            new_result["verdict"] = "incomplete"
            failed_goals = [v for v in goal_verdicts if not v.get("met", False)]
            failed_summary = "\n".join(
                f"  - {v.get('goal_id', '?')}: {v.get('evidence_summary', 'no evidence')}"
                for v in failed_goals
            )
            return {
                "current_agent": "formalize_goals_agent",
                "verify_rework_attempts": verify_rework + 1,
                "verify_completion_result": new_result,
                "verify_completion_history": prev_history,
                "agent_task": (
                    f"VERIFY COMPLETION: INCOMPLETE (attempt {verify_rework + 1}/3)\n"
                    f"{goals_met}/{total_goals} goals met ({ratio:.0%}).\n"
                    f"Failed goals:\n{failed_summary}\n\n"
                    "Revise the research goals to address the unmet criteria."
                ),
            }

        # < 50% — fundamental rethink
        if brainstorm_cyc >= 3:
            print(
                f"Warning: verify_completion 'no_half' after {brainstorm_cyc} "
                "brainstorm cycles, forcing forward."
            )
            new_result["verdict"] = "complete_forced"
            return {
                "current_agent": "formalize_results_agent",
                "verify_completion_result": new_result,
                "verify_completion_history": prev_history,
                "agent_task": None,
            }

        new_result["verdict"] = "no_half"
        return {
            "current_agent": "brainstorm_agent",
            "brainstorm_cycle": brainstorm_cyc + 1,
            "verify_completion_result": new_result,
            "verify_completion_history": prev_history,
            "agent_task": (
                f"VERIFY COMPLETION: FUNDAMENTAL RETHINK NEEDED (cycle {brainstorm_cyc + 1}/3)\n"
                f"Only {goals_met}/{total_goals} goals met ({ratio:.0%}).\n"
                f"Assessment: {result.get('overall_assessment', 'N/A')}\n\n"
                "The current approach is not working. Brainstorm substantially "
                "different approaches to the research question."
            ),
        }

    verify_completion_node.__name__ = "verify_completion"
    return verify_completion_node


def build_duality_gate_node() -> Any:
    """Gate after duality check: routes based on Check A + B results.

    Routes to:
      - ``resource_preparation_agent`` if both checks pass
      - ``followup_lit_review`` if either check fails (retries remaining)
    """

    def duality_gate_node(state: dict) -> dict:
        duality_result = state.get("duality_check_result") or {}
        both_passed = duality_result.get("both_passed", False)
        duality_rework = safe_int(state.get("duality_rework_attempts", 0), 0)
        max_attempts = 2

        if not duality_result:
            print(
                "Warning: duality_check_result is missing or empty, "
                "proceeding to resource_preparation anyway."
            )
            return {
                "current_agent": "resource_preparation_agent",
                "agent_task": None,
            }

        if both_passed:
            check_a = duality_result.get("check_a", {})
            check_b = duality_result.get("check_b", {})
            summary_parts = []
            for label, check in [("Practice", check_a), ("Rigor", check_b)]:
                score = check.get("score", "?")
                summary_parts.append(f"Check {label}: {score}/10")
                for s in check.get("suggestions", []):
                    summary_parts.append(f"  - {s}")
            return {
                "current_agent": "resource_preparation_agent",
                "agent_task": (
                    "DUALITY CHECK PASSED. Summary of findings for paper production:\n"
                    + "\n".join(summary_parts)
                    + "\n\nFull results: paper_workspace/duality_check.json"
                ),
            }

        if duality_rework >= max_attempts:
            print(
                f"Warning: duality_gate failed after {max_attempts} attempts, "
                "proceeding to resource_preparation anyway."
            )
            check_a = duality_result.get("check_a", {})
            check_b = duality_result.get("check_b", {})
            failures = []
            if not check_a.get("passed", True):
                failures.append(f"Practice: {check_a.get('reasoning', 'Failed')} (score {check_a.get('score', '?')}/10)")
            if not check_b.get("passed", True):
                failures.append(f"Rigor: {check_b.get('reasoning', 'Failed')} (score {check_b.get('score', '?')}/10)")
            return {
                "current_agent": "resource_preparation_agent",
                "agent_task": (
                    f"DUALITY CHECK FORCED FORWARD after {max_attempts} attempts.\n"
                    "Unresolved issues (address in paper where possible):\n"
                    + "\n".join(f"- {f}" for f in failures)
                    + "\n\nFull results: paper_workspace/duality_check.json"
                ) if failures else None,
            }

        check_a = duality_result.get("check_a", {})
        check_b = duality_result.get("check_b", {})
        failures = []
        suggestions = []
        if not check_a.get("passed", True):
            failures.append(f"Check A (Practice): {check_a.get('reasoning', 'Failed')}")
            suggestions.extend(check_a.get("suggestions", []))
        if not check_b.get("passed", True):
            failures.append(f"Check B (Rigor): {check_b.get('reasoning', 'Failed')}")
            suggestions.extend(check_b.get("suggestions", []))

        return {
            "current_agent": "followup_lit_review",
            "duality_rework_attempts": duality_rework + 1,
            "agent_task": (
                f"DUALITY CHECK FAILURE (attempt {duality_rework + 1}/{max_attempts}):\n"
                + "\n".join(f"- {f}" for f in failures)
                + "\n\nSuggestions:\n"
                + "\n".join(f"- {s}" for s in suggestions)
                + "\n\nConduct a targeted follow-up literature review addressing "
                "these gaps, then re-enter the brainstorm phase."
                "\n\nFull structured results: paper_workspace/duality_check.json"
            ),
        }

    duality_gate_node.__name__ = "duality_gate"
    return duality_gate_node


# V2 router functions

def lit_review_gate_router(state: ResearchState) -> str:
    return state.get("current_agent") or "brainstorm_agent"


def verify_completion_router(state: ResearchState) -> str:
    return state.get("current_agent") or "formalize_results_agent"


def duality_gate_router(state: ResearchState) -> str:
    target = state.get("current_agent")
    if target in {"resource_preparation_agent", "followup_lit_review"}:
        return target
    return "resource_preparation_agent"


def build_followup_lit_review_node(
    model: Any,
    workspace_dir: str,
    authorized_imports: Optional[List[str]] = None,
    counsel_models: Optional[List[Any]] = None,
) -> Any:
    """Followup lit review: reuses literature_review_agent under a different output key."""
    base_node = build_literature_review_node(
        model,
        workspace_dir,
        authorized_imports,
        **({"counsel_models": counsel_models} if counsel_models is not None else {}),
    )

    def followup_node(state: dict) -> dict:
        previous_output = state.get("agent_outputs", {}).get("literature_review_agent")
        result = base_node(state)
        outputs = dict(result.get("agent_outputs", {}))
        new_output = outputs.pop("literature_review_agent", previous_output)
        if previous_output is not None:
            outputs["literature_review_agent"] = previous_output
        outputs["followup_lit_review_agent"] = new_output
        return {
            **result,
            "agent_outputs": outputs,
        }

    followup_node.__name__ = "followup_lit_review"
    return followup_node


# ---------------------------------------------------------------------------
# V2 graph builder
# ---------------------------------------------------------------------------

def build_research_graph_v2(config: "ResearchGraphConfig"):
    """
    Build the V2 persona-council-driven research pipeline.

    Args:
        config: A :class:`~consortium.graph_config.ResearchGraphConfig`
            bundling all graph-construction knobs.

    Flow:
      persona_council → lit_review → lit_review_gate → brainstorm → formalize_goals
      → {theory | experiment} → track_merge → verify_completion → formalize_results
      → duality_check → duality_gate → resource_prep → writeup → proofreading
      → reviewer → validation_gate

    With feedback loops:
      - lit_review_gate → persona_council (infeasible)
      - verify_completion → formalize_goals (incomplete)
      - verify_completion → brainstorm (fundamental rethink)
      - duality_gate → followup_lit_review → brainstorm (quality failure)
    """
    from .graph_config import ResearchGraphConfig  # noqa: F811 (type hint above)

    # Unpack config to local variables.
    workspace_dir = config.workspace_dir
    pipeline_mode = config.pipeline_mode
    enable_math_agents = config.enable_math_agents
    enable_milestone_gates = config.enable_milestone_gates
    adversarial_verification = config.adversarial_verification
    min_review_score = config.min_review_score
    followup_max_iterations = config.followup_max_iterations
    manager_max_steps = config.manager_max_steps
    authorized_imports = config.authorized_imports
    summary_model_id = config.summary_model_id
    checkpointer = config.checkpointer
    tree_search_config = config.tree_search
    # Kept as a local for sub-builder compatibility (passed but overridden by registry)
    model = None

    # Sub-config unpacking
    enforce_paper_artifacts = config.artifacts.enforce_paper_artifacts
    enforce_editorial_artifacts = config.artifacts.enforce_editorial_artifacts
    require_pdf = config.artifacts.require_pdf
    require_experiment_plan = config.artifacts.require_experiment_plan
    lit_review_max_attempts = config.artifacts.lit_review_max_attempts

    persona_council_specs = config.persona_council.specs
    persona_debate_rounds = config.persona_council.debate_rounds
    persona_synthesis_model = config.persona_council.synthesis_model
    persona_max_post_vote_retries = config.persona_council.max_post_vote_retries

    duality_check_model = config.duality_check.model
    enable_duality_check = config.duality_check.enabled
    from .persona_council import create_persona_council_node, create_duality_check_node

    counsel_kwargs = {}
    cli_backend_registry = config.cli_backend_registry

    def _m(agent_name: str) -> Any:
        """Resolve the CLI backend spec for *agent_name*."""
        return cli_backend_registry.get(agent_name)

    def _wrap(node, name):
        return with_pdf_summary(node, name, workspace_dir, summary_model_id)

    # Build track subgraphs (same as v1)
    theory_track_node = build_noop_track_node("theory_track_status")
    if enable_math_agents:
        if tree_search_config and getattr(tree_search_config, "enabled", False):
            from consortium.tree_search.graph_integration import (
                build_tree_search_theory_track,
            )
            theory_subgraph = build_tree_search_theory_track(
                model=model,
                workspace_dir=workspace_dir,
                authorized_imports=authorized_imports,
                counsel_models=None,
                summary_model_id=summary_model_id,
                tree_config=tree_search_config,
                adversarial_verification=adversarial_verification,
            )
        else:
            theory_subgraph = build_theory_track_subgraph(
                model=model,
                workspace_dir=workspace_dir,
                authorized_imports=authorized_imports,
                counsel_models=None,
                summary_model_id=summary_model_id,
                model_registry=cli_backend_registry,
                adversarial_verification=adversarial_verification,
            )
        theory_track_node = build_track_subgraph_node(theory_subgraph, "theory_track_status", workspace_dir=workspace_dir)

    if tree_search_config and getattr(tree_search_config, "enabled", False):
        from consortium.tree_search.experiment_tree_integration import (
            build_tree_search_experiment_track,
        )
        experiment_subgraph = build_tree_search_experiment_track(
            model=model,
            workspace_dir=workspace_dir,
            authorized_imports=authorized_imports,
            counsel_models=None,
            summary_model_id=summary_model_id,
            tree_config=tree_search_config,
            adversarial_verification=adversarial_verification,
        )
    else:
        experiment_subgraph = build_experiment_track_subgraph(
            model=model,
            workspace_dir=workspace_dir,
            authorized_imports=authorized_imports,
            counsel_models=None,
            summary_model_id=summary_model_id,
            model_registry=cli_backend_registry,
        )

    # Build all nodes
    nodes: dict[str, Any] = {
        # Pre-track (new v2 flow)
        "persona_council": create_persona_council_node(
            workspace_dir=workspace_dir,
            persona_specs=persona_council_specs,
            max_debate_rounds=persona_debate_rounds,
            synthesis_model=persona_synthesis_model,
            max_post_vote_retries=persona_max_post_vote_retries,
        ),
        "literature_review_agent": _wrap(
            build_literature_review_node(_m("literature_review_agent"), workspace_dir, authorized_imports, **counsel_kwargs),
            "literature_review_agent",
        ),
        "lit_review_gate": build_lit_review_gate_node(workspace_dir, max_attempts=lit_review_max_attempts),
        "brainstorm_agent": _wrap(
            build_brainstorm_node(_m("brainstorm_agent"), workspace_dir, authorized_imports, **counsel_kwargs),
            "brainstorm_agent",
        ),
        "formalize_goals_entry": build_formalize_goals_entry_node(workspace_dir),
        "formalize_goals_agent": _wrap(
            build_formalize_goals_node(_m("formalize_goals_agent"), workspace_dir, authorized_imports, **counsel_kwargs),
            "formalize_goals_agent",
        ),
        "research_plan_writeup_agent": _wrap(
            build_research_plan_writeup_node(_m("research_plan_writeup_agent"), workspace_dir, authorized_imports, **counsel_kwargs),
            "research_plan_writeup_agent",
        ),
        "track_decomposition_gate": build_track_decomposition_gate_node(
            workspace_dir, enable_math_agents=enable_math_agents
        ),
        "milestone_goals": build_milestone_gate_node("research_plan", workspace_dir),
        # Execution tracks (reused from v1)
        "theory_track": theory_track_node,
        "experiment_track": build_track_subgraph_node(experiment_subgraph, "experiment_track_status"),
        "track_merge": build_track_merge_node(workspace_dir=workspace_dir),
        # Post-track verification (new v2 gates)
        "verify_completion": build_verify_completion_node(workspace_dir),
        "formalize_results_agent": _wrap(
            _formalize_results_state_mapper(
                build_formalize_results_node(_m("formalize_results_agent"), workspace_dir, authorized_imports, **counsel_kwargs)
            ),
            "formalize_results_agent",
        ),
        "followup_lit_review": _wrap(
            build_followup_lit_review_node(_m("followup_lit_review"), workspace_dir, authorized_imports, None),
            "followup_lit_review_agent",
        ),
        # Paper production chain (reused from v1)
        "resource_preparation_agent": _wrap(
            build_resource_preparation_node(_m("resource_preparation_agent"), workspace_dir, authorized_imports, **counsel_kwargs),
            "resource_preparation_agent",
        ),
        "writeup_agent": _wrap(
            build_writeup_node(_m("writeup_agent"), workspace_dir, authorized_imports, **counsel_kwargs),
            "writeup_agent",
        ),
        "proofreading_entry": build_proofreading_entry_node(workspace_dir),
        "proofreading_agent": _wrap(
            build_proofreading_node(_m("proofreading_agent"), workspace_dir, authorized_imports, **counsel_kwargs),
            "proofreading_agent",
        ),
        "reviewer_agent": _wrap(
            build_reviewer_node(_m("reviewer_agent"), workspace_dir, authorized_imports, **counsel_kwargs),
            "reviewer_agent",
        ),
        "validation_gate": build_validation_gate_node(),
        "milestone_review": build_milestone_gate_node("review", workspace_dir),
    }

    # Add duality check nodes if enabled
    if enable_duality_check:
        nodes["duality_check"] = create_duality_check_node(
            workspace_dir=workspace_dir,
            check_model=duality_check_model,
        )
        nodes["duality_gate"] = build_duality_gate_node()

    graph = StateGraph(ResearchState)
    for name, node in nodes.items():
        graph.add_node(name, node)

    # --- Edge wiring ---

    # Entry: persona council
    graph.set_entry_point("persona_council")
    graph.add_edge("persona_council", "literature_review_agent")

    # Lit review → gate
    graph.add_edge("literature_review_agent", "lit_review_gate")
    graph.add_conditional_edges(
        "lit_review_gate",
        lit_review_gate_router,
        {
            "persona_council": "persona_council",
            "brainstorm_agent": "brainstorm_agent",
        },
    )

    # Brainstorm → entry gate → formalize goals → writeup → milestone → track execution
    graph.add_edge("brainstorm_agent", "formalize_goals_entry")
    graph.add_edge("formalize_goals_entry", "formalize_goals_agent")
    graph.add_edge("formalize_goals_agent", "research_plan_writeup_agent")
    graph.add_edge("research_plan_writeup_agent", "track_decomposition_gate")
    graph.add_edge("track_decomposition_gate", "milestone_goals")
    graph.add_conditional_edges("milestone_goals", track_router)

    # Track execution → merge → verify
    graph.add_edge("theory_track", "track_merge")
    graph.add_edge("experiment_track", "track_merge")
    graph.add_edge("track_merge", "verify_completion")

    # Verify completion three-way routing
    graph.add_conditional_edges(
        "verify_completion",
        verify_completion_router,
        {
            "formalize_results_agent": "formalize_results_agent",
            "formalize_goals_agent": "formalize_goals_agent",
            "brainstorm_agent": "brainstorm_agent",
        },
    )

    if enable_duality_check:
        # Formalize results → duality check → duality gate
        graph.add_edge("formalize_results_agent", "duality_check")
        graph.add_edge("duality_check", "duality_gate")
        graph.add_conditional_edges(
            "duality_gate",
            duality_gate_router,
            {
                "resource_preparation_agent": "resource_preparation_agent",
                "followup_lit_review": "followup_lit_review",
            },
        )
        # Followup lit review → brainstorm (duality failure path)
        graph.add_edge("followup_lit_review", "brainstorm_agent")
    else:
        # No duality check — go straight to paper production
        graph.add_edge("formalize_results_agent", "resource_preparation_agent")

    # Paper production chain (same as v1)
    graph.add_edge("resource_preparation_agent", "writeup_agent")
    graph.add_edge("writeup_agent", "proofreading_entry")
    graph.add_edge("proofreading_entry", "proofreading_agent")
    graph.add_edge("proofreading_agent", "reviewer_agent")
    graph.add_edge("reviewer_agent", "milestone_review")
    graph.add_edge("milestone_review", "validation_gate")
    graph.add_conditional_edges(
        "validation_gate",
        validation_router,
        {
            END: END,
            "writeup_agent": "writeup_agent",
            "experiment_track": "experiment_track",
            "theory_track": "theory_track",
        },
    )

    compile_kwargs: dict = {}
    if checkpointer is not None:
        compile_kwargs["checkpointer"] = checkpointer

    return graph.compile(**compile_kwargs)


def get_default_checkpointer(workspace_dir: str):
    """Return a SqliteSaver checkpointer scoped to the workspace directory."""
    try:
        from langgraph.checkpoint.sqlite import SqliteSaver
        import sqlite3
        db_path = os.path.join(workspace_dir, "checkpoints.db")
        conn = sqlite3.connect(db_path, check_same_thread=False)
        return SqliteSaver(conn)
    except (ImportError, Exception) as e:
        print(f"Checkpointer unavailable ({e}); resumability disabled.")
        return None
