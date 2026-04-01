"""
LaTeXReflectionTool - Review and improve LaTeX documents for publication quality.

This tool provides intelligent review and improvement suggestions for LaTeX documents.
It analyzes academic writing style, structure, clarity, and technical correctness.

Key changes:
- Tool now writes improved LaTeX content directly to files when requested
- Returns review report with file paths instead of content in JSON
- Eliminates JSON parsing issues for improved content

The WriteupAgent provides document paths and focus areas.
This tool performs deep analysis, generates recommendations, and can write improved versions.
"""

import json
import os
import re
from typing import Dict, Any, List, Optional, Tuple, Type

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field, ConfigDict

from ...cli_completion import cli_completion


class LaTeXReflectionToolInput(BaseModel):
    latex_file_path: str = Field(description="Path to the LaTeX file to review")
    research_context: Optional[str] = Field(
        default=None,
        description=(
            "Natural language description of the research context, experimental setup, key findings, "
            "and any specific aspects to focus the review on. Include details about methodology, "
            "results, limitations, and intended contributions."
        ),
    )
    review_focus: Optional[str] = Field(
        default=None,
        description="Focus area: 'style', 'structure', 'technical', 'citations', 'figures', 'comprehensive' (default)",
    )
    target_venue: Optional[str] = Field(
        default=None,
        description="Target publication venue for style-specific recommendations",
    )
    generate_improvements: Optional[bool] = Field(
        default=None,
        description="Whether to generate improved LaTeX content (default: true)",
    )
    compilation_errors: Optional[str] = Field(
        default=None,
        description="Raw LaTeX compilation errors from tectonic to address specific syntax issues (optional)",
    )


class LaTeXReflectionTool(BaseTool):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    name: str = "latex_reflection_tool"
    description: str = """
    Review and improve LaTeX documents for publication quality and writes files directly.

    This tool provides expert-level review of academic papers, focusing on:
    - Writing clarity and academic style
    - Logical structure and flow
    - Technical accuracy and completeness
    - Citation and reference quality
    - Figure and table integration
    - Publication readiness assessment

    Key features:
    - Analyzes existing LaTeX files and provides detailed feedback
    - Detects structural issues (duplicate sections, repeated content)
    - Writes improved versions directly to files in-place (no versioning)
    - Preserves ALL data by commenting instead of deleting (git provides version history)
    - Returns review report with improvement status

    Output behavior:
    - Always returns detailed review report with recommendations
    - If generate_improvements=true: Writes improved content directly to original file (in-place update)
    - Uses git for version history instead of filesystem versioning
    - Follows CRITICAL PRESERVATION RULES: comments out unused content instead of deleting
    """
    args_schema: Type[BaseModel] = LaTeXReflectionToolInput
    model_id: str = ""
    working_dir: Optional[str] = None

    def __init__(self, model=None, working_dir: Optional[str] = None, **kwargs: Any):
        super().__init__(
            model_id="",
            working_dir=os.path.abspath(working_dir) if working_dir else None,
            **kwargs,
        )
        from ..model_utils import get_raw_model
        raw = get_raw_model(model)
        if raw is not None:
            object.__setattr__(self, "model_id", getattr(raw, "model_id", str(raw)))
        object.__setattr__(self, "working_dir", os.path.abspath(working_dir) if working_dir else None)
        object.__setattr__(self, "available_citations", self._load_citations())

    def _run(
        self,
        latex_file_path: str,
        research_context: Optional[str] = None,
        review_focus: Optional[str] = None,
        target_venue: Optional[str] = None,
        generate_improvements: Optional[bool] = None,
        compilation_errors: Optional[str] = None,
    ) -> str:
        research_context = research_context or ""
        review_focus = review_focus or "comprehensive"
        if generate_improvements is None:
            generate_improvements = True
        compilation_errors = compilation_errors or ""

        try:
            resolved_path = self._safe_path(latex_file_path)

            if not os.path.exists(resolved_path):
                return json.dumps({
                    "error": f"LaTeX file not found: {latex_file_path} (resolved: {resolved_path})",
                    "review_report": None,
                })

            with open(resolved_path, "r", encoding="utf-8") as f:
                latex_content = f.read()

            if not self.model_id:
                return json.dumps({
                    "error": "No LLM model provided to LaTeXReflectionTool",
                    "review_report": None,
                })

            static_analysis = self._perform_static_analysis(latex_content)

            llm_review = self._generate_llm_review(
                latex_content, research_context, review_focus, target_venue, compilation_errors
            )

            improvement_file_path = None
            improvement_status = None
            if generate_improvements:
                improvement_result = self._generate_and_save_improvements(
                    latex_content, llm_review, target_venue, latex_file_path
                )
                improvement_file_path = improvement_result.get("file_path")
                improvement_status = improvement_result.get("status")

            review_report = {
                "file_path": latex_file_path,
                "review_focus": review_focus,
                "target_venue": target_venue,
                "static_analysis": static_analysis,
                "expert_review": llm_review,
                "improvement_file_path": improvement_file_path,
                "improvement_status": improvement_status,
                "model_used": self.model_id if self.model_id else "none",
                "overall_score": self._calculate_overall_score(static_analysis, llm_review),
            }

            return json.dumps(review_report, indent=2)

        except Exception as e:
            error_result = {
                "error": f"LaTeX reflection failed: {str(e)}",
                "review_report": None,
                "file_path": latex_file_path,
            }
            return json.dumps(error_result, indent=2)


    def _safe_path(self, path: str) -> str:
        """Convert path to absolute workspace path with clear error messages for agents."""
        if not self.working_dir:
            return path

        abs_working_dir = os.path.abspath(self.working_dir)

        if os.path.isabs(path):
            abs_path = os.path.abspath(path)
            if abs_path.startswith(abs_working_dir):
                return abs_path
            else:
                raise PermissionError(
                    f"Access denied: The absolute path '{path}' is outside the workspace. "
                    f"Please use a relative path or an absolute path within '{abs_working_dir}'. "
                    f"Example: Use 'paper_workspace/introduction.tex' instead of the full path."
                )
        else:
            abs_path = os.path.abspath(os.path.join(abs_working_dir, path))

            if path.endswith(".tex") and not os.path.exists(abs_path):
                parent_dir = os.path.dirname(abs_path)
                if os.path.exists(parent_dir):
                    raise FileNotFoundError(
                        f"LaTeX file not found: '{path}' does not exist in the workspace. "
                        f"The parent directory exists. Please check the filename."
                    )
                else:
                    raise FileNotFoundError(
                        f"LaTeX file not found: '{path}' does not exist in the workspace. "
                        f"The directory '{os.path.dirname(path)}' was not found."
                    )

            return abs_path

    def _perform_static_analysis(self, latex_content: str) -> Dict[str, Any]:
        """Perform static analysis of the LaTeX document."""
        return {
            "document_stats": self._get_document_stats(latex_content),
            "structure_analysis": self._analyze_structure(latex_content),
            "citation_analysis": self._analyze_citations(latex_content),
            "figure_analysis": self._analyze_figures(latex_content),
            "technical_checks": self._perform_technical_checks(latex_content),
            "structural_validation": self._detect_structural_issues(latex_content),
        }

    def _get_document_stats(self, content: str) -> Dict[str, Any]:
        """Calculate basic document statistics."""
        text_content = re.sub(r"\\[a-zA-Z]+(?:\[[^\]]*\])?(?:\{[^}]*\})*", " ", content)
        text_content = re.sub(r"[{}\\]", " ", text_content)

        words = text_content.split()
        word_count = len([w for w in words if w.strip()])

        return {
            "total_characters": len(content),
            "estimated_word_count": word_count,
            "line_count": len(content.split("\n")),
            "section_count": len(re.findall(r"\\section\{", content)),
            "subsection_count": len(re.findall(r"\\subsection\{", content)),
        }

    def _analyze_structure(self, content: str) -> Dict[str, Any]:
        """Analyze document structure and organization."""
        sections = []
        for match in re.finditer(r"\\section\{([^}]+)\}", content):
            sections.append(match.group(1))

        expected_sections = ["introduction", "related work", "method", "results", "discussion", "conclusion"]
        found_sections = [s.lower() for s in sections]

        missing_sections = [s for s in expected_sections if not any(s in found for found in found_sections)]

        return {
            "sections_found": sections,
            "section_order_check": self._check_section_order(found_sections),
            "missing_common_sections": missing_sections,
            "has_abstract": "\\begin{abstract}" in content or "\\abstract{" in content,
            "has_title": "\\title{" in content,
            "has_author": "\\author{" in content,
        }

    def _analyze_citations(self, content: str) -> Dict[str, Any]:
        """Analyze citation usage and quality."""
        cite_patterns = [r"\\cite\{([^}]+)\}", r"\\citep\{([^}]+)\}", r"\\citet\{([^}]+)\}"]
        citations = []

        for pattern in cite_patterns:
            matches = re.findall(pattern, content)
            for match in matches:
                citations.extend([c.strip() for c in match.split(",")])

        has_bibliography = "\\bibliography{" in content or "\\bibitem{" in content

        return {
            "total_citations": len(citations),
            "unique_citations": len(set(citations)),
            "has_bibliography": has_bibliography,
            "citation_density": len(citations) / max(1, len(content.split("\n"))),
            "uncited_references": self._find_uncited_references(content, citations),
        }

    def _analyze_figures(self, content: str) -> Dict[str, Any]:
        """Analyze figure usage and references."""
        figures = re.findall(r"\\begin\{figure\}.*?\\end\{figure\}", content, re.DOTALL)
        figure_refs = re.findall(r"\\ref\{fig:([^}]+)\}", content)
        figure_labels = re.findall(r"\\label\{fig:([^}]+)\}", content)

        return {
            "figure_count": len(figures),
            "figure_references": len(figure_refs),
            "figure_labels": figure_labels,
            "unreferenced_figures": [label for label in figure_labels if label not in figure_refs],
            "missing_figure_refs": [ref for ref in figure_refs if ref not in figure_labels],
        }

    def _perform_technical_checks(self, content: str) -> Dict[str, Any]:
        """Perform technical LaTeX checks."""
        issues = []

        if "\\begin{document}" not in content:
            issues.append("Missing \\begin{document}")

        if "\\end{document}" not in content:
            issues.append("Missing \\end{document}")

        open_braces = content.count("{")
        close_braces = content.count("}")
        if open_braces != close_braces:
            issues.append(f"Unmatched braces: {open_braces} open, {close_braces} close")

        if "  " in content:
            issues.append("Contains double spaces")

        if "$" in content and content.count("$") % 2 != 0:
            issues.append("Unmatched dollar signs for math mode")

        return {
            "technical_issues": issues,
            "package_usage": self._analyze_packages(content),
            "math_environments": len(re.findall(r"\\begin\{(equation|align|gather)", content)),
        }

    def _analyze_packages(self, content: str) -> List[str]:
        """Extract used LaTeX packages."""
        packages = re.findall(r"\\usepackage(?:\[[^\]]*\])?\{([^}]+)\}", content)
        return list(set(packages))

    def _check_section_order(self, sections: List[str]) -> Dict[str, Any]:
        """Check if sections are in logical order."""
        expected_order = ["abstract", "introduction", "related", "method", "result", "discussion", "conclusion"]

        section_positions = {}
        for i, section in enumerate(sections):
            for j, expected in enumerate(expected_order):
                if expected in section.lower():
                    section_positions[expected] = i
                    break

        order_issues = []
        prev_pos = -1
        for expected in expected_order:
            if expected in section_positions:
                if section_positions[expected] < prev_pos:
                    order_issues.append(f"{expected} appears before expected position")
                prev_pos = section_positions[expected]

        return {
            "order_correct": len(order_issues) == 0,
            "order_issues": order_issues,
            "detected_sections": section_positions,
        }

    def _find_uncited_references(self, content: str, citations: List[str]) -> List[str]:
        """Find bibliography entries that are not cited."""
        bibitem_keys = re.findall(r"\\bibitem\{([^}]+)\}", content)
        uncited = [key for key in bibitem_keys if key not in citations]
        return uncited

    def _generate_llm_review(
        self,
        content: str,
        research_context: str,
        focus: str,
        target_venue: Optional[str],
        compilation_errors: str = "",
    ) -> Dict[str, Any]:
        """Generate expert review using LLM."""
        citations_context = self._format_citations_context()

        system_prompt = f"""You are an expert academic reviewer specializing in AI/ML research papers.
        Provide detailed, constructive feedback on the LaTeX document with focus on: {focus}.

        Target venue: {target_venue or 'General academic venue'}

        AVAILABLE CITATIONS IN REFERENCES.BIB:
        {citations_context}

        When reviewing citations, consider whether proper \\cite{{key}} format is used with available citation keys.

        CRITICAL LATEX FORMATTING ISSUES TO IDENTIFY:
        - Use of \\subref{{}} command (should be \\ref{{}} instead)
        - Math mode errors causing "Missing $ inserted"
        - Dollar signs ($) inside math environments (align, equation) where they shouldn't be
        - Orphaned \\item commands outside list environments
        - Unbalanced braces {{}} in LaTeX commands
        - Unclosed environments (equations, itemize, enumerate)
        - Improper figure/equation referencing

        Analyze the document for:
        - Academic writing quality and clarity
        - Logical structure and flow
        - Technical accuracy and completeness
        - Citation appropriateness and coverage
        - Figure integration and referencing
        - LaTeX formatting correctness and compilation safety
        - Alignment with actual research context and findings
        - Overall publication readiness

        Provide specific, actionable feedback with examples from the text."""

        context_section = f"\n\nResearch Context:\n{research_context}" if research_context.strip() else ""

        compilation_errors_section = ""
        if compilation_errors.strip():
            compilation_errors_section = (
                f"\n\nCOMPILATION ERRORS TO FIX:\n{compilation_errors}\n\n"
                "IMPORTANT: Prioritize fixing these compilation errors in your review and improvements."
            )

        user_prompt = f"""Please review this LaTeX document:{context_section}{compilation_errors_section}

LaTeX Content:
{content}

Focus area: {focus}

Provide a structured review with:
1. Strengths of the paper
2. Areas for improvement
3. Specific suggestions with examples
4. Alignment with research context (if provided)
5. Publication readiness assessment
6. Score (1-10) for writing quality"""

        try:
            full_prompt = f"{system_prompt}\n\n{user_prompt}"
            output = cli_completion(full_prompt)

            return {
                "raw_review": output,
                "focus_area": focus,
                "recommendations": self._extract_recommendations(output),
                "quality_score": self._extract_score(output),
            }
        except Exception as e:
            return {
                "error": f"LLM review failed: {str(e)}",
                "raw_review": None,
            }

    def _generate_and_save_improvements(
        self,
        content: str,
        review: Dict[str, Any],
        target_venue: Optional[str],
        original_file_path: str,
    ) -> Dict[str, Any]:
        """Generate improved LaTeX content based on review and save in-place (no versioning)."""
        try:
            improved_content = self._generate_improved_content(content, review, target_venue)

            if improved_content is None:
                return {"status": "error", "file_path": None, "message": "Failed to generate improvements"}

            if self.working_dir:
                file_path = self._safe_path(original_file_path)
            else:
                file_path = original_file_path

            with open(file_path, "w", encoding="utf-8") as f:
                f.write(improved_content)

            if self.working_dir:
                relative_path = os.path.relpath(file_path, self.working_dir)
            else:
                relative_path = file_path

            return {
                "status": "success",
                "file_path": relative_path,
                "absolute_path": file_path,
                "message": f"Improved LaTeX content written to {os.path.basename(file_path)} (in-place update, git provides version history)",
            }

        except Exception as e:
            return {
                "status": "error",
                "file_path": None,
                "message": f"Failed to save improvements: {str(e)}",
            }

    def _generate_improved_content(
        self,
        content: str,
        review: Dict[str, Any],
        target_venue: Optional[str],
    ) -> Optional[str]:
        """Generate improved LaTeX content based on review."""
        if "error" in review:
            return None

        research_context = self._extract_research_context(content)
        citations_context = self._format_citations_context()

        system_prompt = f"""You are an expert academic editor. Based on the review feedback,
        improve the LaTeX document while STRICTLY preserving its core research content and topic.

        Target venue: {target_venue or 'General academic venue'}

        AVAILABLE CITATIONS IN REFERENCES.BIB:
        {citations_context}

        CRITICAL CITATION INSTRUCTIONS:
        - NEVER use [? ] or [?] placeholders for citations
        - ALWAYS use proper \\cite{{key}} format with actual citation keys from the list above
        - If you need a citation but don't find a perfect match, use the closest relevant one
        - If no relevant citations exist, use [cite: brief_description] format (lowercase, colon space) for automatic resolution

        🚨 CRITICAL DATA PRESERVATION RULES - NEVER VIOLATE THESE 🚨:

        **NEVER DELETE DATA - COMMENT OUT INSTEAD**:
        1. When removing/simplifying content, ALWAYS preserve original as LaTeX comments
        2. Use format: % REFLECTION_PRESERVED (YYYY-MM-DD): Clear reason for preservation
        3. Comment out the entire original section, don't delete it
        4. This allows immediate recovery without git archaeology

        🚨 CRITICAL CONTENT PRESERVATION REQUIREMENTS 🚨:
        - MUST preserve the exact research topic, methodology, and findings from the original
        - MUST NOT change the research domain, problem statement, or experimental setup
        - MUST NOT introduce new research topics, methods, or datasets not in the original
        - MUST NOT hallucinate different technical approaches or architectures
        - MUST maintain all original research claims, metrics, and results
        - Improvements should be LIMITED to writing quality, clarity, and academic style ONLY

        🚨 CRITICAL NUMERICAL DATA PRESERVATION - ABSOLUTE REQUIREMENTS 🚨:

        **YOU MUST PRESERVE EVERY NUMERICAL VALUE EXACTLY**:
        - ✅ PRESERVE exact F1 scores, accuracy values, hyperparameters, model specs, dataset sizes
        - ❌ NEVER round, approximate, or modify ANY numerical value
        - ❌ NEVER change specific numbers to ranges
        - ❌ NEVER replace exact values with qualitative descriptions

        DETECTED RESEARCH CONTEXT TO PRESERVE:
        {research_context}

        CRITICAL OUTPUT REQUIREMENTS:
        - Output ONLY clean LaTeX content
        - NO conversational text, explanations, or markdown formatting
        - Start directly with LaTeX content (e.g., \\section{{...}} or \\begin{{abstract}})
        - End with the last LaTeX command
        - PRESERVE citation placeholders in [cite: key] format exactly as-is (do NOT convert to \\cite{{}})

        **REMEMBER: Your job is to polish the WRITING, not to change the SCIENCE. Every number is sacred.**"""

        user_prompt = f"""Original LaTeX content:
{content}

Review feedback:
{review.get('raw_review', '')}

CRITICAL: Generate ONLY the improved LaTeX content that preserves the EXACT same research topic, methods, and findings. Make ONLY writing quality improvements without changing any research content."""

        try:
            full_prompt = f"{system_prompt}\n\n{user_prompt}"
            improved_content = cli_completion(full_prompt)
            cleaned_content = self._clean_improved_content(improved_content)

            if not self._validate_content_consistency(content, cleaned_content):
                return "Content validation failed: Improved content changed research topic or key findings. Rejecting changes to prevent hallucination."

            return cleaned_content
        except Exception as e:
            return f"Error generating improvements: {str(e)}"

    def _extract_recommendations(self, review_text: str) -> List[str]:
        """Extract specific recommendations from review text."""
        recommendations = []
        lines = review_text.split("\n")
        for line in lines:
            line = line.strip()
            if re.match(r"^\d+\.", line) or line.startswith("- ") or line.startswith("* "):
                recommendations.append(line)
        return recommendations[:10]

    def _extract_score(self, review_text: str) -> Optional[float]:
        """Extract numerical score from review text."""
        score_patterns = [
            r"score[:\s]*(\d+(?:\.\d+)?)[/\s]*10",
            r"(\d+(?:\.\d+)?)[/\s]*10",
            r"quality[:\s]*(\d+(?:\.\d+)?)",
        ]

        for pattern in score_patterns:
            match = re.search(pattern, review_text.lower())
            if match:
                try:
                    return float(match.group(1))
                except ValueError:
                    continue

        return None

    def _clean_improved_content(self, content: str) -> str:
        """Clean the improved LaTeX content."""
        content = content.replace("```latex", "").replace("```", "").strip()
        content = re.sub(r"\n{3,}", "\n\n", content)
        return content

    def _calculate_overall_score(
        self, static_analysis: Dict[str, Any], llm_review: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate overall document quality score."""
        scores: Dict[str, Any] = {}

        static_score = 10.0

        if static_analysis["technical_checks"]["technical_issues"]:
            static_score -= len(static_analysis["technical_checks"]["technical_issues"]) * 1.0

        if static_analysis["structure_analysis"]["missing_common_sections"]:
            static_score -= len(static_analysis["structure_analysis"]["missing_common_sections"]) * 0.5

        scores["technical_score"] = max(0, static_score)

        if "quality_score" in llm_review and llm_review["quality_score"]:
            scores["writing_quality_score"] = llm_review["quality_score"]
        else:
            scores["writing_quality_score"] = 7.0

        scores["overall_score"] = (scores["technical_score"] * 0.3 + scores["writing_quality_score"] * 0.7)

        return scores

    def _detect_structural_issues(self, content: str) -> Dict[str, Any]:
        """Detect structural issues like duplicate sections and repeated content."""
        issues = []
        warnings = []

        section_pattern = r"\\section\*?\{([^}]+)\}"
        sections = re.findall(section_pattern, content)
        section_counts: Dict[str, int] = {}
        for section in sections:
            normalized = section.lower().strip()
            section_counts[normalized] = section_counts.get(normalized, 0) + 1

        duplicates = {title: count for title, count in section_counts.items() if count > 1}
        if duplicates:
            for title, count in duplicates.items():
                issues.append(f"Duplicate section '{title}' appears {count} times")

        subsection_pattern = r"\\subsection\*?\{([^}]+)\}"
        subsections = re.findall(subsection_pattern, content)
        subsection_counts: Dict[str, int] = {}
        for subsection in subsections:
            normalized = subsection.lower().strip()
            subsection_counts[normalized] = subsection_counts.get(normalized, 0) + 1

        duplicate_subsections = {title: count for title, count in subsection_counts.items() if count > 1}
        if duplicate_subsections:
            for title, count in duplicate_subsections.items():
                warnings.append(f"Duplicate subsection '{title}' appears {count} times")

        paragraphs = self._extract_text_paragraphs(content)
        repeated_paragraphs = self._find_similar_paragraphs(paragraphs, threshold=0.8)
        if repeated_paragraphs:
            for i, j, similarity in repeated_paragraphs:
                issues.append(f"Paragraphs {i+1} and {j+1} are {similarity:.1%} similar (likely duplicated)")

        repeated_sentences = self._find_repeated_sentences(content)
        if repeated_sentences:
            for sentence, locations in repeated_sentences.items():
                if len(locations) > 1:
                    warnings.append(f"Sentence repeated {len(locations)} times: '{sentence[:50]}...'")

        structural_problems = self._check_document_structure(content)
        issues.extend(structural_problems)

        return {
            "duplicate_sections": list(duplicates.keys()),
            "duplicate_subsections": list(duplicate_subsections.keys()),
            "repeated_paragraph_count": len(repeated_paragraphs),
            "repeated_sentence_count": len(repeated_sentences),
            "structural_issues": issues,
            "structural_warnings": warnings,
            "has_structural_problems": len(issues) > 0,
            "total_issues": len(issues),
            "total_warnings": len(warnings),
        }

    def _extract_text_paragraphs(self, content: str) -> List[str]:
        """Extract text paragraphs from LaTeX content, removing commands."""
        content = re.sub(r"%.*$", "", content, flags=re.MULTILINE)
        content = re.sub(
            r"\\begin\{(?:figure|table|equation|align)\}.*?\\end\{(?:figure|table|equation|align)\}",
            "",
            content,
            flags=re.DOTALL,
        )
        content = re.sub(r"\\(?:textbf|textit|emph|texttt|underline)\{([^}]*)\}", r"\1", content)
        content = re.sub(r"\\(?:section|subsection|subsubsection)\*?\{[^}]*\}", "", content)
        content = re.sub(r"\\(?:cite|ref|label)\{[^}]*\}", "", content)
        content = re.sub(r"\\[a-zA-Z]+(?:\[[^\]]*\])?(?:\{[^}]*\})*", "", content)

        paragraphs = content.split("\n\n")

        cleaned_paragraphs = []
        for para in paragraphs:
            cleaned = re.sub(r"\s+", " ", para.strip())
            cleaned = re.sub(r"[{}\\]", "", cleaned)
            if len(cleaned) > 50 and not cleaned.startswith("\\"):
                cleaned_paragraphs.append(cleaned)

        return cleaned_paragraphs

    def _find_similar_paragraphs(self, paragraphs: List[str], threshold: float = 0.8) -> List[Tuple[int, int, float]]:
        """Find paragraphs with high similarity (indicating duplication)."""
        similar_pairs = []

        for i in range(len(paragraphs)):
            for j in range(i + 1, len(paragraphs)):
                similarity = self._calculate_text_similarity(paragraphs[i], paragraphs[j])
                if similarity >= threshold:
                    similar_pairs.append((i, j, similarity))

        return similar_pairs

    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts using word overlap."""
        if not text1 or not text2:
            return 0.0

        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))

        return intersection / union if union > 0 else 0.0

    def _find_repeated_sentences(self, content: str) -> Dict[str, List[int]]:
        """Find sentences that appear multiple times in the document."""
        sentences = re.split(r"[.!?]+", content)
        sentence_locations: Dict[str, List[int]] = {}

        for i, sentence in enumerate(sentences):
            cleaned = re.sub(r"\\[a-zA-Z]+(?:\[[^\]]*\])?(?:\{[^}]*\})*", "", sentence)
            cleaned = re.sub(r"\s+", " ", cleaned.strip())

            if len(cleaned) > 30 and not cleaned.startswith("\\"):
                if cleaned in sentence_locations:
                    sentence_locations[cleaned].append(i)
                else:
                    sentence_locations[cleaned] = [i]

        return {sentence: locations for sentence, locations in sentence_locations.items() if len(locations) > 1}

    def _check_document_structure(self, content: str) -> List[str]:
        """Check for structural problems in the document."""
        problems = []

        if "\\begin{document}" not in content:
            problems.append("Missing \\begin{document} - document may be incomplete")

        if "\\end{document}" not in content:
            problems.append("Missing \\end{document} - document may be incomplete")

        abstract_count = len(re.findall(r"\\begin\{abstract\}", content))
        if abstract_count > 1:
            problems.append(f"Multiple abstract environments found ({abstract_count}) - should have only one")

        title_count = len(re.findall(r"\\title\{", content))
        if title_count > 1:
            problems.append(f"Multiple title commands found ({title_count}) - should have only one")

        begin_commands = re.findall(r"\\begin\{([^}]+)\}", content)
        end_commands = re.findall(r"\\end\{([^}]+)\}", content)

        begin_counts: Dict[str, int] = {}
        for cmd in begin_commands:
            begin_counts[cmd] = begin_counts.get(cmd, 0) + 1

        end_counts: Dict[str, int] = {}
        for cmd in end_commands:
            end_counts[cmd] = end_counts.get(cmd, 0) + 1

        for env in set(begin_commands + end_commands):
            begin_count = begin_counts.get(env, 0)
            end_count = end_counts.get(env, 0)
            if begin_count != end_count:
                problems.append(f"Unmatched {env} environment: {begin_count} begin, {end_count} end")

        return problems

    def _extract_research_context(self, content: str) -> str:
        """Extract key research elements to preserve topic consistency."""
        context_elements = []

        section_titles = re.findall(r"\\section\*?\{([^}]+)\}", content)
        if section_titles:
            context_elements.append(f"Paper sections: {', '.join(section_titles)}")

        research_indicators = [
            r"([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+){1,3})\s*\(([A-Z]{2,6})\)",
            r"\\textbf\{([^}]+(?:Framework|Method|Model|Algorithm|Approach|Network|System))\}",
            r"(?:benchmark|dataset|corpus)(?:s)?\s*:?\s*([A-Za-z0-9\-\_]+)",
            r"(?:evaluate|experiment|test)(?:d?)?\s+on\s+(?:the\s+)?([A-Za-z0-9\-\_\s]+)",
            r"([0-9]+(?:\.[0-9]+)?%?)\s*(?:success rate|accuracy|performance|score)",
            r"(?:achieving|achieve[sd]?)\s+(?:a\s+)?([0-9]+(?:\.[0-9]+)?%?)",
        ]

        extracted_terms: set = set()
        for pattern in research_indicators:
            matches = re.findall(pattern, content, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    extracted_terms.update([m.strip() for m in match if m.strip()])
                else:
                    extracted_terms.add(match.strip())

        if extracted_terms:
            context_elements.append(f"Key research terms: {', '.join(list(extracted_terms)[:10])}")

        abstract_match = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", content, re.DOTALL)
        if abstract_match:
            abstract_text = abstract_match.group(1).strip()
            first_sentence = re.split(r"[.!?]", abstract_text)[0].strip()
            if len(first_sentence) > 20:
                context_elements.append(f"Research topic: {first_sentence}")

        return "\n".join(context_elements) if context_elements else "No specific research context detected"

    def _validate_content_consistency(self, original: str, improved: str) -> bool:
        """Validate that improved content preserves the original research topic and key findings."""
        original_terms = self._extract_key_terms(original)
        improved_terms = self._extract_key_terms(improved)

        if not original_terms or not improved_terms:
            return True

        overlap_ratio = len(original_terms.intersection(improved_terms)) / len(original_terms)

        if overlap_ratio < 0.6:
            return False

        new_terms = improved_terms - original_terms
        technical_patterns = [
            r"\b[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)+\b",
            r"\b[A-Z]{2,6}\b",
        ]

        new_technical_terms: set = set()
        improved_text = " ".join(new_terms)
        for pattern in technical_patterns:
            matches = re.findall(pattern, improved_text)
            new_technical_terms.update(matches)

        if len(new_technical_terms) > 3:
            return False

        return True

    def _extract_key_terms(self, content: str) -> set:
        """Extract key research terms from content."""
        terms: set = set()

        patterns = [
            r"\b[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+){1,3}\b",
            r"\b[A-Z]{2,6}\b",
            r"\\textbf\{([^}]+)\}",
            r"\\emph\{([^}]+)\}",
        ]

        for pattern in patterns:
            matches = re.findall(pattern, content)
            for match in matches:
                if isinstance(match, tuple):
                    terms.update([m.strip() for m in match if m.strip()])
                else:
                    term = match.strip()
                    if len(term) > 2:
                        terms.add(term)

        return terms

    def _load_citations(self) -> Dict[str, str]:
        """Load and parse available citations from references.bib file."""
        citations: Dict[str, str] = {}

        if not self.working_dir:
            return citations

        bib_paths = [
            os.path.join(self.working_dir, "paper_workspace", "references.bib"),
            os.path.join(self.working_dir, "references.bib"),
        ]

        for bib_path in bib_paths:
            if os.path.exists(bib_path):
                try:
                    with open(bib_path, "r", encoding="utf-8") as f:
                        content = f.read()

                    bib_entries = content.split("@")[1:]

                    for entry in bib_entries:
                        key_match = re.search(r"^(\w+)\{([^,]+),", entry)
                        if not key_match:
                            continue

                        entry_type, cite_key = key_match.groups()
                        cite_key = cite_key.strip()

                        title_match = re.search(r"title\s*=\s*\{([^}]*)\}", entry, re.DOTALL)
                        if title_match:
                            title = title_match.group(1).strip()
                            title = " ".join(title.split())
                        else:
                            title = "Unknown title"

                        citations[cite_key] = title

                    break

                except Exception as e:
                    print(f"Warning: Could not parse {bib_path}: {e}")
                    continue

        return citations

    def _format_citations_context(self) -> str:
        """Format available citations for inclusion in LLM prompts."""
        available_citations = getattr(self, "available_citations", {})
        if not available_citations:
            return "No citations available in references.bib"

        context = "Available citations (use \\cite{key} format):\n"
        for cite_key, title in available_citations.items():
            context += f"- {cite_key}: {title}\n"

        return context
