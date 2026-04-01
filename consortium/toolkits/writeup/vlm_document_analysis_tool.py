"""
VLMDocumentAnalysisTool - Analyze scientific figures and PDF documents using Vision-Language Models.

This tool provides comprehensive visual analysis including:
- Scientific figure analysis (plots, charts, visualizations)
- PDF document visual validation (layout, citations, figures)
- Content description and quality assessment
- Technical element evaluation (axes, legends, labels)
- Publication quality evaluation and error detection
- Layout problem identification (formula overflow, spacing issues)
- Missing element detection (figures, citations, references)

Uses the VLM functionality from consortium.llm for image and document analysis.
"""

import json
import os
import re
import tempfile
import hashlib
from typing import List, Dict, Any, Optional, Tuple, Type

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field, ConfigDict

import subprocess as _sp


def _vlm_via_cli(prompt: str, image_paths: list[str], model: str | None = None) -> str:
    """Analyse images via Claude Code CLI (replaces direct API VLM calls).

    Claude Code can read image files natively via its Read tool.
    """
    image_refs = "\n".join(f"- {p}" for p in image_paths if os.path.isfile(p))
    full_prompt = (
        f"Read and analyse the following image file(s):\n{image_refs}\n\n{prompt}\n\n"
        "Provide your analysis as plain text."
    )
    cmd = ["claude", "-p", "--output-format", "text", "--max-turns", "5"]
    if model:
        cmd.extend(["--model", model])
    cmd.extend(["--allowedTools", "Read"])

    try:
        result = _sp.run(
            cmd, input=full_prompt, capture_output=True, text=True, timeout=120,
        )
        return result.stdout.strip() or result.stderr.strip() or "(no output)"
    except (_sp.TimeoutExpired, FileNotFoundError) as e:
        return f"VLM analysis unavailable: {e}"

# Try to import PyMuPDF for PDF processing
try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False


class VLMDocumentAnalysisToolInput(BaseModel):
    file_paths: str = Field(
        description="Single file path or JSON list of file paths to analyze (supports images: PNG, JPG, PDF pages)"
    )
    analysis_focus: Optional[str] = Field(
        default=None,
        description=(
            "Analysis mode (default: 'comprehensive'):\n"
            "**IMAGE-ONLY MODES (PNG, JPG, etc.):**\n"
            "- 'image_content': Extract scientific insights and research conclusions from figures/plots\n"
            "- 'image_quality': Assess figure publication readiness, visual clarity, professional presentation\n"
            "- 'image_trends': Focus on data patterns, experimental trends, quantitative results in visualizations\n"
            "- 'image_technical': Analyze technical details like axes labels, legends, statistical significance\n"
            "- 'comprehensive': Complete image analysis combining content, quality, and technical assessment\n"
            "**PDF-ONLY MODES:**\n"
            "- 'pdf_reading': Read and analyze PDF research papers\n"
            "- 'pdf_validation': Check LaTeX compilation quality - missing citations, broken figures, layout errors"
        ),
    )


class VLMDocumentAnalysisTool(BaseTool):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    name: str = "vlm_document_analysis_tool"
    description: str = """
    Analyze scientific figures and PDF documents using Vision-Language Models for comprehensive visual validation.

    This tool is essential for:
    - Understanding experimental results presented in figures
    - Creating accurate figure captions and descriptions
    - Identifying key trends and findings in visualizations
    - Assessing figure quality for publication standards
    - PDF VISUAL VALIDATION: Detecting layout problems, missing figures, missing citations
    - QUALITY ASSURANCE: Identifying severe layout issues (formulas exceeding page boundaries)
    - CITATION VERIFICATION: Detecting question marks indicating missing bibliography entries
    - FIGURE VALIDATION: Identifying empty spaces where figures should appear

    Use this tool when:
    - You need to describe figures in your LaTeX writeup
    - You want to understand what insights figures convey
    - You need to assess if figures are publication-ready
    - Validating final PDF for visual correctness before task completion
    - Checking compiled PDF for layout problems and missing elements
    - You are writing results/discussion sections referencing figures

    The tool analyzes: line plots, bar charts, heatmaps, scatter plots, subplots,
    PDF pages, LaTeX-compiled documents, and other scientific visualizations.

    Input: Path to image file(s), PDF file(s), or list of paths
    Output: Detailed structured analysis with scientific insights and visual validation results
    """
    args_schema: Type[BaseModel] = VLMDocumentAnalysisToolInput
    vlm_model: str = ""
    working_dir: Optional[str] = None

    def __init__(self, model=None, working_dir: Optional[str] = None, **kwargs: Any):
        super().__init__(
            vlm_model=os.getenv("CONSORTIUM_VLM_MODEL", "claude-sonnet-4-5"),
            working_dir=os.path.abspath(working_dir) if working_dir else None,
            **kwargs,
        )

    def _run(
        self,
        file_paths: str,
        analysis_focus: Optional[str] = None,
    ) -> str:
        analysis_focus = analysis_focus or "comprehensive"
        try:
            if isinstance(file_paths, list):
                paths = file_paths
            elif isinstance(file_paths, str):
                if file_paths.startswith("[") and file_paths.endswith("]"):
                    paths = json.loads(file_paths)
                else:
                    paths = [file_paths.strip()]
            else:
                paths = [str(file_paths)]

            valid_paths = []
            for path in paths:
                try:
                    resolved_path = self._safe_path(path) if self.working_dir else path
                    print(f"Resolved Path {resolved_path}")
                    if os.path.exists(resolved_path):
                        valid_paths.append(resolved_path)
                    else:
                        print(f"Warning: Image path does not exist: {path}")
                except PermissionError as e:
                    print(f"Warning: Access denied to path {path}: {e}")

            if not valid_paths:
                return json.dumps({
                    "error": "No valid file paths found",
                    "analysis": None,
                })

            pdf_files = [p for p in valid_paths if p.lower().endswith(".pdf")]
            image_files = [p for p in valid_paths if not p.lower().endswith(".pdf")]

            if pdf_files:
                if analysis_focus == "pdf_validation":
                    cached = self._load_pdf_validation_cache(pdf_files[0])
                    if cached is not None:
                        return json.dumps(cached, indent=2)
                    result_str = self._analyze_pdf_comprehensively(pdf_files[0])
                    self._save_pdf_validation_cache(pdf_files[0], result_str)
                    return result_str
                elif analysis_focus == "pdf_reading":
                    return self._analyze_pdf_for_research(pdf_files[0], analysis_focus)
                else:
                    return self._analyze_pdf_for_research(pdf_files[0], analysis_focus)

            if not image_files:
                return json.dumps({
                    "error": "No valid image or PDF files found",
                    "provided_files": valid_paths,
                })

            analysis_prompt = self._get_analysis_prompt(analysis_focus, len(image_files))
            response = _vlm_via_cli(analysis_prompt, image_files, self.vlm_model)

            structured_analysis = self._structure_analysis(response, valid_paths, analysis_focus)
            return json.dumps(structured_analysis, indent=2)

        except Exception as e:
            error_result = {
                "error": f"VLM document analysis failed: {str(e)}",
                "analysis": None,
                "file_paths": file_paths,
                "focus": analysis_focus,
            }
            return json.dumps(error_result, indent=2)


    def _analyze_pdf_comprehensively(self, pdf_path: str) -> str:
        """Comprehensive PDF analysis pipeline."""
        if not PYMUPDF_AVAILABLE:
            return json.dumps({
                "error": "PyMuPDF not available",
                "message": "Install PyMuPDF with: pip install PyMuPDF",
                "fallback_suggestion": "Use LaTeX compilation analysis instead",
            })

        try:
            safe_pdf_path = self._safe_path(pdf_path) if self.working_dir else pdf_path
            extracted_data = self._extract_pdf_content(safe_pdf_path)

            image_analyses = []
            for image_info in extracted_data["images"]:
                questions = self._generate_context_questions(image_info["context"], image_info["expected_content"])
                vlm_analysis = self._analyze_image_with_questions(image_info["image_path"], questions)
                image_analyses.append({
                    "image_id": image_info["image_id"],
                    "context": image_info["context"],
                    "questions": questions,
                    "vlm_analysis": vlm_analysis,
                    "image_path": image_info["image_path"],
                })

            final_text = self._reconstruct_document_with_analysis(extracted_data["text"], image_analyses)
            publication_issues = self._identify_publication_issues(extracted_data, image_analyses)

            return json.dumps({
                "status": "success",
                "analysis_type": "comprehensive_pdf_analysis",
                "original_text_length": len(extracted_data["text"]),
                "images_analyzed": len(image_analyses),
                "publication_issues": publication_issues,
                "reconstructed_text": final_text,
                "image_analyses": image_analyses,
                "pdf_path": safe_pdf_path,
            }, indent=2)

        except Exception as e:
            return json.dumps({
                "error": f"Comprehensive PDF analysis failed: {str(e)}",
                "pdf_path": pdf_path,
            })

    def _cache_root(self) -> str:
        base = self.working_dir if self.working_dir else os.getcwd()
        cache_dir = os.path.join(base, ".consortium_cache", "vlm_pdf_validation")
        os.makedirs(cache_dir, exist_ok=True)
        return cache_dir

    @staticmethod
    def _file_sha256(path: str) -> str:
        hasher = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    def _cache_path_for_pdf(self, pdf_path: str) -> str:
        digest = self._file_sha256(pdf_path)
        return os.path.join(self._cache_root(), f"{digest}.json")

    def _load_pdf_validation_cache(self, pdf_path: str) -> Optional[Dict[str, Any]]:
        try:
            cache_path = self._cache_path_for_pdf(pdf_path)
            if not os.path.exists(cache_path):
                return None
            with open(cache_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            if isinstance(payload, dict):
                payload["cache_hit"] = True
                payload["cache_source"] = "vlm_pdf_validation"
                return payload
            return None
        except Exception:
            return None

    def _save_pdf_validation_cache(self, pdf_path: str, result_str: str) -> None:
        try:
            payload = json.loads(result_str)
            if not isinstance(payload, dict):
                return
            if payload.get("error"):
                return
            payload["cache_hit"] = False
            payload["cache_source"] = "vlm_pdf_validation"
            cache_path = self._cache_path_for_pdf(pdf_path)
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
        except Exception:
            return

    def _analyze_pdf_for_research(self, pdf_path: str, analysis_focus: str) -> str:
        """Research paper analysis pipeline."""
        if not PYMUPDF_AVAILABLE:
            return json.dumps({
                "error": "PyMuPDF not available",
                "message": "Install PyMuPDF with: pip install PyMuPDF",
                "fallback_suggestion": "Use individual image analysis instead",
            })

        try:
            safe_pdf_path = self._safe_path(pdf_path) if self.working_dir else pdf_path
            extracted_data = self._extract_pdf_content(safe_pdf_path)

            image_analyses = []
            for image_info in extracted_data["images"]:
                research_questions = self._generate_research_questions(
                    image_info["context"],
                    image_info["expected_content"],
                    analysis_focus,
                )
                analysis_result = self._analyze_image_with_questions(
                    image_info["image_path"],
                    research_questions,
                )
                image_analyses.append({
                    "image_id": image_info["image_id"],
                    "page_number": image_info["page_number"],
                    "context": image_info["context"],
                    "expected_content": image_info["expected_content"],
                    "research_analysis": analysis_result,
                    "image_path": image_info["image_path"],
                })

            research_insights = self._extract_research_insights(extracted_data["text"], image_analyses, analysis_focus)

            return json.dumps({
                "status": "success",
                "analysis_type": f"research_paper_analysis_{analysis_focus}",
                "document_length": len(extracted_data["text"]),
                "images_analyzed": len(image_analyses),
                "research_insights": research_insights,
                "full_text": extracted_data["text"][:5000] + "..." if len(extracted_data["text"]) > 5000 else extracted_data["text"],
                "image_analyses": image_analyses,
                "pdf_path": safe_pdf_path,
            }, indent=2)

        except Exception as e:
            return json.dumps({
                "error": f"Research PDF analysis failed: {str(e)}",
                "pdf_path": pdf_path,
            })

    def _extract_pdf_content(self, pdf_path: str) -> Dict[str, Any]:
        """Extract text and images from PDF using PyMuPDF."""
        doc = fitz.open(pdf_path)
        full_text = ""
        images: List[Dict[str, Any]] = []
        image_counter = 0
        total_pages = len(doc)

        for page_num in range(total_pages):
            page = doc.load_page(page_num)
            page_text = page.get_text()
            image_list = page.get_images()

            for img_index, img in enumerate(image_list):
                try:
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    image_ext = base_image["ext"]

                    if not self._is_valid_image_data(image_bytes, image_ext):
                        print(f"Warning: Skipping invalid/missing image data on page {page_num}, image {img_index}")
                        context = self._extract_image_context(page_text, page_num, img_index)
                        images.append({
                            "image_id": image_counter,
                            "image_path": None,
                            "page_number": page_num,
                            "context": context,
                            "expected_content": self._infer_expected_content(context),
                            "placeholder": f"[MISSING_IMAGE_{image_counter}_PLACEHOLDER]",
                            "status": "missing_or_invalid",
                        })
                        image_counter += 1
                        continue

                    temp_file = tempfile.NamedTemporaryFile(
                        suffix=f"_image_{image_counter}.{image_ext}", delete=False
                    )
                    temp_file.write(image_bytes)
                    temp_file.close()

                except Exception as e:
                    print(f"Warning: Failed to extract image on page {page_num}, image {img_index}: {e}")
                    context = self._extract_image_context(page_text, page_num, img_index)
                    images.append({
                        "image_id": image_counter,
                        "image_path": None,
                        "page_number": page_num,
                        "context": context,
                        "expected_content": self._infer_expected_content(context),
                        "placeholder": f"[FAILED_IMAGE_{image_counter}_PLACEHOLDER]",
                        "status": "extraction_failed",
                    })
                    image_counter += 1
                    continue

                context = self._extract_image_context(page_text, page_num, img_index)
                expected_content = self._infer_expected_content(context)

                placeholder = f"[IMAGE_{image_counter}_PLACEHOLDER]"
                if page_text.strip():
                    lines = page_text.split("\n")
                    for i, line in enumerate(lines):
                        if line.strip() and not line.startswith(" ") and i < len(lines) - 1:
                            lines.insert(i + 1, placeholder)
                            break
                    page_text = "\n".join(lines)
                else:
                    page_text = placeholder

                images.append({
                    "image_id": image_counter,
                    "image_path": temp_file.name,
                    "page_number": page_num,
                    "context": context,
                    "expected_content": expected_content,
                    "placeholder": placeholder,
                })
                image_counter += 1

            full_text += f"\n--- Page {page_num + 1} ---\n{page_text}\n"

        doc.close()

        return {
            "text": full_text,
            "images": images,
            "total_pages": total_pages,
            "total_images": image_counter,
        }

    def _extract_image_context(self, page_text: str, page_num: int, img_index: int) -> str:
        """Extract contextual text around where an image appears."""
        lines = page_text.split("\n")
        context_lines: List[str] = []

        figure_patterns = [
            r"[Ff]igure\s+\d+",
            r"[Tt]able\s+\d+",
            r"[Ff]ig\.\s+\d+",
            r"[Pp]lot\s+\d+",
            r"[Gg]raph\s+\d+",
            r"[Cc]hart\s+\d+",
        ]

        for i, line in enumerate(lines):
            for pattern in figure_patterns:
                if re.search(pattern, line):
                    start = max(0, i - 2)
                    end = min(len(lines), i + 3)
                    context_lines.extend(lines[start:end])
                    break

        if not context_lines and len(lines) > 5:
            mid = len(lines) // 2
            context_lines = lines[max(0, mid - 2) : min(len(lines), mid + 3)]

        return " ".join(context_lines).strip()

    def _infer_expected_content(self, context: str) -> str:
        """Infer what type of content the image should contain based on context."""
        context_lower = context.lower()

        if any(word in context_lower for word in ["accuracy", "performance", "score", "metric"]):
            return "performance_chart"
        elif any(word in context_lower for word in ["loss", "error", "training"]):
            return "training_curve"
        elif any(word in context_lower for word in ["comparison", "versus", "vs", "compare"]):
            return "comparison_plot"
        elif any(word in context_lower for word in ["distribution", "histogram", "density"]):
            return "distribution_plot"
        elif any(word in context_lower for word in ["table", "matrix", "grid"]):
            return "table_or_matrix"
        elif any(word in context_lower for word in ["architecture", "model", "network"]):
            return "architecture_diagram"
        else:
            return "general_scientific_figure"

    def _generate_context_questions(self, context: str, expected_content: str) -> List[str]:
        """Generate specific questions about the image based on textual context."""
        base_questions = [
            "What type of visualization is shown in this image?",
            "What are the main data points or values that can be extracted?",
            "Are there any missing elements, broken displays, or quality issues?",
            "Does the image match publication quality standards?",
        ]

        content_specific: Dict[str, List[str]] = {
            "performance_chart": [
                "What performance metrics are being compared?",
                "What are the specific numerical values shown?",
                "Are error bars or confidence intervals present?",
                "Which method or approach performs best?",
            ],
            "training_curve": [
                "What is being plotted on the x and y axes?",
                "Does the curve show convergence or instability?",
                "Are there multiple curves being compared?",
                "What can be inferred about the training process?",
            ],
            "comparison_plot": [
                "What entities or methods are being compared?",
                "What metric or dimension is used for comparison?",
                "Are the differences statistically significant?",
                "Which approach shows superior performance?",
            ],
            "table_or_matrix": [
                "What data is organized in this table/matrix?",
                "What are the row and column headers?",
                "Are there any notable patterns or trends?",
                "Are all cells properly filled with data?",
            ],
        }

        questions = base_questions.copy()
        if expected_content in content_specific:
            questions.extend(content_specific[expected_content])

        if context:
            questions.append(
                f"Based on this context: '{context[:200]}...', does the image content match the expectation?"
            )

        return questions

    def _generate_research_questions(self, context: str, expected_content: str, analysis_focus: str) -> List[str]:
        """Generate research-focused questions about images for paper analysis."""
        base_questions = [
            "What type of scientific visualization or data is presented?",
            "What are the main experimental results or findings shown?",
            "What research claims or hypotheses does this support?",
            "What numerical values, trends, or patterns are visible?",
        ]

        if analysis_focus in ["content", "image_content"]:
            analysis_focus = "image_content"
        elif analysis_focus in ["trends", "image_trends"]:
            analysis_focus = "image_trends"
        elif analysis_focus in ["technical", "image_technical"]:
            analysis_focus = "image_technical"

        focus_questions: Dict[str, List[str]] = {
            "image_content": [
                "What scientific insights can be extracted from this figure?",
                "How does this relate to the research problem being solved?",
                "What evidence does this provide for the paper's claims?",
                "What are the key takeaways for understanding the research?",
            ],
            "image_trends": [
                "What data patterns or trends are evident?",
                "How do different conditions or methods compare?",
                "Are there any surprising or counterintuitive results?",
                "What does the progression or relationship show?",
            ],
            "image_technical": [
                "What methodology or experimental setup is illustrated?",
                "What technical details about the approach are revealed?",
                "Are there statistical significance indicators?",
                "What parameters or hyperparameters are being varied?",
            ],
            "pdf_reading": [
                "What is the main contribution illustrated by this figure?",
                "How does this figure support the paper's thesis?",
                "What experimental evidence is provided?",
                "What can be learned about the proposed method's performance?",
            ],
        }

        questions = base_questions.copy()
        if analysis_focus in focus_questions:
            questions.extend(focus_questions[analysis_focus])
        else:
            questions.extend(focus_questions["pdf_reading"])

        if context:
            questions.append(
                f"Given this context from the paper: '{context[:300]}...', what specific insights does this figure provide?"
            )

        return questions

    def _extract_research_insights(
        self, full_text: str, image_analyses: List[Dict], analysis_focus: str
    ) -> Dict[str, Any]:
        """Extract high-level research insights from text and image analyses."""
        insights: Dict[str, Any] = {
            "paper_summary": "",
            "key_findings": [],
            "methodology": "",
            "experimental_results": [],
            "limitations": [],
            "contributions": [],
        }

        text_lower = full_text.lower()

        if "abstract" in text_lower:
            abstract_start = text_lower.find("abstract")
            abstract_end = min(
                text_lower.find("introduction", abstract_start),
                text_lower.find("1.", abstract_start)
                if text_lower.find("1.", abstract_start) != -1
                else len(full_text),
            )
            if abstract_end > abstract_start:
                insights["paper_summary"] = full_text[abstract_start:abstract_end][:500]

        contribution_keywords = ["contribution", "propose", "present", "novel", "new method"]
        for keyword in contribution_keywords:
            if keyword in text_lower:
                sentences = full_text.split(". ")
                for sentence in sentences:
                    if keyword in sentence.lower() and len(sentence) < 200:
                        insights["contributions"].append(sentence.strip())

        for img_analysis in image_analyses:
            if img_analysis.get("research_analysis"):
                analysis_text = str(img_analysis["research_analysis"])
                if "performance" in analysis_text.lower() or "result" in analysis_text.lower():
                    insights["experimental_results"].append({
                        "figure_context": img_analysis["context"][:100],
                        "findings": analysis_text[:300],
                    })

        return insights

    def _analyze_image_with_questions(self, image_path: str, questions: List[str]) -> Dict[str, Any]:
        """Use VLM to analyze image with specific questions."""
        if image_path is None:
            return {
                "status": "missing_image",
                "response": (
                    "IMAGE NOT FOUND: This appears to be a missing or invalid image placeholder. "
                    "The PDF likely references an image file that was not available during compilation "
                    "(e.g., missing image file when running pdflatex). This results in a placeholder "
                    "or broken image reference in the final PDF."
                ),
                "questions_asked": len(questions),
                "missing_image": True,
            }

        try:
            questions_text = "\n".join([f"{i+1}. {q}" for i, q in enumerate(questions)])

            prompt = (
                "Analyze this scientific image and answer the following specific questions:\n\n"
                f"{questions_text}\n\n"
                "Please provide detailed, specific answers to each question. "
                "If any issues are detected (missing data, poor quality, broken elements), describe them clearly."
            )

            response = _vlm_via_cli(prompt, [image_path], self.vlm_model)

            return {
                "status": "success",
                "response": response,
                "questions_asked": len(questions),
            }

        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "response": f"Failed to analyze image: {str(e)}",
            }

    def _reconstruct_document_with_analysis(self, original_text: str, image_analyses: List[Dict]) -> str:
        """Reconstruct document by replacing placeholders with VLM analysis."""
        reconstructed_text = original_text

        for analysis in image_analyses:
            placeholder = analysis.get("placeholder", f"[IMAGE_{analysis['image_id']}_PLACEHOLDER]")
            vlm_response = analysis["vlm_analysis"].get("response", "Analysis failed")
            context = analysis.get("context", "")

            replacement_text = (
                f"\n[IMAGE {analysis['image_id']} ANALYSIS]\n"
                f"Context: {context}\n"
                f"VLM Analysis: {vlm_response}\n"
                f"[END IMAGE {analysis['image_id']} ANALYSIS]\n"
            )
            reconstructed_text = reconstructed_text.replace(placeholder, replacement_text)

        return reconstructed_text

    def _identify_publication_issues(self, extracted_data: Dict, image_analyses: List[Dict]) -> List[str]:
        """Identify publication quality issues from the analysis."""
        issues: List[str] = []

        if extracted_data["total_images"] == 0:
            issues.append("No images found in PDF - figures may not be displaying")

        for analysis in image_analyses:
            vlm_analysis = analysis["vlm_analysis"]
            vlm_response = vlm_analysis.get("response", "").lower()

            if vlm_analysis.get("status") == "missing_image" or vlm_analysis.get("missing_image"):
                issues.append(
                    f"Image {analysis['image_id']}: MISSING IMAGE - PDF contains placeholder for image "
                    "that was not available during compilation"
                )
            elif any(phrase in vlm_response for phrase in ["missing", "empty", "blank", "not visible", "broken"]):
                issues.append(f"Image {analysis['image_id']}: Potential display or content issues detected")

            if any(phrase in vlm_response for phrase in ["low quality", "poor resolution", "unclear", "blurry"]):
                issues.append(f"Image {analysis['image_id']}: Quality issues detected")

        text = extracted_data["text"]
        question_marks = text.count("?")

        if question_marks > 3:
            issues.append(f"Multiple question marks detected ({question_marks}) - likely missing citations")

        citation_patterns = [r"\[\?\]", r"\(\?\)", r"\\cite\{\?\}", r"\\ref\{\?\}"]
        for pattern in citation_patterns:
            if re.search(pattern, text):
                issues.append("Missing citation references detected (citation commands with ? marks)")

        return issues

    def _get_analysis_prompt(self, focus: str, num_images: int) -> str:
        """Generate appropriate analysis prompt based on focus and number of images."""
        base_instruction = f"You are analyzing {num_images} scientific figure(s) from an AI/ML research paper. "

        if num_images > 1:
            base_instruction += "Compare and contrast the figures, noting relationships between them. "

        if focus in ["content", "image_content"]:
            focus = "image_content"
        elif focus in ["quality", "image_quality"]:
            focus = "image_quality"
        elif focus in ["trends", "image_trends"]:
            focus = "image_trends"
        elif focus in ["technical", "image_technical"]:
            focus = "image_technical"

        focus_instructions: Dict[str, str] = {
            "image_content": (
                "\nFocus primarily on:\n"
                "1. Content Description: What data, results, or concepts are presented?\n"
                "2. Key Findings: What are the main insights, trends, or conclusions?\n"
                "3. Data Interpretation: What do the values, patterns, and relationships indicate?\n"
            ),
            "image_quality": (
                "\nFocus primarily on:\n"
                "1. Visual Quality: Clarity, readability, and aesthetic appeal\n"
                "2. Technical Standards: Proper axes, labels, legends, and annotations\n"
                "3. Publication Readiness: Meets scientific publication standards?\n"
                "4. Improvement Suggestions: How could the figure be enhanced?\n"
            ),
            "image_trends": (
                "\nFocus primarily on:\n"
                "1. Pattern Identification: What trends, patterns, or relationships are visible?\n"
                "2. Comparative Analysis: How do different conditions/methods compare?\n"
                "3. Statistical Insights: What do the distributions, correlations, or progressions show?\n"
                "4. Experimental Outcomes: What do the results suggest about the research hypothesis?\n"
            ),
            "image_technical": (
                "\nFocus primarily on:\n"
                "1. Technical Elements: Axes scales, units, labels, legends, annotations\n"
                "2. Methodology Indicators: What experimental setup or analysis method is shown?\n"
                "3. Data Presentation: How is the data organized, scaled, and presented?\n"
                "4. Figure Construction: Layout, subplots, color schemes, line styles\n"
            ),
            "pdf_validation": (
                "\nFocus on PDF document visual validation:\n"
                "1. Layout Problems: Are formulas, text, or figures cut off or extending beyond page boundaries?\n"
                "2. Missing Citations: Are there question marks where citations should appear?\n"
                "3. Missing Figures: Are there empty spaces, broken image placeholders, or missing figure content?\n"
                "4. Structural Issues: Are there duplicate sections, malformed layouts, or spacing problems?\n"
                "5. Text Quality: Is text properly formatted with correct line spacing and margins?\n"
                "6. Overall Presentation: Does the document meet publication quality standards?\n"
                "\nFor each issue found, provide specific details about location and severity.\n"
            ),
            "comprehensive": (
                "\nProvide a comprehensive analysis covering:\n"
                "1. Content Type: Figure/document type and what data/results are shown\n"
                "2. Key Findings: Main insights, trends, and scientific conclusions (for figures)\n"
                "3. Technical Assessment: Quality of axes, labels, legends, and overall clarity\n"
                "4. Visual Validation: Layout problems, missing elements, formatting issues (for PDFs)\n"
                "5. Scientific Value: What this contributes to the research narrative\n"
                "6. Publication Quality: Readiness for publication and potential improvements\n"
            ),
        }

        if focus == "pdf_validation":
            return (
                base_instruction
                + focus_instructions["pdf_validation"]
                + "\nCRITICAL PDF VALIDATION INSTRUCTIONS:\n"
                "- Examine EVERY page of the PDF carefully\n"
                "- Look for question mark symbols where citations should appear (indicates missing bibliography)\n"
                "- Check for empty spaces or broken image placeholders where figures should be\n"
                "- Identify any text, formulas, or figures that extend beyond page margins\n"
                "- Note any duplicate sections or repeated content blocks\n"
                "- Assess overall layout quality and formatting consistency\n"
                "- Report ALL issues found with specific page numbers and locations\n"
                "\nProvide a PASS/FAIL assessment for publication readiness.\n"
            )

        return (
            base_instruction
            + focus_instructions.get(focus, focus_instructions["comprehensive"])
            + "\nProvide your analysis in a clear, structured format with specific observations and actionable insights. "
            "Be precise about what you observe and avoid speculation beyond what's directly visible.\n"
        )

    def _structure_analysis(self, raw_response: str, image_paths: List[str], focus: str) -> Dict[str, Any]:
        """Structure the VLM response into a standardized format."""
        result: Dict[str, Any] = {
            "analysis_type": "vlm_document_analysis",
            "focus": focus,
            "file_count": len(image_paths),
            "file_paths": image_paths,
            "detailed_analysis": raw_response,
            "metadata": {
                "model_used": self.vlm_model,
                "analysis_timestamp": None,
                "character_count": len(raw_response),
            },
        }

        if focus == "pdf_validation":
            validation_results = self._extract_pdf_validation_results(raw_response)
            result["pdf_validation"] = validation_results

        try:
            sections = self._extract_analysis_sections(raw_response)
            if sections:
                result["structured_sections"] = sections
        except Exception:
            pass

        return result

    def _extract_analysis_sections(self, response: str) -> Optional[Dict[str, str]]:
        """Attempt to extract structured sections from the VLM response."""
        sections: Dict[str, str] = {}

        section_patterns = [
            (r"(?:^|\n)\*\*Figure Type[:\s]*\*\*([^\n]*(?:\n(?!\*\*)[^\n]*)*)", "figure_type"),
            (r"(?:^|\n)\*\*Content[:\s]*\*\*([^\n]*(?:\n(?!\*\*)[^\n]*)*)", "content"),
            (r"(?:^|\n)\*\*Key Findings[:\s]*\*\*([^\n]*(?:\n(?!\*\*)[^\n]*)*)", "key_findings"),
            (r"(?:^|\n)\*\*Technical[:\s]*\*\*([^\n]*(?:\n(?!\*\*)[^\n]*)*)", "technical"),
            (r"(?:^|\n)\*\*Quality[:\s]*\*\*([^\n]*(?:\n(?!\*\*)[^\n]*)*)", "quality"),
        ]

        for pattern, key in section_patterns:
            match = re.search(pattern, response, re.IGNORECASE | re.MULTILINE)
            if match:
                sections[key] = match.group(1).strip()

        return sections if sections else None

    def _extract_pdf_validation_results(self, response: str) -> Dict[str, Any]:
        """Extract PDF validation results from VLM response."""
        validation: Dict[str, Any] = {
            "overall_assessment": "UNKNOWN",
            "layout_issues": [],
            "missing_citations": [],
            "missing_figures": [],
            "structural_problems": [],
            "publication_ready": False,
            "critical_issues_found": False,
        }

        response_lower = response.lower()

        if "pass" in response_lower and "fail" not in response_lower:
            validation["overall_assessment"] = "PASS"
            validation["publication_ready"] = True
        elif "fail" in response_lower:
            validation["overall_assessment"] = "FAIL"
            validation["publication_ready"] = False

        if "?" in response or "missing citation" in response_lower or "undefined citation" in response_lower:
            citation_matches = re.findall(r"citation[^.]*?\?[^.]*", response, re.IGNORECASE)
            validation["missing_citations"].extend(citation_matches)

        figure_patterns = [
            r"missing figure[^.]*",
            r"empty.*?figure[^.]*",
            r"figure.*?not.*?found[^.]*",
            r"broken.*?image[^.]*",
        ]
        for pattern in figure_patterns:
            matches = re.findall(pattern, response, re.IGNORECASE)
            validation["missing_figures"].extend(matches)

        layout_patterns = [
            r"formula.*?extend[^.]*",
            r"text.*?overflow[^.]*",
            r"margin[^.]*problem[^.]*",
            r"page.*?boundary[^.]*",
        ]
        for pattern in layout_patterns:
            matches = re.findall(pattern, response, re.IGNORECASE)
            validation["layout_issues"].extend(matches)

        structure_patterns = [
            r"duplicate.*?section[^.]*",
            r"repeated.*?content[^.]*",
            r"malformed[^.]*",
            r"structural.*?issue[^.]*",
        ]
        for pattern in structure_patterns:
            matches = re.findall(pattern, response, re.IGNORECASE)
            validation["structural_problems"].extend(matches)

        total_issues = (
            len(validation["layout_issues"])
            + len(validation["missing_citations"])
            + len(validation["missing_figures"])
            + len(validation["structural_problems"])
        )
        validation["critical_issues_found"] = total_issues > 0

        return validation

    def _is_valid_image_data(self, image_bytes: bytes, image_ext: str) -> bool:
        """Validate that extracted image data represents a real image."""
        if not image_bytes or len(image_bytes) < 100:
            return False

        image_signatures: Dict[str, List[bytes]] = {
            "png": [b"\x89PNG\r\n\x1a\n"],
            "jpg": [b"\xff\xd8\xff", b"\xff\xd8"],
            "jpeg": [b"\xff\xd8\xff", b"\xff\xd8"],
            "gif": [b"GIF87a", b"GIF89a"],
            "bmp": [b"BM"],
            "tiff": [b"II*\x00", b"MM\x00*"],
            "webp": [b"RIFF"],
        }

        ext_lower = image_ext.lower()
        if ext_lower in image_signatures:
            signatures = image_signatures[ext_lower]
            for sig in signatures:
                if image_bytes.startswith(sig):
                    return True
            return False

        return len(image_bytes) > 1000

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
                    f"Example: Use 'paper_workspace/final_paper.pdf' instead of the full path."
                )
        else:
            abs_path = os.path.abspath(os.path.join(abs_working_dir, path))

            if not os.path.exists(abs_path):
                parent_dir = os.path.dirname(abs_path)
                if os.path.exists(parent_dir):
                    raise FileNotFoundError(
                        f"File not found: '{path}' does not exist in the workspace. "
                        f"The parent directory exists. Please check the filename."
                    )
                else:
                    raise FileNotFoundError(
                        f"File not found: '{path}' does not exist in the workspace. "
                        f"The directory '{os.path.dirname(path)}' was not found."
                    )

            return abs_path
