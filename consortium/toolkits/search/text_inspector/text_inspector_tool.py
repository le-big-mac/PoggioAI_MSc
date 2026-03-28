from __future__ import annotations
from typing import Optional, Type, Any
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field, ConfigDict


class TextInspectorToolInput(BaseModel):
    file_path: str = Field(description="The path to the file you want to read as text. Must be a '.something' file, like '.pdf'. If it is an image, use the visualizer tool instead! DO NOT use this tool for an HTML webpage: use the web_search tool instead!")
    question: Optional[str] = Field(default=None, description="[Optional]: Your question, as a natural language sentence. Provide as much context as possible. Do not pass this parameter if you just want to directly return the content of the file.")


class TextInspectorTool(BaseTool):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    name: str = "inspect_file_as_text"
    description: str = """
Converts documents to text and returns the content directly.
Handles: PDF, Word (.docx), Excel (.xlsx), PowerPoint (.pptx), HTML, audio files (transcription), and plain text.
NOT for: Images, or simple workspace file reading (use see_file for that).
Returns the converted text content (no LLM analysis)."""
    args_schema: Type[BaseModel] = TextInspectorToolInput
    model_id: str = ""
    text_limit: int = 100000
    working_dir: Optional[str] = None

    md_converter: Any = None

    def __init__(self, model=None, text_limit: int = 100000, working_dir: str = None, **kwargs: Any):
        model_id = model if isinstance(model, str) else getattr(model, 'model', str(model)) if model else ""
        from ..text_web_browser.mdconvert import MarkdownConverter
        converter = MarkdownConverter()
        super().__init__(
            model_id=model_id,
            text_limit=text_limit,
            working_dir=working_dir,
            md_converter=converter,
            **kwargs
        )

    def _safe_path(self, path: str) -> str:
        """Convert path to absolute path, resolving relative paths with working_dir if provided."""
        if not self.working_dir:
            # No working directory - use path as-is (supports absolute paths and paths relative to current dir)
            return path

        import os
        # If path is already absolute, use it directly
        if os.path.isabs(path):
            return path
        else:
            # Relative path - join with working_dir to create absolute path
            return os.path.abspath(os.path.join(self.working_dir, path))

    def _run(self, file_path, question: str | None = None) -> str:
        safe_file_path = self._safe_path(file_path)
        result = self.md_converter.convert(safe_file_path)

        if file_path[-4:] in [".png", ".jpg"]:
            raise Exception("Cannot use inspect_file_as_text tool with images: use visualizer instead!")

        if ".zip" in file_path:
            return result.text_content

        # Return the converted text content directly (truncated to text_limit)
        return result.text_content[:self.text_limit]
