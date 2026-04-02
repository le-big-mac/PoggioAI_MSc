"""
Lightweight deep-research novelty scan compatibility tool.

Preserves the old novelty-checker interface for tests and optional workflows,
but routes through ``litellm.completion`` only if that module is present.
"""

from __future__ import annotations

import json
import re

try:
    import litellm  # type: ignore
except Exception:  # pragma: no cover - fallback for environments without litellm
    class _LiteLLMStub:
        def completion(self, *args, **kwargs):
            raise RuntimeError("litellm is not installed")

    litellm = _LiteLLMStub()


_NOVELTY_SYSTEM_PROMPT = """\
You are an adversarial novelty checker for mathematical research claims.
Your job is to find evidence that the given claim is NOT novel.

Respond with a structured analysis:
- VERDICT: OPEN | PARTIAL | KNOWN | EQUIVALENT_KNOWN
- CONFIDENCE: high | medium | low
- EVIDENCE: relevant sources
- SUMMARY: brief explanation
"""


class DeepResearchNoveltyScanTool:
    def __init__(self, model_name: str | None = None, **kwargs):
        self.model_name = model_name

    def _run(self, claim_text: str, context: str = "") -> str:
        user_prompt = f"CLAIM TO CHECK:\n{claim_text}"
        if context:
            user_prompt += f"\n\nCONTEXT:\n{context}"
        try:
            resp = litellm.completion(
                model=self.model_name or "perplexity/sonar-deep-research",
                messages=[
                    {"role": "system", "content": _NOVELTY_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=4096,
            )
            finding = resp.choices[0].message.content or ""
        except Exception:
            return json.dumps({
                "claim_text": claim_text,
                "finding": "[deep research unavailable]",
                "sources": [],
                "verdict": "UNCERTAIN",
                "confidence": "low",
            })

        return json.dumps({
            "claim_text": claim_text,
            "finding": finding,
            "sources": self._extract_sources(finding),
            "verdict": self._extract_verdict(finding),
            "confidence": self._extract_confidence(finding),
        })

    @staticmethod
    def _extract_verdict(text: str) -> str:
        upper = text.upper()
        match = re.search(r"VERDICT\s*:\s*(OPEN|PARTIAL|KNOWN|EQUIVALENT_KNOWN)", upper)
        if match:
            return match.group(1)
        for phrase, verdict in [
            ("HAS BEEN PROVEN", "KNOWN"),
            ("IS A KNOWN RESULT", "KNOWN"),
            ("WAS ESTABLISHED BY", "KNOWN"),
            ("ALREADY PROVEN", "KNOWN"),
            ("IS EQUIVALENT TO", "EQUIVALENT_KNOWN"),
            ("EQUIVALENT RESULT", "EQUIVALENT_KNOWN"),
            ("PARTIAL RESULT", "PARTIAL"),
            ("SPECIAL CASE", "PARTIAL"),
            ("APPEARS NOVEL", "OPEN"),
            ("NO EVIDENCE", "OPEN"),
            ("GENUINELY NOVEL", "OPEN"),
        ]:
            if phrase in upper:
                return verdict
        return "UNCERTAIN"

    @staticmethod
    def _extract_confidence(text: str) -> str:
        upper = text.upper()
        match = re.search(r"CONFIDENCE\s*:\s*(HIGH|MEDIUM|LOW)", upper)
        if match:
            return match.group(1).lower()
        return "medium"

    @staticmethod
    def _extract_sources(text: str) -> list[str]:
        urls = re.findall(r"https?://[^\s\)\"'>]+", text)
        seen = set()
        unique = []
        for url in urls:
            if url not in seen:
                seen.add(url)
                unique.append(url)
        return unique
