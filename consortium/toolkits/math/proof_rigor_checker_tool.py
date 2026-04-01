"""
MathProofRigorCheckerTool - enhanced rigor checks for proof drafts.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple


class MathProofRigorCheckerTool:
    working_dir: Optional[str] = None

    _NAMED_RULE_PATTERNS = [
        r"\bhoeffding\b",
        r"\bbernstein\b",
        r"\bmcdiarmid\b",
        r"\bjensen\b",
        r"\bcauchy[-\s]?schwarz\b|\bcs\b",
        r"\byoung['']s?\b",
        r"\bholder['']s?\b",
        r"\bmarkov\b",
        r"\bchebyshev\b",
        r"\bunion bound\b",
        r"\bdescent lemma\b",
        r"\blyapunov\b",
        r"\bfano\b",
        r"\ble cam\b",
        r"\bmatrix bernstein\b",
    ]
    _SUSPICIOUS_JUMP_PATTERNS = [
        r"\bby standard argument\b",
        r"\bclearly\b",
        r"\bobviously\b",
        r"\bit is easy to see\b",
        r"\btrivially\b",
        r"\bimmediately\b",
    ]
    _BOUND_SYMBOLS = {
        "n",
        "d",
        "m",
        "t",
        "k",
        "l",
        "mu",
        "sigma",
        "delta",
        "epsilon",
        "eta",
        "gamma",
        "lambda",
        "L",
        "B",
        "R",
    }

    def __init__(self, working_dir: Optional[str] = None, **kwargs: Any):
        self.working_dir = os.path.abspath(working_dir) if working_dir else None

    def run(
        self,
        claim_id: Optional[str] = None,
        proof_text: Optional[str] = None,
        check_level: Optional[str] = None,
        workspace_subdir: Optional[str] = "math_workspace",
    ) -> str:
        try:
            check_level = (check_level or "strict").strip().lower()
            if check_level not in {"basic", "strict"}:
                return self._error("check_level must be 'basic' or 'strict'")

            workspace_dir = self._resolve_workspace_dir(workspace_subdir or "math_workspace")
            claim_graph = self._read_claim_graph(workspace_dir)
            claim = self._get_claim(claim_graph, claim_id) if claim_id else None

            text = (proof_text or "").strip()
            if not text and claim_id:
                proof_path = os.path.join(workspace_dir, "proofs", f"{self._safe_id(claim_id)}.md")
                if os.path.exists(proof_path):
                    with open(proof_path, "r", encoding="utf-8") as f:
                        text = f.read().strip()

            if not text:
                return json.dumps(
                    {
                        "success": False,
                        "claim_id": claim_id,
                        "verdict": "fail",
                        "score": 0.0,
                        "critical_issues": ["proof text is empty"],
                    },
                    indent=2,
                )

            checks = self._run_checks(text, claim, claim_graph)
            score = self._score_checks(checks)
            threshold = 0.55 if check_level == "basic" else 0.78
            has_critical = len(checks["critical_issues"]) > 0
            verdict = "pass" if (score >= threshold and not has_critical) else "fail"

            return json.dumps(
                {
                    "success": True,
                    "claim_id": claim_id,
                    "check_level": check_level,
                    "threshold": threshold,
                    "score": round(score, 4),
                    "verdict": verdict,
                    **checks,
                },
                indent=2,
            )

        except Exception as e:
            return self._error(f"proof rigor checker failed: {e}")


    def _run_checks(
        self,
        text: str,
        claim: Optional[Dict[str, Any]],
        claim_graph: Dict[str, Any],
    ) -> Dict[str, Any]:
        low = text.lower()
        sections = {
            "claim": bool(re.search(r"^##\s*claim\b", text, flags=re.IGNORECASE | re.MULTILINE)),
            "assumptions": bool(re.search(r"^##\s*assumptions\b", text, flags=re.IGNORECASE | re.MULTILINE)),
            "dependencies": bool(re.search(r"^##\s*dependencies\b", text, flags=re.IGNORECASE | re.MULTILINE)),
            "definitions": bool(
                re.search(r"^##\s*(definitions?\s*/?\s*notation|notation)\b", text, flags=re.IGNORECASE | re.MULTILINE)
            ),
            "steps": bool(re.search(r"^##\s*(detailed\s+steps|proof\s+steps)\b", text, flags=re.IGNORECASE | re.MULTILINE)),
            "conclusion": bool(re.search(r"^##\s*conclusion\b", text, flags=re.IGNORECASE | re.MULTILINE)),
        }

        step_blocks = self._extract_step_blocks(text)
        step_count = len(step_blocks)
        logical_chain = self._analyze_logical_chain(step_blocks)
        placeholder_hits = re.findall(r"\[(?:fill|todo|tbd)[^\]]*\]", low)

        named_rule_hits = self._find_named_rule_hits(low)
        suspicious_jump_hits = self._find_suspicious_jumps(low)

        dependency_mentions: Dict[str, bool] = {}
        dependency_statuses: Dict[str, str] = {}
        if claim:
            for dep in claim.get("depends_on", []):
                dep = str(dep)
                dependency_mentions[dep] = dep.lower() in low
                dep_claim = self._get_claim(claim_graph, dep)
                dependency_statuses[dep] = str(dep_claim.get("status")) if dep_claim else "missing"

        assumption_mentions: Dict[str, bool] = {}
        if claim:
            for assumption in claim.get("assumptions", []):
                a = str(assumption).strip()
                if not a:
                    continue
                key = a[:120]
                assumption_mentions[key] = self._assumption_is_referenced(a, low)

        claim_symbols = self._extract_claim_symbols(claim)
        symbol_coverage = {sym: (sym.lower() in low) for sym in claim_symbols}

        conclusion_matches_claim = self._conclusion_mentions_claim(text, claim)

        critical_issues: List[str] = []
        warnings: List[str] = []

        for name, present in sections.items():
            if not present:
                if name in {"steps", "conclusion", "claim"}:
                    critical_issues.append(f"missing critical section: {name}")
                else:
                    warnings.append(f"missing section: {name}")

        if step_count < 4:
            critical_issues.append("proof has fewer than four explicit steps")
        elif step_count < 6:
            warnings.append("proof has low step granularity (<6 steps)")

        if placeholder_hits:
            critical_issues.append("proof still contains placeholders/TODOs")

        if suspicious_jump_hits:
            warnings.append(
                "suspicious logical-jump phrases present: "
                + ", ".join(sorted(set(suspicious_jump_hits))[:6])
            )
            if len(suspicious_jump_hits) >= 3:
                critical_issues.append("too many suspicious logical jumps")

        if logical_chain["broken_links"] >= 2:
            critical_issues.append("logical chain appears broken across multiple proof steps")
        elif logical_chain["broken_links"] == 1:
            warnings.append("one potential logical chain break detected")

        if dependency_mentions and not all(dependency_mentions.values()):
            missing = [k for k, v in dependency_mentions.items() if not v]
            warnings.append("dependency references not mentioned: " + ", ".join(missing))

        unresolved_dependency_statuses = {
            dep: status
            for dep, status in dependency_statuses.items()
            if status in {"missing", "proposed", "rejected"}
        }
        if unresolved_dependency_statuses:
            warnings.append(
                "some dependencies are unresolved in claim graph: "
                + ", ".join(f"{k}:{v}" for k, v in unresolved_dependency_statuses.items())
            )

        if assumption_mentions and not all(assumption_mentions.values()):
            missing = [k for k, v in assumption_mentions.items() if not v]
            warnings.append("assumptions not explicitly referenced: " + ", ".join(missing[:8]))

        if symbol_coverage and not all(symbol_coverage.values()):
            missing_symbols = [k for k, v in symbol_coverage.items() if not v]
            warnings.append("bound symbols/constants not mentioned: " + ", ".join(missing_symbols))

        if step_count >= 8 and not named_rule_hits:
            warnings.append("long proof has no explicit named inequality/theorem references")

        if not conclusion_matches_claim:
            warnings.append("conclusion does not appear to restate target claim")

        return {
            "section_presence": sections,
            "step_count": step_count,
            "logical_chain": logical_chain,
            "placeholder_hits": placeholder_hits,
            "named_rule_hits": named_rule_hits,
            "suspicious_jump_hits": suspicious_jump_hits,
            "dependency_mentions": dependency_mentions,
            "dependency_statuses": dependency_statuses,
            "assumption_mentions": assumption_mentions,
            "claim_symbol_coverage": symbol_coverage,
            "conclusion_matches_claim": conclusion_matches_claim,
            "critical_issues": critical_issues,
            "warnings": warnings,
        }

    def _extract_step_blocks(self, text: str) -> List[str]:
        lines = text.splitlines()
        step_lines: List[str] = []
        for line in lines:
            if re.match(r"^\s*(?:step\s*\d+\.|\d+\.|-\s*step\s*\d+)", line, flags=re.IGNORECASE):
                step_lines.append(line.strip())
        return step_lines

    def _analyze_logical_chain(self, step_lines: List[str]) -> Dict[str, Any]:
        if not step_lines:
            return {"broken_links": 0, "link_indicators_found": 0, "step_count": 0}

        link_keywords = [
            "therefore",
            "thus",
            "hence",
            "from",
            "using",
            "by",
            "combine",
            "substitute",
            "plug",
            "apply",
        ]
        link_indicators = 0
        broken_links = 0
        for idx, step in enumerate(step_lines):
            low = step.lower()
            if any(k in low for k in link_keywords):
                link_indicators += 1
            elif idx > 0:
                broken_links += 1
        return {
            "broken_links": broken_links,
            "link_indicators_found": link_indicators,
            "step_count": len(step_lines),
        }

    def _find_named_rule_hits(self, low_text: str) -> List[str]:
        hits: List[str] = []
        for pattern in self._NAMED_RULE_PATTERNS:
            if re.search(pattern, low_text):
                hits.append(pattern)
        return hits

    def _find_suspicious_jumps(self, low_text: str) -> List[str]:
        hits: List[str] = []
        for pattern in self._SUSPICIOUS_JUMP_PATTERNS:
            for match in re.findall(pattern, low_text):
                if isinstance(match, tuple):
                    hits.extend([m for m in match if m])
                elif match:
                    hits.append(match)
        return hits

    def _extract_claim_symbols(self, claim: Optional[Dict[str, Any]]) -> List[str]:
        if not claim:
            return []
        statement = str(claim.get("statement", ""))
        symbols = set()
        for token in re.findall(r"\b[A-Za-z][A-Za-z0-9_]*\b", statement):
            if token in self._BOUND_SYMBOLS or token.lower() in {s.lower() for s in self._BOUND_SYMBOLS}:
                symbols.add(token)
        for assumption in claim.get("assumptions", []):
            for token in re.findall(r"\b[A-Za-z][A-Za-z0-9_]*\b", str(assumption)):
                if token in self._BOUND_SYMBOLS or token.lower() in {s.lower() for s in self._BOUND_SYMBOLS}:
                    symbols.add(token)
        return sorted(symbols)

    def _assumption_is_referenced(self, assumption: str, low_text: str) -> bool:
        assumption_low = assumption.lower()
        if assumption_low in low_text:
            return True
        # fallback: if two+ salient tokens from assumption appear, treat as referenced
        tokens = [
            t
            for t in re.findall(r"\b[a-zA-Z][a-zA-Z0-9_]{2,}\b", assumption_low)
            if t not in {"assume", "assumption", "for", "all", "there", "exists", "such", "that"}
        ]
        if not tokens:
            return False
        hit_count = sum(1 for t in set(tokens[:8]) if t in low_text)
        return hit_count >= min(2, len(set(tokens[:8])))

    def _conclusion_mentions_claim(self, text: str, claim: Optional[Dict[str, Any]]) -> bool:
        if not claim:
            return True
        conclusion_match = re.search(
            r"^##\s*conclusion\b(.*)$",
            text,
            flags=re.IGNORECASE | re.MULTILINE | re.DOTALL,
        )
        if not conclusion_match:
            return False
        conclusion = conclusion_match.group(1).lower()
        statement = str(claim.get("statement", "")).lower()
        key_terms = [
            t
            for t in re.findall(r"\b[a-zA-Z][a-zA-Z0-9_]{3,}\b", statement)
            if t not in {"under", "with", "from", "that", "then", "therefore", "exists", "holds"}
        ]
        if not key_terms:
            return bool(conclusion.strip())
        hits = sum(1 for t in set(key_terms[:10]) if t in conclusion)
        return hits >= min(2, len(set(key_terms[:10])))

    @staticmethod
    def _score_checks(checks: Dict[str, Any]) -> float:
        score = 0.0

        sections = checks.get("section_presence", {})
        section_ratio = sum(1 for v in sections.values() if v) / max(1, len(sections))
        score += 0.25 * section_ratio

        steps = int(checks.get("step_count", 0))
        score += 0.15 * min(1.0, steps / 10.0)

        dep = checks.get("dependency_mentions", {})
        dep_ratio = (sum(1 for v in dep.values() if v) / len(dep)) if dep else 1.0
        score += 0.1 * dep_ratio

        asm = checks.get("assumption_mentions", {})
        asm_ratio = (sum(1 for v in asm.values() if v) / len(asm)) if asm else 1.0
        score += 0.1 * asm_ratio

        chain = checks.get("logical_chain", {})
        chain_steps = max(1, int(chain.get("step_count", 0)))
        broken_links = int(chain.get("broken_links", 0))
        chain_ratio = max(0.0, 1.0 - (broken_links / chain_steps))
        score += 0.15 * chain_ratio

        rule_hits = checks.get("named_rule_hits", [])
        score += 0.1 * min(1.0, len(rule_hits) / 3.0)

        sym = checks.get("claim_symbol_coverage", {})
        sym_ratio = (sum(1 for v in sym.values() if v) / len(sym)) if sym else 1.0
        score += 0.1 * sym_ratio

        conclusion_ok = bool(checks.get("conclusion_matches_claim", False))
        score += 0.05 * (1.0 if conclusion_ok else 0.0)

        if checks.get("placeholder_hits"):
            score -= 0.25
        jump_hits = checks.get("suspicious_jump_hits", [])
        if jump_hits:
            score -= min(0.15, 0.03 * len(jump_hits))
        if checks.get("critical_issues"):
            score -= 0.25

        return max(0.0, min(1.0, score))

    def _resolve_workspace_dir(self, workspace_subdir: str) -> str:
        if self.working_dir:
            abs_working = os.path.abspath(self.working_dir)
            if os.path.isabs(workspace_subdir):
                target = os.path.abspath(workspace_subdir)
            else:
                target = os.path.abspath(os.path.join(abs_working, workspace_subdir))
            try:
                within_root = os.path.commonpath([abs_working, target]) == abs_working
            except ValueError:
                within_root = False
            if not within_root:
                raise PermissionError(
                    f"Access denied: workspace_subdir '{workspace_subdir}' resolves outside '{abs_working}'"
                )
            return target
        return os.path.abspath(workspace_subdir)

    @staticmethod
    def _safe_id(claim_id: str) -> str:
        return re.sub(r"[^a-zA-Z0-9_.-]", "_", str(claim_id))

    @staticmethod
    def _read_claim_graph(workspace_dir: str) -> Dict[str, Any]:
        path = os.path.join(workspace_dir, "claim_graph.json")
        if not os.path.exists(path):
            return {"claims": []}
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def _get_claim(graph: Dict[str, Any], claim_id: Optional[str]) -> Optional[Dict[str, Any]]:
        if not claim_id:
            return None
        for claim in graph.get("claims", []):
            if str(claim.get("id")) == str(claim_id):
                return claim
        return None

    @staticmethod
    def _error(message: str) -> str:
        return json.dumps({"success": False, "error": message}, indent=2)
