"""
MathClaimGraphTool - manage theorem/lemma claim graphs for theory workflows.
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


class MathClaimGraphTool:
    working_dir: Optional[str] = None
    allow_accepted_transition: bool = False

    _VALID_STATUSES = {
        "proposed",
        "proved_draft",
        "verified_symbolic",
        "verified_numeric",
        "accepted",
        "rejected",
    }
    _CLAIM_ID_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
    _ALLOWED_TRANSITIONS = {
        "proposed": {"proved_draft", "rejected"},
        "proved_draft": {"proposed", "verified_symbolic", "rejected"},
        "verified_symbolic": {"proved_draft", "verified_numeric", "rejected"},
        "verified_numeric": {"verified_symbolic", "accepted", "rejected"},
        "accepted": {"verified_numeric", "rejected"},
        "rejected": {"proposed"},
    }

    def __init__(
        self,
        working_dir: Optional[str] = None,
        allow_accepted_transition: bool = False,
        **kwargs: Any,
    ):
        self.working_dir = os.path.abspath(working_dir) if working_dir else None
        self.allow_accepted_transition = bool(allow_accepted_transition)

    def run(
        self,
        action: str,
        claim_id: Optional[str] = None,
        statement: Optional[str] = None,
        assumptions_json: Optional[str] = None,
        depends_on_json: Optional[str] = None,
        tags_json: Optional[str] = None,
        status: Optional[str] = None,
        notes: Optional[str] = None,
        must_accept: Optional[bool] = None,
        workspace_subdir: Optional[str] = "math_workspace",
        lemma_id: Optional[str] = None,
        lemma_tier: Optional[str] = None,
        lemma_statement: Optional[str] = None,
        lemma_conditions: Optional[str] = None,
        lemma_source: Optional[str] = None,
        lemma_usage_notes: Optional[str] = None,
        lemma_tags_json: Optional[str] = None,
        lemma_status: Optional[str] = None,
        lemma_limit: Optional[int] = None,
    ) -> str:
        try:
            action = (action or "").strip()
            if not action:
                return self._error("'action' is required")

            workspace_dir = self._resolve_workspace_dir(workspace_subdir or "math_workspace")
            os.makedirs(workspace_dir, exist_ok=True)
            graph_path = os.path.join(workspace_dir, "claim_graph.json")

            if action == "init":
                graph = self._load_graph(graph_path)
                self._save_graph(graph_path, graph)
                lemma_library_path = self._ensure_lemma_library(workspace_dir)
                lemma_index_path = self._ensure_lemma_library_index(workspace_dir)
                return json.dumps(
                    {
                        "success": True,
                        "action": action,
                        "graph_path": graph_path,
                        "lemma_library_path": lemma_library_path,
                        "lemma_index_path": lemma_index_path,
                        "claim_count": len(graph["claims"]),
                    },
                    indent=2,
                )

            if action in {"list_lemmas", "get_lemma", "upsert_lemma", "touch_lemma_usage"}:
                lemma_index_path = self._ensure_lemma_library_index(workspace_dir)
                index = self._load_lemma_library_index(lemma_index_path)

                if action == "list_lemmas":
                    entries = index.get("entries", [])
                    if lemma_limit is not None:
                        try:
                            lim = max(0, int(lemma_limit))
                            entries = entries[:lim]
                        except Exception:
                            pass
                    return json.dumps(
                        {
                            "success": True,
                            "action": action,
                            "lemma_index_path": lemma_index_path,
                            "lemma_count": len(index.get("entries", [])),
                            "entries": entries,
                        },
                        indent=2,
                    )

                if action == "get_lemma":
                    if not lemma_id:
                        return self._error("'lemma_id' is required for get_lemma")
                    lemma = self._find_lemma(index, lemma_id)
                    if lemma is None:
                        return self._error(f"lemma '{lemma_id}' not found")
                    return json.dumps(
                        {
                            "success": True,
                            "action": action,
                            "lemma": lemma,
                            "lemma_index_path": lemma_index_path,
                        },
                        indent=2,
                    )

                if action == "upsert_lemma":
                    if not lemma_id:
                        return self._error("'lemma_id' is required for upsert_lemma")

                    lemma = self._find_lemma(index, lemma_id)
                    created = lemma is None
                    if created:
                        if not lemma_statement:
                            return self._error(
                                "'lemma_statement' is required when creating a new lemma entry"
                            )
                        lemma = {
                            "id": lemma_id,
                            "tier": "tier1",
                            "statement": "",
                            "conditions": "",
                            "source": "",
                            "usage_notes": "",
                            "tags": [],
                            "status": "active",
                            "usage_count": 0,
                            "created_at": self._now_iso(),
                            "updated_at": self._now_iso(),
                            "last_used_at": "",
                        }
                        index.setdefault("entries", []).append(lemma)

                    if lemma_tier is not None:
                        lemma["tier"] = self._normalize_lemma_tier(lemma_tier)
                    if lemma_statement is not None:
                        lemma["statement"] = lemma_statement
                    if lemma_conditions is not None:
                        lemma["conditions"] = lemma_conditions
                    if lemma_source is not None:
                        lemma["source"] = lemma_source
                    if lemma_usage_notes is not None:
                        lemma["usage_notes"] = lemma_usage_notes
                    if lemma_tags_json is not None:
                        lemma["tags"] = self._parse_json_list(lemma_tags_json)
                    if lemma_status is not None:
                        lemma["status"] = self._normalize_lemma_status(lemma_status)

                    lemma["updated_at"] = self._now_iso()
                    self._save_lemma_library_index(lemma_index_path, index)
                    self._sync_lemma_library_markdown(workspace_dir, index)

                    return json.dumps(
                        {
                            "success": True,
                            "action": action,
                            "created": created,
                            "lemma": lemma,
                            "lemma_index_path": lemma_index_path,
                            "lemma_library_path": os.path.join(workspace_dir, "lemma_library.md"),
                        },
                        indent=2,
                    )

                if action == "touch_lemma_usage":
                    if not lemma_id:
                        return self._error("'lemma_id' is required for touch_lemma_usage")
                    lemma = self._find_lemma(index, lemma_id)
                    if lemma is None:
                        return self._error(f"lemma '{lemma_id}' not found")
                    lemma["usage_count"] = int(lemma.get("usage_count", 0) or 0) + 1
                    lemma["last_used_at"] = self._now_iso()
                    lemma["updated_at"] = self._now_iso()
                    self._save_lemma_library_index(lemma_index_path, index)
                    self._sync_lemma_library_markdown(workspace_dir, index)
                    return json.dumps(
                        {
                            "success": True,
                            "action": action,
                            "lemma": lemma,
                            "lemma_index_path": lemma_index_path,
                        },
                        indent=2,
                    )

            graph = self._load_graph(graph_path)

            if action == "add_claim":
                if not claim_id or not statement:
                    return self._error("'claim_id' and 'statement' are required for add_claim")
                claim_id_err = self._validate_claim_id(claim_id)
                if claim_id_err:
                    return self._error(claim_id_err)
                if self._find_claim(graph, claim_id) is not None:
                    return self._error(f"claim '{claim_id}' already exists")

                parsed_deps = self._parse_json_list(depends_on_json)
                dep_err = self._validate_claim_ids(parsed_deps, field_name="depends_on")
                if dep_err:
                    return self._error(dep_err)

                requested_status = status if status in self._VALID_STATUSES else "proposed"
                if requested_status == "accepted":
                    return self._error(
                        "add_claim cannot initialize directly to accepted; use set_status through the review workflow"
                    )
                claim = {
                    "id": claim_id,
                    "statement": statement,
                    "assumptions": self._parse_json_list(assumptions_json),
                    "depends_on": parsed_deps,
                    "tags": self._parse_json_list(tags_json),
                    "status": requested_status,
                    "notes": notes or "",
                    "must_accept": bool(must_accept),
                    "created_at": self._now_iso(),
                    "updated_at": self._now_iso(),
                }
                graph["claims"].append(claim)
                self._save_graph(graph_path, graph)
                return json.dumps({"success": True, "action": action, "claim": claim}, indent=2)

            if action == "update_claim":
                if not claim_id:
                    return self._error("'claim_id' is required for update_claim")
                claim_id_err = self._validate_claim_id(claim_id)
                if claim_id_err:
                    return self._error(claim_id_err)
                claim = self._find_claim(graph, claim_id)
                if claim is None:
                    return self._error(f"claim '{claim_id}' not found")

                material_change = False
                if statement is not None:
                    claim["statement"] = statement
                    material_change = True
                if assumptions_json is not None:
                    claim["assumptions"] = self._parse_json_list(assumptions_json)
                    material_change = True
                if depends_on_json is not None:
                    parsed_deps = self._parse_json_list(depends_on_json)
                    dep_err = self._validate_claim_ids(parsed_deps, field_name="depends_on")
                    if dep_err:
                        return self._error(dep_err)
                    claim["depends_on"] = parsed_deps
                    material_change = True
                if tags_json is not None:
                    claim["tags"] = self._parse_json_list(tags_json)
                if notes is not None:
                    claim["notes"] = notes
                if must_accept is not None:
                    claim["must_accept"] = bool(must_accept)
                if status is not None:
                    if status not in self._VALID_STATUSES:
                        return self._error(
                            f"invalid status '{status}'. valid: {sorted(self._VALID_STATUSES)}"
                        )
                    if status == "accepted" and not self.allow_accepted_transition:
                        return self._error(
                            "setting status=accepted is manager-only; this tool instance is not allowed to accept claims"
                        )
                    transition_err = self._validate_transition(
                        old_status=str(claim.get("status", "proposed")),
                        new_status=str(status),
                        context_action="update_claim",
                    )
                    if transition_err:
                        return self._error(transition_err)
                    claim["status"] = status

                if material_change and status is None:
                    current_status = str(claim.get("status", "proposed"))
                    if current_status in {"verified_symbolic", "verified_numeric", "accepted"}:
                        claim["status"] = "proved_draft"
                        auto_note = (
                            "Status auto-reset to proved_draft because statement/assumptions/dependencies changed."
                        )
                        claim["notes"] = (
                            f"{claim.get('notes', '').rstrip()}\n{auto_note}".strip()
                        )
                claim["updated_at"] = self._now_iso()
                self._save_graph(graph_path, graph)
                return json.dumps({"success": True, "action": action, "claim": claim}, indent=2)

            if action == "set_status":
                if not claim_id or not status:
                    return self._error("'claim_id' and 'status' are required for set_status")
                claim_id_err = self._validate_claim_id(claim_id)
                if claim_id_err:
                    return self._error(claim_id_err)
                if status not in self._VALID_STATUSES:
                    return self._error(f"invalid status '{status}'. valid: {sorted(self._VALID_STATUSES)}")
                if status == "accepted" and not self.allow_accepted_transition:
                    return self._error(
                        "setting status=accepted is manager-only; this tool instance is not allowed to accept claims"
                    )

                claim = self._find_claim(graph, claim_id)
                if claim is None:
                    return self._error(f"claim '{claim_id}' not found")
                transition_err = self._validate_transition(
                    old_status=str(claim.get("status", "proposed")),
                    new_status=str(status),
                    context_action="set_status",
                )
                if transition_err:
                    return self._error(transition_err)
                claim["status"] = status
                claim["updated_at"] = self._now_iso()
                self._save_graph(graph_path, graph)
                return json.dumps({"success": True, "action": action, "claim": claim}, indent=2)

            if action == "add_dependency":
                if not claim_id or not depends_on_json:
                    return self._error("'claim_id' and 'depends_on_json' are required for add_dependency")
                claim_id_err = self._validate_claim_id(claim_id)
                if claim_id_err:
                    return self._error(claim_id_err)
                claim = self._find_claim(graph, claim_id)
                if claim is None:
                    return self._error(f"claim '{claim_id}' not found")

                deps = self._parse_json_list(depends_on_json)
                dep_err = self._validate_claim_ids(deps, field_name="depends_on")
                if dep_err:
                    return self._error(dep_err)
                for dep in deps:
                    if dep not in claim["depends_on"]:
                        claim["depends_on"].append(dep)
                claim["updated_at"] = self._now_iso()
                self._save_graph(graph_path, graph)
                return json.dumps({"success": True, "action": action, "claim": claim}, indent=2)

            if action == "get_claim":
                if not claim_id:
                    return self._error("'claim_id' is required for get_claim")
                claim_id_err = self._validate_claim_id(claim_id)
                if claim_id_err:
                    return self._error(claim_id_err)
                claim = self._find_claim(graph, claim_id)
                if claim is None:
                    return self._error(f"claim '{claim_id}' not found")
                return json.dumps({"success": True, "action": action, "claim": claim}, indent=2)

            if action == "list_claims":
                counts: Dict[str, int] = {}
                for claim in graph["claims"]:
                    s = claim.get("status", "proposed")
                    counts[s] = counts.get(s, 0) + 1
                return json.dumps(
                    {
                        "success": True,
                        "action": action,
                        "graph_path": graph_path,
                        "claim_count": len(graph["claims"]),
                        "status_counts": counts,
                        "claims": graph["claims"],
                    },
                    indent=2,
                )

            if action == "validate_graph":
                validation = self._validate_graph(graph)
                return json.dumps({"success": True, "action": action, **validation}, indent=2)

            return self._error(f"unknown action '{action}'")

        except Exception as e:
            return self._error(f"math claim graph tool failed: {e}")


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
    def _default_graph() -> Dict[str, Any]:
        now = MathClaimGraphTool._now_iso()
        return {
            "version": 1,
            "created_at": now,
            "updated_at": now,
            "claims": [],
        }

    def _load_graph(self, graph_path: str) -> Dict[str, Any]:
        if not os.path.exists(graph_path):
            return self._default_graph()

        with open(graph_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if "claims" not in data or not isinstance(data["claims"], list):
            raise ValueError("invalid claim graph format: missing 'claims' list")
        return data

    def _save_graph(self, graph_path: str, graph: Dict[str, Any]) -> None:
        graph["updated_at"] = self._now_iso()
        with open(graph_path, "w", encoding="utf-8") as f:
            json.dump(graph, f, indent=2)

    def _ensure_lemma_library(self, workspace_dir: str) -> str:
        path = os.path.join(workspace_dir, "lemma_library.md")
        if os.path.exists(path) and os.path.getsize(path) > 0:
            return path

        index = self._load_lemma_library_index(self._ensure_lemma_library_index(workspace_dir))
        self._sync_lemma_library_markdown(workspace_dir, index)
        return path

    def _lemma_library_index_path(self, workspace_dir: str) -> str:
        return os.path.join(workspace_dir, "lemma_library_index.json")

    def _ensure_lemma_library_index(self, workspace_dir: str) -> str:
        path = self._lemma_library_index_path(workspace_dir)
        if os.path.exists(path):
            return path
        md_path = os.path.join(workspace_dir, "lemma_library.md")
        migrated_entries = self._parse_lemma_library_markdown(md_path)
        index = {
            "version": 1,
            "created_at": self._now_iso(),
            "updated_at": self._now_iso(),
            "entries": migrated_entries if migrated_entries else self._default_lemma_entries(),
        }
        self._save_lemma_library_index(path, index)
        return path

    def _load_lemma_library_index(self, path: str) -> Dict[str, Any]:
        if not os.path.exists(path):
            return {
                "version": 1,
                "created_at": self._now_iso(),
                "updated_at": self._now_iso(),
                "entries": self._default_lemma_entries(),
            }
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if "entries" not in data or not isinstance(data["entries"], list):
            raise ValueError("invalid lemma_library_index format: missing 'entries' list")
        return data

    def _save_lemma_library_index(self, path: str, index: Dict[str, Any]) -> None:
        index["updated_at"] = self._now_iso()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(index, f, indent=2)

    def _sync_lemma_library_markdown(self, workspace_dir: str, index: Dict[str, Any]) -> None:
        path = os.path.join(workspace_dir, "lemma_library.md")
        lines: List[str] = []
        lines.append("# Standard Lemma Library (Math Fast Path)")
        lines.append("")
        lines.append("This file is synchronized from `lemma_library_index.json`.")
        lines.append("Use `math_claim_graph_tool` lemma actions for incremental updates:")
        lines.append("- `list_lemmas`")
        lines.append("- `get_lemma`")
        lines.append("- `upsert_lemma`")
        lines.append("- `touch_lemma_usage`")
        lines.append("")
        lines.append("Policy:")
        lines.append("- Tier 0 (primitive): inline only, usually no claim node.")
        lines.append("- Tier 1 (standard ML-theory): prefer library-backed claim nodes (`origin:library`).")
        lines.append("- Tier 2 (specialized known): explicit conditions and source required.")
        lines.append("- Tier 3 (novel): full proof workflow required.")
        lines.append("")

        entries = index.get("entries", [])
        if not entries:
            lines.append("_No entries yet._")
        else:
            for entry in entries:
                lines.append(f"## {entry.get('id', '')}")
                lines.append(f"- tier: {entry.get('tier', 'tier1')}")
                lines.append(f"- status: {entry.get('status', 'active')}")
                lines.append(f"- statement: {entry.get('statement', '')}")
                lines.append(f"- conditions: {entry.get('conditions', '')}")
                lines.append(f"- source: {entry.get('source', '')}")
                lines.append(f"- usage_notes: {entry.get('usage_notes', '')}")
                tags = entry.get("tags", [])
                lines.append(f"- tags: {', '.join(str(t) for t in tags) if tags else ''}")
                lines.append(f"- usage_count: {entry.get('usage_count', 0)}")
                lines.append(f"- last_used_at: {entry.get('last_used_at', '')}")
                lines.append("")

        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines).rstrip() + "\n")

    def _default_lemma_entries(self) -> List[Dict[str, Any]]:
        now = self._now_iso()
        return [
            {
                "id": "L_smooth_descent_standard",
                "tier": "tier1",
                "statement": "Descent inequality for L-smooth objectives.",
                "conditions": "Objective is differentiable and L-smooth.",
                "source": "Canonical optimization texts (set exact source per project).",
                "usage_notes": "Use for one-step progress bounds.",
                "tags": ["area:optimization", "origin:library"],
                "status": "active",
                "usage_count": 0,
                "created_at": now,
                "updated_at": now,
                "last_used_at": "",
            },
            {
                "id": "L_operator_nuclear_duality",
                "tier": "tier1",
                "statement": "<A,B> <= ||A||_2 ||B||_* for compatible real matrices.",
                "conditions": "Finite-dimensional real matrices with compatible shapes.",
                "source": "Standard matrix analysis references.",
                "usage_notes": "Use in operator-vs-nuclear norm arguments.",
                "tags": ["area:matrix", "origin:library"],
                "status": "active",
                "usage_count": 0,
                "created_at": now,
                "updated_at": now,
                "last_used_at": "",
            },
        ]

    def _parse_lemma_library_markdown(self, md_path: str) -> List[Dict[str, Any]]:
        """
        Best-effort migration parser for existing manual lemma_library.md files.
        Expected loose schema:
          ## L_id
          - tier: ...
          - statement: ...
          - conditions: ...
          - source: ...
          - usage_notes: ...
          - tags: a, b
          - status: active
        """
        if not os.path.exists(md_path):
            return []
        try:
            with open(md_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
        except Exception:
            return []

        entries: List[Dict[str, Any]] = []
        current: Optional[Dict[str, Any]] = None

        def _finish_current():
            nonlocal current
            if not current:
                return
            if not current.get("id"):
                current = None
                return
            now = self._now_iso()
            current.setdefault("tier", "tier1")
            current.setdefault("statement", "")
            current.setdefault("conditions", "")
            current.setdefault("source", "")
            current.setdefault("usage_notes", "")
            current.setdefault("tags", [])
            current.setdefault("status", "active")
            current.setdefault("usage_count", 0)
            current.setdefault("created_at", now)
            current.setdefault("updated_at", now)
            current.setdefault("last_used_at", "")
            entries.append(current)
            current = None

        for raw in lines:
            line = raw.strip()
            if line.startswith("## "):
                _finish_current()
                cid = line[3:].strip()
                if cid:
                    current = {"id": cid}
                continue
            if current is None:
                continue
            if not line.startswith("-"):
                continue
            body = line[1:].strip()
            if ":" not in body:
                continue
            key, value = body.split(":", 1)
            key = key.strip().lower()
            value = value.strip()
            if key == "tier":
                current["tier"] = self._normalize_lemma_tier(value)
            elif key == "statement":
                current["statement"] = value
            elif key == "conditions":
                current["conditions"] = value
            elif key == "source":
                current["source"] = value
            elif key in {"usage_notes", "usage"}:
                current["usage_notes"] = value
            elif key == "tags":
                current["tags"] = [x.strip() for x in value.split(",") if x.strip()]
            elif key == "status":
                current["status"] = self._normalize_lemma_status(value)
            elif key == "usage_count":
                try:
                    current["usage_count"] = int(value)
                except Exception:
                    current["usage_count"] = 0
            elif key == "last_used_at":
                current["last_used_at"] = value
        _finish_current()

        # Keep deterministic order and unique IDs (first occurrence wins).
        seen = set()
        uniq: List[Dict[str, Any]] = []
        for e in entries:
            cid = str(e.get("id", "")).strip()
            if not cid or cid in seen:
                continue
            seen.add(cid)
            uniq.append(e)
        return uniq

    @staticmethod
    def _find_lemma(index: Dict[str, Any], lemma_id: str) -> Optional[Dict[str, Any]]:
        for entry in index.get("entries", []):
            if str(entry.get("id", "")) == str(lemma_id):
                return entry
        return None

    @staticmethod
    def _normalize_lemma_tier(value: str) -> str:
        v = str(value or "").strip().lower().replace(" ", "")
        if v in {"0", "tier0"}:
            return "tier0"
        if v in {"1", "tier1"}:
            return "tier1"
        if v in {"2", "tier2"}:
            return "tier2"
        if v in {"3", "tier3"}:
            return "tier3"
        return "tier1"

    @staticmethod
    def _normalize_lemma_status(value: str) -> str:
        v = str(value or "").strip().lower()
        if v in {"active", "deprecated", "draft"}:
            return v
        return "active"

    @staticmethod
    def _find_claim(graph: Dict[str, Any], claim_id: str) -> Optional[Dict[str, Any]]:
        for claim in graph.get("claims", []):
            if claim.get("id") == claim_id:
                return claim
        return None

    @staticmethod
    def _parse_json_list(raw: Optional[str]) -> List[str]:
        if raw is None:
            return []
        text = raw.strip()
        if not text:
            return []
        value = json.loads(text)
        if not isinstance(value, list):
            raise ValueError("expected a JSON array")
        return [str(x) for x in value]

    def _validate_graph(self, graph: Dict[str, Any]) -> Dict[str, Any]:
        claims = graph.get("claims", [])
        ids = {str(c.get("id")) for c in claims if c.get("id")}
        missing_dependencies: Dict[str, List[str]] = {}
        self_dependencies: Dict[str, List[str]] = {}
        invalid_statuses: Dict[str, str] = {}
        accepted_dependency_violations: Dict[str, List[str]] = {}
        invalid_claim_ids: List[str] = []
        duplicate_claim_ids: List[str] = []
        seen_ids = set()

        for claim in claims:
            cid = str(claim.get("id", ""))
            if not cid:
                continue
            if cid in seen_ids:
                duplicate_claim_ids.append(cid)
            else:
                seen_ids.add(cid)
            if not self._CLAIM_ID_RE.match(cid):
                invalid_claim_ids.append(cid)
            status = str(claim.get("status", "proposed"))
            if status not in self._VALID_STATUSES:
                invalid_statuses[cid] = status
            deps = [str(d) for d in claim.get("depends_on", [])]
            self_deps = [d for d in deps if d == cid]
            if self_deps:
                self_dependencies[cid] = self_deps
            missing = [d for d in deps if d not in ids]
            if missing:
                missing_dependencies[cid] = missing
            if status == "accepted":
                bad = []
                for dep in deps:
                    dep_claim = self._find_claim(graph, dep)
                    if dep_claim is None or str(dep_claim.get("status", "")) != "accepted":
                        bad.append(dep)
                if bad:
                    accepted_dependency_violations[cid] = bad

        adjacency: Dict[str, List[str]] = {
            str(claim.get("id")): [str(d) for d in claim.get("depends_on", []) if str(claim.get("id")) != str(d)]
            for claim in claims
            if claim.get("id")
        }

        cycles = self._find_cycles(adjacency)
        is_valid = (
            not missing_dependencies
            and not self_dependencies
            and not invalid_statuses
            and not invalid_claim_ids
            and not duplicate_claim_ids
            and not accepted_dependency_violations
            and not cycles
        )

        return {
            "is_valid": is_valid,
            "missing_dependencies": missing_dependencies,
            "self_dependencies": self_dependencies,
            "invalid_statuses": invalid_statuses,
            "invalid_claim_ids": sorted(set(invalid_claim_ids)),
            "duplicate_claim_ids": sorted(set(duplicate_claim_ids)),
            "accepted_dependency_violations": accepted_dependency_violations,
            "cycles": cycles,
        }

    def _validate_claim_id(self, claim_id: str) -> Optional[str]:
        cid = str(claim_id or "").strip()
        if not cid:
            return "claim_id must be non-empty"
        if not self._CLAIM_ID_RE.match(cid):
            return (
                f"invalid claim_id '{cid}'. Allowed pattern: [A-Za-z0-9_.-]+ "
                "(this prevents proof/check filename collisions)"
            )
        return None

    def _validate_claim_ids(self, claim_ids: List[str], field_name: str) -> Optional[str]:
        for cid in claim_ids:
            err = self._validate_claim_id(str(cid))
            if err:
                return f"{field_name} contains invalid claim id: {err}"
        return None

    def _validate_transition(self, old_status: str, new_status: str, context_action: str) -> Optional[str]:
        if old_status == new_status:
            return None
        allowed_next = self._ALLOWED_TRANSITIONS.get(old_status, set())
        if new_status not in allowed_next:
            return (
                f"invalid status transition in {context_action}: "
                f"{old_status} -> {new_status}. Allowed next: {sorted(allowed_next)}"
            )
        return None

    @staticmethod
    def _find_cycles(adjacency: Dict[str, List[str]]) -> List[List[str]]:
        visited: Dict[str, int] = {}
        stack: List[str] = []
        cycles: List[List[str]] = []

        def dfs(node: str) -> None:
            visited[node] = 1
            stack.append(node)
            for nxt in adjacency.get(node, []):
                if nxt not in adjacency:
                    continue
                if visited.get(nxt, 0) == 0:
                    dfs(nxt)
                elif visited.get(nxt) == 1:
                    if nxt in stack:
                        i = stack.index(nxt)
                        cycles.append(stack[i:] + [nxt])
            stack.pop()
            visited[node] = 2

        for n in adjacency:
            if visited.get(n, 0) == 0:
                dfs(n)

        dedup: List[List[str]] = []
        seen = set()
        for c in cycles:
            key = tuple(c)
            if key not in seen:
                seen.add(key)
                dedup.append(c)
        return dedup

    @staticmethod
    def _now_iso() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _error(message: str) -> str:
        return json.dumps({"success": False, "error": message}, indent=2)


# ---------------------------------------------------------------------------
# Standalone helpers for tree search integration
# ---------------------------------------------------------------------------

def load_claim_graph(workspace_dir: str, subdir: str = "math_workspace") -> Dict[str, Any]:
    """Load and return the claim_graph.json dict from *workspace_dir*."""
    path = os.path.join(workspace_dir, subdir, "claim_graph.json")
    if not os.path.exists(path):
        return {"claims": []}
    with open(path) as f:
        return json.load(f)


def get_frontier_claims(
    claim_graph: Dict[str, Any],
    completed_claims: List[str],
) -> List[Dict[str, Any]]:
    """Return claims whose dependencies are all in *completed_claims*.

    A claim is *frontier* when it is not yet resolved (not in
    completed_claims) and every entry in its ``depends_on`` is resolved.
    Skips claims with status ``accepted`` or ``rejected``.
    """
    completed = set(completed_claims)
    frontier: List[Dict[str, Any]] = []
    for c in claim_graph.get("claims", []):
        cid = c["id"]
        if cid in completed:
            continue
        if c.get("status") in ("accepted", "rejected"):
            continue
        if all(d in completed for d in c.get("depends_on", [])):
            frontier.append(c)
    return frontier


def get_downstream_impact(claim_id: str, claim_graph: Dict[str, Any]) -> int:
    """Count claims that transitively depend on *claim_id*."""
    dependents: Dict[str, set] = {}
    for c in claim_graph.get("claims", []):
        for dep in c.get("depends_on", []):
            dependents.setdefault(dep, set()).add(c["id"])
    visited: set = set()
    stack = [claim_id]
    while stack:
        cid = stack.pop()
        if cid in visited:
            continue
        visited.add(cid)
        stack.extend(dependents.get(cid, set()))
    return len(visited) - 1
