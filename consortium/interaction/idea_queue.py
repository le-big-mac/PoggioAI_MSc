"""
Idea queue — file-backed queue for research ideas submitted via WhatsApp or web form.

Ideas are stored as JSON objects in a single file with file-locking to allow
concurrent reads/writes from the webhook server and the heartbeat consumer.

Queue file format (ideas.json):
    [
        {
            "id": "uuid",
            "idea": "research idea text",
            "source": "whatsapp" | "webform",
            "sender": "+447...",
            "submitted_at": "2026-04-01T12:00:00Z",
            "status": "pending" | "running" | "completed" | "failed",
            "workspace": null,
            "started_at": null,
            "completed_at": null
        },
        ...
    ]
"""

from __future__ import annotations

import fcntl
import json
import os
import uuid
from datetime import datetime, timezone
from typing import List, Optional


def _default_queue_path() -> str:
    return os.environ.get(
        "IDEA_QUEUE_PATH",
        os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__)))), "ideas.json"),
    )


def _read_locked(path: str) -> list:
    if not os.path.exists(path):
        return []
    with open(path, "r") as f:
        fcntl.flock(f, fcntl.LOCK_SH)
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            data = []
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)
    return data


def _write_locked(path: str, data: list) -> None:
    with open(path, "w") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            json.dump(data, f, indent=2)
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def submit_idea(
    idea: str,
    source: str = "whatsapp",
    sender: str = "",
    queue_path: Optional[str] = None,
) -> dict:
    """Add a new idea to the queue. Returns the created entry."""
    path = queue_path or _default_queue_path()
    entry = {
        "id": str(uuid.uuid4()),
        "idea": idea.strip(),
        "source": source,
        "sender": sender,
        "submitted_at": datetime.now(timezone.utc).isoformat(),
        "status": "pending",
        "workspace": None,
        "started_at": None,
        "completed_at": None,
    }
    queue = _read_locked(path)
    queue.append(entry)
    _write_locked(path, queue)
    return entry


def next_pending(queue_path: Optional[str] = None) -> Optional[dict]:
    """Return the oldest pending idea, or None."""
    path = queue_path or _default_queue_path()
    queue = _read_locked(path)
    for entry in queue:
        if entry["status"] == "pending":
            return entry
    return None


def mark_running(idea_id: str, workspace: str, queue_path: Optional[str] = None) -> None:
    """Mark an idea as running with its workspace path."""
    path = queue_path or _default_queue_path()
    queue = _read_locked(path)
    for entry in queue:
        if entry["id"] == idea_id:
            entry["status"] = "running"
            entry["workspace"] = workspace
            entry["started_at"] = datetime.now(timezone.utc).isoformat()
            break
    _write_locked(path, queue)


def mark_completed(idea_id: str, queue_path: Optional[str] = None) -> None:
    path = queue_path or _default_queue_path()
    queue = _read_locked(path)
    for entry in queue:
        if entry["id"] == idea_id:
            entry["status"] = "completed"
            entry["completed_at"] = datetime.now(timezone.utc).isoformat()
            break
    _write_locked(path, queue)


def mark_failed(idea_id: str, queue_path: Optional[str] = None) -> None:
    path = queue_path or _default_queue_path()
    queue = _read_locked(path)
    for entry in queue:
        if entry["id"] == idea_id:
            entry["status"] = "failed"
            entry["completed_at"] = datetime.now(timezone.utc).isoformat()
            break
    _write_locked(path, queue)


def list_ideas(queue_path: Optional[str] = None) -> List[dict]:
    """Return all ideas in the queue."""
    path = queue_path or _default_queue_path()
    return _read_locked(path)
