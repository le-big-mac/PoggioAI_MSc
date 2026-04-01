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


def _read_queue(path: str) -> list:
    """Read queue from file (no locking — caller must hold the lock)."""
    if not os.path.exists(path):
        return []
    with open(path, "r") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return []


def _write_queue(path: str, data: list) -> None:
    """Write queue to file (no locking — caller must hold the lock)."""
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def _with_exclusive_lock(path: str, fn):
    """Run fn(queue) under an exclusive lock on a sidecar .lock file.

    Uses a separate .lock file so we can open the data file for both
    reading and writing without truncating it on open.
    """
    lock_path = path + ".lock"
    lock_fd = os.open(lock_path, os.O_CREAT | os.O_RDWR)
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        queue = _read_queue(path)
        result = fn(queue)
        _write_queue(path, queue)
        return result
    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        os.close(lock_fd)


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

    def _append(queue: list):
        queue.append(entry)

    _with_exclusive_lock(path, _append)
    return entry


def next_pending(queue_path: Optional[str] = None) -> Optional[dict]:
    """Return the oldest pending idea, or None."""
    path = queue_path or _default_queue_path()
    # Read-only — still take exclusive lock for consistency with writers
    result = [None]

    def _find(queue: list):
        for entry in queue:
            if entry["status"] == "pending":
                result[0] = entry
                return

    _with_exclusive_lock(path, _find)
    return result[0]


def mark_running(idea_id: str, workspace: str, queue_path: Optional[str] = None) -> None:
    """Mark an idea as running with its workspace path."""
    path = queue_path or _default_queue_path()

    def _update(queue: list):
        for entry in queue:
            if entry["id"] == idea_id:
                entry["status"] = "running"
                entry["workspace"] = workspace
                entry["started_at"] = datetime.now(timezone.utc).isoformat()
                break

    _with_exclusive_lock(path, _update)


def mark_completed(idea_id: str, queue_path: Optional[str] = None) -> None:
    path = queue_path or _default_queue_path()

    def _update(queue: list):
        for entry in queue:
            if entry["id"] == idea_id:
                entry["status"] = "completed"
                entry["completed_at"] = datetime.now(timezone.utc).isoformat()
                break

    _with_exclusive_lock(path, _update)


def mark_failed(idea_id: str, queue_path: Optional[str] = None) -> None:
    path = queue_path or _default_queue_path()

    def _update(queue: list):
        for entry in queue:
            if entry["id"] == idea_id:
                entry["status"] = "failed"
                entry["completed_at"] = datetime.now(timezone.utc).isoformat()
                break

    _with_exclusive_lock(path, _update)


def list_ideas(queue_path: Optional[str] = None) -> List[dict]:
    """Return all ideas in the queue."""
    path = queue_path or _default_queue_path()
    result = []

    def _copy(queue: list):
        result.extend(queue)

    _with_exclusive_lock(path, _copy)
    return result
