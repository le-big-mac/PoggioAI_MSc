"""
Webhook server — receives research ideas via JSON POST.

Endpoints:
    POST /idea        — JSON API  (web form / programmatic submissions)
    GET  /ideas       — list all ideas in the queue (JSON)
    GET  /health      — health check

Usage:
    python -m consortium.interaction.webhook_server --port 5003

Environment variables:
    IDEA_QUEUE_PATH         — path to ideas.json (default: repo_root/ideas.json)
    WEBHOOK_AUTH_TOKEN      — bearer token for /idea endpoint (optional)
"""

from __future__ import annotations

import os

from flask import Flask, request, jsonify, Response

from .idea_queue import submit_idea, list_ideas


def create_app(queue_path: str | None = None) -> Flask:
    app = Flask(__name__)

    @app.after_request
    def _add_cors(response):
        """Allow cross-origin requests."""
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
        return response

    @app.route("/idea", methods=["OPTIONS"])
    @app.route("/ideas", methods=["OPTIONS"])
    @app.route("/health", methods=["OPTIONS"])
    def cors_preflight():
        """Handle CORS preflight requests."""
        return Response(status=204)

    @app.route("/idea", methods=["POST"])
    def submit_idea_api():
        """Receive an idea via JSON POST (web form / programmatic)."""
        expected_token = os.environ.get("WEBHOOK_AUTH_TOKEN")
        if expected_token:
            auth_header = request.headers.get("Authorization", "")
            if auth_header != f"Bearer {expected_token}":
                return jsonify({"error": "unauthorized"}), 401

        data = request.get_json(silent=True) or {}
        idea = data.get("idea", "").strip()
        sender = data.get("sender", "webform")

        if not idea:
            return jsonify({"error": "idea field is required"}), 400

        entry = submit_idea(
            idea=idea,
            source="webform",
            sender=sender,
            queue_path=queue_path,
        )
        return jsonify({"ok": True, "id": entry["id"], "position": len(list_ideas(queue_path))}), 201

    @app.route("/ideas", methods=["GET"])
    def get_ideas():
        """List all ideas in the queue."""
        ideas = list_ideas(queue_path)
        return jsonify(ideas)

    @app.route("/health", methods=["GET"])
    def health():
        return jsonify({"status": "ok"})

    return app


def main():
    import argparse
    from dotenv import load_dotenv
    load_dotenv(override=False)

    parser = argparse.ArgumentParser(description="Idea webhook server")
    parser.add_argument("--host", default="0.0.0.0", help="Bind address")
    parser.add_argument("--port", type=int, default=5003, help="Port")
    parser.add_argument("--queue-path", default=None, help="Path to ideas.json")
    args = parser.parse_args()

    app = create_app(queue_path=args.queue_path)
    print(f"[webhook] Listening on http://{args.host}:{args.port}")
    print(f"[webhook]   POST /idea      — JSON API for submissions")
    print(f"[webhook]   GET  /ideas     — list queued ideas")
    app.run(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
