"""
Webhook server — receives research ideas via WhatsApp (Twilio) and web form POST.

Endpoints:
    POST /whatsapp    — Twilio webhook (receives WhatsApp messages)
    POST /idea        — JSON API  (web form / programmatic submissions)
    GET  /ideas       — list all ideas in the queue (JSON)
    GET  /health      — health check

Twilio webhook setup:
    In your Twilio Console → WhatsApp Sandbox (or production number),
    set the "WHEN A MESSAGE COMES IN" webhook URL to:
        https://your-domain.com/whatsapp

    For development, use ngrok:
        ngrok http 5003
    Then set the Twilio webhook to the ngrok HTTPS URL + /whatsapp.

Usage:
    # Standalone:
    python -m consortium.interaction.webhook_server --port 5003

    # From code:
    from consortium.interaction.webhook_server import create_app
    app = create_app()
    app.run(host="0.0.0.0", port=5003)

Environment variables:
    IDEA_QUEUE_PATH         — path to ideas.json (default: repo_root/ideas.json)
    TWILIO_ACCOUNT_SID      — for signature validation (optional but recommended)
    TWILIO_AUTH_TOKEN        — for signature validation (optional but recommended)
    WHATSAPP_FROM           — Twilio WhatsApp sender (e.g. whatsapp:+14155238886)
    WEBHOOK_AUTH_TOKEN      — bearer token for /idea endpoint (optional)
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import os
from xml.sax.saxutils import escape as xml_escape

from flask import Flask, request, jsonify, Response

from .idea_queue import submit_idea, list_ideas


def _validate_twilio_signature(url: str, params: dict, signature: str, auth_token: str) -> bool:
    """Validate Twilio webhook signature (X-Twilio-Signature header).

    Twilio computes HMAC-SHA1 over: URL + sorted(param key+value pairs),
    then base64-encodes the digest.
    """
    sorted_params = sorted(params.items())
    data = url + "".join(f"{k}{v}" for k, v in sorted_params)
    expected = hmac.new(
        auth_token.encode("utf-8"),
        data.encode("utf-8"),
        hashlib.sha1,
    ).digest()
    expected_b64 = base64.b64encode(expected).decode("utf-8")
    return hmac.compare_digest(expected_b64, signature)


def _get_twilio_url(req) -> str:
    """Get the URL Twilio signed against.

    Behind a reverse proxy (ngrok, cloudflare tunnel, etc.), request.url
    may show http:// but Twilio signed the https:// URL it actually called.
    Use X-Forwarded-Proto to reconstruct the correct URL.
    """
    url = req.url
    forwarded_proto = req.headers.get("X-Forwarded-Proto")
    if forwarded_proto and url.startswith("http://"):
        url = "https://" + url[len("http://"):]
    return url


def create_app(queue_path: str | None = None) -> Flask:
    app = Flask(__name__)

    @app.after_request
    def _add_cors(response):
        """Allow cross-origin requests (GitHub Pages form → this server)."""
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

    @app.route("/whatsapp", methods=["POST"])
    def whatsapp_webhook():
        """Receive an incoming WhatsApp message from Twilio."""
        body = request.form.get("Body", "").strip()
        sender = request.form.get("From", "")  # e.g. whatsapp:+447...

        if not body:
            return Response(
                '<?xml version="1.0" encoding="UTF-8"?><Response/>',
                content_type="text/xml",
            )

        # Optional: validate Twilio signature
        auth_token = os.environ.get("TWILIO_AUTH_TOKEN")
        if auth_token:
            sig = request.headers.get("X-Twilio-Signature", "")
            url = _get_twilio_url(request)
            if not _validate_twilio_signature(url, request.form.to_dict(), sig, auth_token):
                return Response("Forbidden", status=403)

        entry = submit_idea(
            idea=body,
            source="whatsapp",
            sender=sender,
            queue_path=queue_path,
        )

        # TwiML response — XML-escape all user-controlled text
        reply = (
            f"Got it! Your research idea has been queued for investigation.\n"
            f"ID: {entry['id'][:8]}\n"
            f"I'll message you when the analysis starts and when results are ready."
        )
        twiml = (
            '<?xml version="1.0" encoding="UTF-8"?>'
            f"<Response><Message>{xml_escape(reply)}</Message></Response>"
        )
        return Response(twiml, content_type="text/xml")

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
    print(f"[webhook]   POST /whatsapp  — Twilio WhatsApp webhook")
    print(f"[webhook]   POST /idea      — JSON API for web form submissions")
    print(f"[webhook]   GET  /ideas     — list queued ideas")
    app.run(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
