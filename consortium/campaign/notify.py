"""
Campaign notifications — dispatch status updates to external channels.

Supports:
  - Slack incoming webhooks (SLACK_WEBHOOK_URL)
  - Telegram bot API (TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID)
  - SMS via Twilio (TWILIO_ACCOUNT_SID + TWILIO_AUTH_TOKEN + TWILIO_FROM_NUMBER + SMS_TO_NUMBER)
  - Push notifications via ntfy.sh (NTFY_TOPIC)
  - stdout fallback (always)

All notification failures are silently swallowed — notifications must never
interrupt the campaign.
"""

from __future__ import annotations

import json
import os
import tempfile
import zipfile
from typing import Optional

import requests

from .spec import NotificationConfig


def notify(
    message: str,
    config: NotificationConfig,
    level: str = "info",
) -> None:
    """
    Dispatch a notification message.

    Args:
        message: Plain-text message to send.
        config:  NotificationConfig (from campaign spec).
        level:   "info" | "success" | "warning" | "error" (used for formatting).
    """
    prefix = {
        "info":    "[consortium]",
        "success": "[consortium] ✓",
        "warning": "[consortium] ⚠",
        "error":   "[consortium] ✗",
    }.get(level, "[consortium]")

    full_message = f"{prefix} {message}"
    print(full_message)

    if config.slack_webhook:
        _slack(full_message, config.slack_webhook)

    if config.telegram_bot_token and config.telegram_chat_id:
        _telegram(full_message, config.telegram_bot_token, config.telegram_chat_id)

    if config.ntfy_topic:
        _ntfy(full_message, config.ntfy_topic, config.ntfy_server)

    if config.twilio_account_sid and config.sms_to_number:
        _sms(full_message, config.twilio_account_sid, config.twilio_auth_token,
             config.twilio_from_number, config.sms_to_number)

    if config.twilio_account_sid and config.whatsapp_from and config.whatsapp_to:
        _whatsapp(full_message, config.twilio_account_sid, config.twilio_auth_token,
                  config.whatsapp_from, config.whatsapp_to)


def notify_stage_complete(stage_id: str, workspace: str, config: NotificationConfig) -> None:
    notify(
        f"Stage '{stage_id}' completed. Workspace: {workspace}",
        config,
        level="success",
    )
    notify_stage_pdfs(stage_id, workspace, config)


def notify_stage_failed(stage_id: str, reason: str, config: NotificationConfig) -> None:
    notify(
        f"Stage '{stage_id}' FAILED: {reason}",
        config,
        level="error",
    )


def notify_stage_launched(stage_id: str, pid: int, workspace: str, config: NotificationConfig) -> None:
    notify(
        f"Stage '{stage_id}' launched (PID {pid}). Workspace: {workspace}",
        config,
        level="info",
    )


def notify_campaign_complete(campaign_name: str, config: NotificationConfig) -> None:
    notify(
        f"Campaign '{campaign_name}' is fully complete.",
        config,
        level="success",
    )


def notify_repair_started(
    stage_id: str, attempt: int, max_attempts: int, config: NotificationConfig,
) -> None:
    notify(
        f"Stage '{stage_id}' repair attempt {attempt}/{max_attempts} — "
        f"deploying Claude Code agent.",
        config,
        level="warning",
    )


def notify_repair_succeeded(
    stage_id: str, attempt: int, diagnosis: str, config: NotificationConfig,
) -> None:
    notify(
        f"Stage '{stage_id}' repair succeeded (attempt {attempt}). "
        f"Diagnosis: {diagnosis[:150]}. Retrying stage.",
        config,
        level="success",
    )


def notify_repair_failed(
    stage_id: str, attempt: int, max_attempts: int, error: str, config: NotificationConfig,
) -> None:
    remaining = max_attempts - attempt
    if remaining > 0:
        suffix = f" {remaining} attempt(s) remaining."
    else:
        suffix = " All repair attempts exhausted — human attention required."
    notify(
        f"Stage '{stage_id}' repair failed (attempt {attempt}/{max_attempts}): "
        f"{error[:150]}.{suffix}",
        config,
        level="error",
    )


def notify_heartbeat(summary: str, config: NotificationConfig) -> None:
    if config.on_heartbeat:
        notify(summary, config, level="info")


# ------------------------------------------------------------------
# Transport implementations
# ------------------------------------------------------------------

def _slack(message: str, webhook_url: str) -> None:
    try:
        resp = requests.post(
            webhook_url,
            json={"text": message},
            timeout=10,
        )
        resp.raise_for_status()
    except Exception as e:
        print(f"[campaign:notify] Slack delivery failed: {e}")


def _telegram(message: str, bot_token: str, chat_id: str) -> None:
    try:
        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        resp = requests.post(
            url,
            json={"chat_id": chat_id, "text": message},
            timeout=10,
        )
        resp.raise_for_status()
    except Exception as e:
        print(f"[campaign:notify] Telegram delivery failed: {e}")


def _ntfy(message: str, topic: str, server: Optional[str] = None) -> None:
    try:
        base = (server or "https://ntfy.sh").rstrip("/")
        resp = requests.post(
            f"{base}/{topic}",
            data=message.encode("utf-8"),
            headers={"Title": "OpenClaw Campaign"},
            timeout=10,
        )
        resp.raise_for_status()
    except Exception as e:
        print(f"[campaign:notify] ntfy delivery failed: {e}")


def _sms(message: str, account_sid: str, auth_token: str,
         from_number: str, to_number: str) -> None:
    try:
        resp = requests.post(
            f"https://api.twilio.com/2010-04-01/Accounts/{account_sid}/Messages.json",
            auth=(account_sid, auth_token),
            data={"From": from_number, "To": to_number, "Body": message[:1600]},
            timeout=10,
        )
        resp.raise_for_status()
    except Exception as e:
        print(f"[campaign:notify] SMS delivery failed: {e}")


def _whatsapp(message: str, account_sid: str, auth_token: str,
              from_number: str, to_number: str) -> None:
    """Send a WhatsApp message via Twilio (same API as SMS, whatsapp: prefix on numbers)."""
    try:
        resp = requests.post(
            f"https://api.twilio.com/2010-04-01/Accounts/{account_sid}/Messages.json",
            auth=(account_sid, auth_token),
            data={"From": from_number, "To": to_number, "Body": message[:1600]},
            timeout=10,
        )
        resp.raise_for_status()
    except Exception as e:
        print(f"[campaign:notify] WhatsApp delivery failed: {e}")


def _telegram_document(
    file_path: str, caption: str, bot_token: str, chat_id: str,
) -> None:
    """Send a document (file) via Telegram Bot API sendDocument."""
    try:
        url = f"https://api.telegram.org/bot{bot_token}/sendDocument"
        with open(file_path, "rb") as f:
            resp = requests.post(
                url,
                data={"chat_id": chat_id, "caption": caption[:1024]},
                files={"document": (os.path.basename(file_path), f)},
                timeout=120,
            )
        resp.raise_for_status()
    except Exception as e:
        print(f"[campaign:notify] Telegram document delivery failed: {e}")


def _collect_stage_pdfs(workspace: str) -> list[str]:
    """Collect all summary/report PDFs from a stage workspace."""
    pdfs: list[str] = []

    for subdir in ("stage_summaries", "milestone_reports"):
        d = os.path.join(workspace, subdir)
        if os.path.isdir(d):
            for fname in sorted(os.listdir(d)):
                if fname.endswith(".pdf"):
                    pdfs.append(os.path.join(d, fname))

    for special in ("final_paper.pdf", "claim_graph_summary.pdf"):
        for parent in (workspace, os.path.join(workspace, "paper_workspace")):
            candidate = os.path.join(parent, special)
            if os.path.isfile(candidate):
                pdfs.append(candidate)

    return pdfs


def _bundle_pdfs_to_zip(pdfs: list[str], stage_id: str) -> Optional[str]:
    """Bundle PDF paths into a temporary zip. Returns zip path or None."""
    if not pdfs:
        return None
    try:
        tmp = tempfile.NamedTemporaryFile(
            prefix=f"{stage_id}_summaries_", suffix=".zip", delete=False,
        )
        tmp_path = tmp.name
        tmp.close()

        with zipfile.ZipFile(tmp_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for pdf_path in pdfs:
                parent = os.path.basename(os.path.dirname(pdf_path))
                arcname = f"{parent}/{os.path.basename(pdf_path)}" if parent else os.path.basename(pdf_path)
                zf.write(pdf_path, arcname)

        return tmp_path
    except Exception as e:
        print(f"[campaign:notify] Failed to create PDF zip bundle: {e}")
        return None


def notify_stage_pdfs(
    stage_id: str, workspace: str, config: NotificationConfig,
) -> None:
    """Send stage summary PDFs as a zip to Telegram. Fail-safe."""
    if not (config.telegram_bot_token and config.telegram_chat_id):
        return

    try:
        pdfs = _collect_stage_pdfs(workspace)
        if not pdfs:
            print(f"[campaign:notify] No PDFs found for stage '{stage_id}' — skipping Telegram document.")
            return

        zip_path = _bundle_pdfs_to_zip(pdfs, stage_id)
        if not zip_path:
            return

        try:
            caption = (
                f"[consortium] Stage '{stage_id}' complete — "
                f"{len(pdfs)} PDF summary/report file(s)."
            )
            _telegram_document(zip_path, caption, config.telegram_bot_token, config.telegram_chat_id)
        finally:
            try:
                os.unlink(zip_path)
            except OSError:
                pass
    except Exception as e:
        print(f"[campaign:notify] notify_stage_pdfs failed: {e}")
