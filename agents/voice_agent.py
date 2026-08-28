"""
agents/voice_agent.py
──────────────────────
VoiceAgent — Voiceflow integration for voice input/output.

Provides:
  • send_to_voiceflow()  — POST a text message to a Voiceflow project and
                           get back the agent's reply text.
  • run()                — LangGraph node that conditionally calls Voiceflow
                           when voice_mode is enabled.

Falls back silently to text when:
  • VOICEFLOW_API_KEY / VOICEFLOW_PROJECT_ID are not set
  • Voiceflow API is unreachable
  • voice_mode flag is False (default)

Voiceflow REST Interact API reference:
  POST https://general-runtime.voiceflow.com/state/user/{user_id}/interact
"""

from __future__ import annotations
import os
from typing import Optional, Dict, Any

import requests


# ── Config ────────────────────────────────────────────────────────────────────
VF_API_KEY       = os.getenv("VOICEFLOW_API_KEY", "")
VF_PROJECT_ID    = os.getenv("VOICEFLOW_PROJECT_ID", "")
VF_RUNTIME_URL   = "https://general-runtime.voiceflow.com"
VF_VERSION_ID    = os.getenv("VOICEFLOW_VERSION_ID", "production")
VF_TIMEOUT       = 10   # seconds


def _is_configured() -> bool:
    return bool(VF_API_KEY and VF_PROJECT_ID)


# ── Voiceflow Interact ────────────────────────────────────────────────────────

def send_to_voiceflow(user_id: str, text: str) -> Optional[str]:
    """
    Send a text utterance to Voiceflow and collect the response text.

    Returns concatenated text from all 'speak' / 'text' traces,
    or None on failure.
    """
    if not _is_configured():
        return None

    url = f"{VF_RUNTIME_URL}/state/user/{user_id}/interact"
    headers = {
        "Authorization": VF_API_KEY,
        "Content-Type":  "application/json",
        "versionID":     VF_VERSION_ID,
    }
    payload: Dict[str, Any] = {
        "action": {
            "type": "text",
            "payload": text,
        }
    }

    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=VF_TIMEOUT)
        resp.raise_for_status()
        traces = resp.json()

        # Collect all text / speak responses
        parts = []
        for trace in traces:
            trace_type = trace.get("type", "")
            payload_data = trace.get("payload", {})
            if trace_type in ("speak", "text"):
                msg = payload_data.get("message") or payload_data.get("ssml") or ""
                # Strip SSML tags for plain-text display
                import re
                msg = re.sub(r"<[^>]+>", "", msg).strip()
                if msg:
                    parts.append(msg)

        return "\n".join(parts) if parts else None

    except requests.RequestException as exc:
        print(f"[VoiceAgent] Voiceflow request failed: {exc}")
        return None


def launch_voiceflow_session(user_id: str) -> bool:
    """
    Launch a new Voiceflow session (send 'launch' action).
    Returns True on success.
    """
    if not _is_configured():
        return False

    url = f"{VF_RUNTIME_URL}/state/user/{user_id}/interact"
    headers = {
        "Authorization": VF_API_KEY,
        "Content-Type":  "application/json",
        "versionID":     VF_VERSION_ID,
    }
    try:
        resp = requests.post(
            url,
            json={"action": {"type": "launch"}},
            headers=headers,
            timeout=VF_TIMEOUT,
        )
        resp.raise_for_status()
        return True
    except Exception as exc:
        print(f"[VoiceAgent] Launch failed: {exc}")
        return False


# ── LangGraph node ────────────────────────────────────────────────────────────

def run(state: dict) -> dict:
    """
    LangGraph node: optionally hand off to Voiceflow.

    State keys consumed:
      voice_mode : bool   — if True, route through Voiceflow
      query      : str    — user's question
      user_id    : str    — session identifier
      explanation: str    — text output from TutorAgent (used as VF input context)

    State keys produced:
      voice_response : str | None  — Voiceflow's reply (or None)
    """
    voice_mode = state.get("voice_mode", False)
    state["voice_response"] = None

    if not voice_mode or not _is_configured():
        return state  # pass-through

    user_id = state.get("user_id", "anonymous")
    query   = state.get("query", "")

    vf_reply = send_to_voiceflow(user_id, query)
    if vf_reply:
        state["voice_response"] = vf_reply

    return state
