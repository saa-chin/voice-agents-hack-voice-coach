"""Coverage for web-py/backend/app/main.py — FastAPI WS server.

Cactus is mocked entirely so this suite runs without the dylib or model
weights. The mock honours the same surface area the real server uses:
  - holder.cactus.cactus_reset(model)
  - holder.cactus.cactus_init(...) — not called here; we pre-load the holder
  - chat._cactus_complete_audio(cactus, model, msgs, opts, pcm, callback)
"""
from __future__ import annotations

import base64
import json
import struct
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient


def _silence_pcm(seconds: float = 1.0, sample_value: int = 1500) -> bytes:
    n = int(16000 * seconds)
    return struct.pack(f"<{n}h", *([sample_value] * n))


def _b64_pcm(seconds: float = 1.0) -> str:
    return base64.b64encode(_silence_pcm(seconds)).decode("ascii")


@pytest.fixture
def server(monkeypatch, tmp_path):
    """A fresh server with model pre-loaded as a stub.

    The lifespan handler eagerly schedules `holder.ensure_loaded()` on app
    start; we replace that with an async no-op so individual tests get full
    control of the holder state at WebSocket handshake time.
    """
    from app import main as srv

    # Sandbox session log per test.
    import coach
    monkeypatch.setattr(coach, "SESSION_DIR", tmp_path / "sessions")

    # Pre-load the model holder so /ws/coach skips the loading dance.
    fake_cactus = MagicMock()
    fake_cactus.cactus_reset = MagicMock(return_value=None)
    srv.holder.cactus = fake_cactus
    srv.holder.model = MagicMock(name="model_handle")
    srv.holder.weights_path = tmp_path / "fake_weights"
    srv.holder.load_error = None

    async def _noop_ensure_loaded():
        return None

    monkeypatch.setattr(srv.holder, "ensure_loaded", _noop_ensure_loaded)

    # Default canned reply — overridable per-test.
    state = {
        "responses": [
            json.dumps({
                "heard": "test",
                "ack": "Nice",
                "feedback": "Good first try.",
                "next_action": "advance",
                "metrics_observed": {"matched_prompt": True, "loudness_ok": True},
            })
        ],
        "raise_on_call": None,
        "calls": [],
    }

    def fake_audio(cactus_mod, model, messages_json, options_json, pcm, on_token):
        state["calls"].append({
            "messages": json.loads(messages_json),
            "options": json.loads(options_json),
            "pcm_len": len(pcm),
        })
        if state["raise_on_call"]:
            raise state["raise_on_call"]
        # If multiple responses configured, pop in order; otherwise reuse last.
        if len(state["responses"]) > 1:
            text = state["responses"].pop(0)
        else:
            text = state["responses"][0]
        if on_token is not None:
            on_token(text, 0)
        return ""  # envelope buffer empty for our stub

    monkeypatch.setattr(srv.chat, "_cactus_complete_audio", fake_audio)

    yield {"app": srv.app, "holder": srv.holder, "state": state}

    # Reset holder so the next test starts clean.
    srv.holder.cactus = None
    srv.holder.model = None
    srv.holder.weights_path = None
    srv.holder.load_error = None


# ---- HTTP endpoints ------------------------------------------------------

def test_health_reports_loaded_model(server):
    with TestClient(server["app"]) as client:
        r = client.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"
        assert body["model_loaded"] is True
        assert body["model_load_error"] is None
        assert "gemma-4" in body["model_id"]


def test_health_reports_unloaded(server):
    server["holder"].model = None
    with TestClient(server["app"]) as client:
        r = client.get("/health")
        assert r.json()["model_loaded"] is False


def test_drills_endpoint_returns_full_set(server):
    import content
    with TestClient(server["app"]) as client:
        r = client.get("/api/drills")
        assert r.status_code == 200
        drills = r.json()
        assert len(drills) == len(content.default_drill_set())
        for d in drills:
            assert {"stage", "index", "prompt", "note", "target_dbfs"} <= d.keys()


# ---- WebSocket: protocol ready/start ------------------------------------

def test_ws_sends_ready_when_model_already_loaded(server):
    with TestClient(server["app"]) as client:
        with client.websocket_connect("/ws/coach") as ws:
            msg = ws.receive_json()
            assert msg["type"] == "ready"


def test_ws_sends_loading_then_ready_if_not_loaded(server, monkeypatch):
    """When the holder reports unloaded, server emits 'loading' before 'ready'.

    We swap `ensure_loaded` AFTER TestClient enters so the lifespan startup
    task (which calls the fixture's no-op) doesn't race us by setting `model`
    before our WebSocket handler runs.
    """
    holder = server["holder"]
    cached = holder.model

    with TestClient(server["app"]) as client:
        # Now lifespan has finished. Mark unloaded for the upcoming WS.
        holder.model = None

        async def fake_ensure_loaded():
            holder.model = cached

        monkeypatch.setattr(holder, "ensure_loaded", fake_ensure_loaded)

        with client.websocket_connect("/ws/coach") as ws:
            assert ws.receive_json()["type"] == "loading"
            assert ws.receive_json()["type"] == "ready"


def test_ws_reports_load_error_to_client(server, monkeypatch):
    holder = server["holder"]

    with TestClient(server["app"]) as client:
        holder.model = None
        holder.load_error = None

        async def fake_ensure_loaded():
            holder.load_error = "missing weights"

        monkeypatch.setattr(holder, "ensure_loaded", fake_ensure_loaded)

        with client.websocket_connect("/ws/coach") as ws:
            assert ws.receive_json()["type"] == "loading"
            err = ws.receive_json()
            assert err["type"] == "error"
            assert "missing weights" in err["message"]


def test_ws_unexpected_first_message_returns_error(server):
    with TestClient(server["app"]) as client:
        with client.websocket_connect("/ws/coach") as ws:
            assert ws.receive_json()["type"] == "ready"
            ws.send_json({"type": "audio", "pcm_b64": "", "sample_rate": 16000})
            err = ws.receive_json()
            assert err["type"] == "error"
            assert "start_session" in err["message"]


# ---- WebSocket: drill loop happy path -----------------------------------

def test_ws_happy_path_first_turn_advance(server):
    with TestClient(server["app"]) as client:
        with client.websocket_connect("/ws/coach") as ws:
            assert ws.receive_json()["type"] == "ready"
            ws.send_json({"type": "start_session"})

            drill0 = ws.receive_json()
            assert drill0["type"] == "drill"
            assert drill0["position"] == 0
            assert drill0["total"] >= 5
            assert drill0["stage"] == "warmup"

            ws.send_json({"type": "audio", "pcm_b64": _b64_pcm(1.0), "sample_rate": 16000})

            metrics = ws.receive_json()
            assert metrics["type"] == "metrics"
            assert metrics["dbfs"] is not None
            assert metrics["duration_s"] > 0

            assert ws.receive_json()["type"] == "thinking"

            coach_msg = ws.receive_json()
            assert coach_msg["type"] == "coach"
            assert coach_msg["ack"] == "Nice"
            assert coach_msg["next_action"] == "advance"
            assert coach_msg["matched_prompt"] is True
            assert coach_msg["heard"] == "test"

            assert ws.receive_json()["type"] == "advance"
            drill1 = ws.receive_json()
            assert drill1["type"] == "drill"
            assert drill1["position"] == 1

    # The fake _cactus_complete_audio was invoked exactly once.
    assert len(server["state"]["calls"]) == 1
    call = server["state"]["calls"][0]
    # System prompt must include the drill text.
    sys_prompt = call["messages"][0]["content"]
    assert drill0["prompt"] in sys_prompt
    # PCM length should match what we sent (1s @ 16kHz int16 = 32000 bytes).
    assert call["pcm_len"] == 32000


def test_ws_retry_action_resends_same_drill(server):
    server["state"]["responses"] = [
        json.dumps({"heard": "x", "ack": "ok", "feedback": "again",
                    "next_action": "retry",
                    "metrics_observed": {"matched_prompt": False}}),
        json.dumps({"heard": "y", "ack": "good", "feedback": "moving on",
                    "next_action": "advance",
                    "metrics_observed": {"matched_prompt": True}}),
    ]
    with TestClient(server["app"]) as client:
        with client.websocket_connect("/ws/coach") as ws:
            ws.receive_json()  # ready
            ws.send_json({"type": "start_session"})
            d0 = ws.receive_json()
            assert d0["position"] == 0

            ws.send_json({"type": "audio", "pcm_b64": _b64_pcm(1.0), "sample_rate": 16000})
            ws.receive_json()  # metrics
            ws.receive_json()  # thinking
            ws.receive_json()  # coach (retry)
            assert ws.receive_json()["type"] == "retry"
            redo = ws.receive_json()
            assert redo["type"] == "drill"
            assert redo["position"] == 0  # SAME drill resent

            # Send audio again, this time advance.
            ws.send_json({"type": "audio", "pcm_b64": _b64_pcm(1.0), "sample_rate": 16000})
            ws.receive_json()  # metrics
            ws.receive_json()  # thinking
            ws.receive_json()  # coach (advance)
            assert ws.receive_json()["type"] == "advance"
            assert ws.receive_json()["position"] == 1


def test_ws_rest_command_ends_session(server):
    with TestClient(server["app"]) as client:
        with client.websocket_connect("/ws/coach") as ws:
            ws.receive_json()  # ready
            ws.send_json({"type": "start_session"})
            ws.receive_json()  # drill 0
            ws.send_json({"type": "command", "action": "rest"})
            done = ws.receive_json()
            assert done["type"] == "session_done"
            assert done["summary"]["rest_called"] is True


def test_ws_skip_command_jumps_to_next_drill(server):
    with TestClient(server["app"]) as client:
        with client.websocket_connect("/ws/coach") as ws:
            ws.receive_json()  # ready
            ws.send_json({"type": "start_session"})
            d0 = ws.receive_json()
            ws.send_json({"type": "command", "action": "skip"})
            assert ws.receive_json()["type"] == "advance"
            d1 = ws.receive_json()
            assert d1["type"] == "drill"
            assert d1["position"] == d0["position"] + 1


def test_ws_repeat_prompt_resends_current_drill(server):
    with TestClient(server["app"]) as client:
        with client.websocket_connect("/ws/coach") as ws:
            ws.receive_json()  # ready
            ws.send_json({"type": "start_session"})
            d0 = ws.receive_json()
            ws.send_json({"type": "command", "action": "repeat_prompt"})
            d0_again = ws.receive_json()
            assert d0_again["type"] == "drill"
            assert d0_again["position"] == d0["position"]


# ---- WebSocket: error paths --------------------------------------------

def test_ws_invalid_message_type_returns_error(server):
    with TestClient(server["app"]) as client:
        with client.websocket_connect("/ws/coach") as ws:
            ws.receive_json()  # ready
            ws.send_json({"type": "start_session"})
            ws.receive_json()  # drill
            ws.send_json({"type": "garbage_type"})
            err = ws.receive_json()
            assert err["type"] == "error"
            assert "garbage_type" in err["message"]


def test_ws_invalid_json_returns_error(server):
    with TestClient(server["app"]) as client:
        with client.websocket_connect("/ws/coach") as ws:
            ws.receive_json()  # ready
            ws.send_json({"type": "start_session"})
            ws.receive_json()  # drill
            ws.send_text("not json at all {")
            err = ws.receive_json()
            assert err["type"] == "error"
            assert "JSON" in err["message"] or "json" in err["message"]


def test_ws_audio_too_short_returns_error(server):
    with TestClient(server["app"]) as client:
        with client.websocket_connect("/ws/coach") as ws:
            ws.receive_json()  # ready
            ws.send_json({"type": "start_session"})
            ws.receive_json()  # drill
            ws.send_json({"type": "audio", "pcm_b64": _b64_pcm(0.05), "sample_rate": 16000})
            err = ws.receive_json()
            assert err["type"] == "error"
            assert "too short" in err["message"]


def test_ws_wrong_sample_rate_returns_error(server):
    with TestClient(server["app"]) as client:
        with client.websocket_connect("/ws/coach") as ws:
            ws.receive_json()  # ready
            ws.send_json({"type": "start_session"})
            ws.receive_json()  # drill
            ws.send_json({"type": "audio", "pcm_b64": _b64_pcm(1.0), "sample_rate": 44100})
            err = ws.receive_json()
            assert err["type"] == "error"
            assert "sample_rate" in err["message"]


def test_ws_invalid_pcm_b64_returns_error(server):
    with TestClient(server["app"]) as client:
        with client.websocket_connect("/ws/coach") as ws:
            ws.receive_json()  # ready
            ws.send_json({"type": "start_session"})
            ws.receive_json()  # drill
            ws.send_json({"type": "audio", "pcm_b64": "not~~base64@@", "sample_rate": 16000})
            err = ws.receive_json()
            assert err["type"] == "error"


def test_ws_model_returns_invalid_json_skips_drill(server):
    """If Gemma replies with text we can't parse, server skips that drill."""
    server["state"]["responses"] = ["this is not json at all, just prose"]
    with TestClient(server["app"]) as client:
        with client.websocket_connect("/ws/coach") as ws:
            ws.receive_json()  # ready
            ws.send_json({"type": "start_session"})
            d0 = ws.receive_json()
            ws.send_json({"type": "audio", "pcm_b64": _b64_pcm(1.0), "sample_rate": 16000})
            ws.receive_json()  # metrics
            ws.receive_json()  # thinking
            err = ws.receive_json()
            assert err["type"] == "error"
            assert "JSON" in err["message"]
            # Server should auto-advance past the broken drill.
            assert ws.receive_json()["type"] == "advance"
            d1 = ws.receive_json()
            assert d1["position"] == d0["position"] + 1


def test_ws_model_inference_error_continues_session(server):
    server["state"]["raise_on_call"] = RuntimeError("kaboom")
    with TestClient(server["app"]) as client:
        with client.websocket_connect("/ws/coach") as ws:
            ws.receive_json()  # ready
            ws.send_json({"type": "start_session"})
            ws.receive_json()  # drill
            ws.send_json({"type": "audio", "pcm_b64": _b64_pcm(1.0), "sample_rate": 16000})
            ws.receive_json()  # metrics
            ws.receive_json()  # thinking
            err = ws.receive_json()
            assert err["type"] == "error"
            assert "kaboom" in err["message"]


def test_ws_completes_session_after_last_drill(server):
    """Walking through all drills should end with session_done containing summary."""
    import content
    total = len(content.default_drill_set())

    with TestClient(server["app"]) as client:
        with client.websocket_connect("/ws/coach") as ws:
            ws.receive_json()  # ready
            ws.send_json({"type": "start_session"})
            ws.receive_json()  # first drill

            for i in range(total - 1):
                ws.send_json({"type": "audio", "pcm_b64": _b64_pcm(1.0), "sample_rate": 16000})
                ws.receive_json()  # metrics
                ws.receive_json()  # thinking
                ws.receive_json()  # coach
                ws.receive_json()  # advance
                ws.receive_json()  # next drill

            # Last drill — no advance/drill, then session_done.
            ws.send_json({"type": "audio", "pcm_b64": _b64_pcm(1.0), "sample_rate": 16000})
            ws.receive_json()  # metrics
            ws.receive_json()  # thinking
            ws.receive_json()  # coach
            done = ws.receive_json()
            assert done["type"] == "session_done"
            assert done["summary"]["advanced"] == total
            assert done["summary"]["total"] == total
            assert done["summary"]["retries"] == 0
            assert done["summary"]["rest_called"] is False


# ---- ModelHolder behaviour ---------------------------------------------

def test_model_holder_loaded_property(server):
    holder = server["holder"]
    assert holder.loaded is True
    holder.model = None
    assert holder.loaded is False


def test_pcm_bytes_to_dbfs_roundtrip():
    """Direct probe of the helper used by the audio handler."""
    from app import main as srv
    # Constant value of 16384 → ~-6 dBFS
    pcm = struct.pack("<%dh" % 1600, *([16384] * 1600))
    dbfs, dur = srv._pcm_bytes_to_dbfs(pcm)
    assert dur == pytest.approx(0.1, abs=0.001)
    assert dbfs == pytest.approx(-6.02, abs=0.05)


def test_pcm_bytes_to_dbfs_empty():
    from app import main as srv
    import math
    dbfs, dur = srv._pcm_bytes_to_dbfs(b"")
    assert dur == 0.0
    assert dbfs == -math.inf


def test_ws_coach_says_rest_ends_session(server):
    """When the model returns next_action='rest', the server should end the
    session even without an explicit 'rest' command from the client."""
    server["state"]["responses"] = [json.dumps({
        "heard": "tired", "ack": "Take five.",
        "feedback": "Rest your voice for a bit.",
        "next_action": "rest",
        "metrics_observed": {"matched_prompt": True},
    })]
    with TestClient(server["app"]) as client:
        with client.websocket_connect("/ws/coach") as ws:
            ws.receive_json()  # ready
            ws.send_json({"type": "start_session"})
            ws.receive_json()  # drill 0
            ws.send_json({"type": "audio", "pcm_b64": _b64_pcm(1.0), "sample_rate": 16000})
            ws.receive_json()  # metrics
            ws.receive_json()  # thinking
            coach_msg = ws.receive_json()
            assert coach_msg["next_action"] == "rest"
            done = ws.receive_json()
            assert done["type"] == "session_done"
            assert done["summary"]["rest_called"] is True


def test_ws_cactus_reset_exception_is_logged_but_continues(server, monkeypatch):
    """If cactus_reset raises (e.g. between turns), the loop must keep going."""
    server["holder"].cactus.cactus_reset.side_effect = RuntimeError("reset boom")
    with TestClient(server["app"]) as client:
        with client.websocket_connect("/ws/coach") as ws:
            ws.receive_json()  # ready
            ws.send_json({"type": "start_session"})
            ws.receive_json()  # drill
            ws.send_json({"type": "audio", "pcm_b64": _b64_pcm(1.0), "sample_rate": 16000})
            ws.receive_json()  # metrics
            ws.receive_json()  # thinking
            coach_msg = ws.receive_json()
            # cactus_reset failure is non-fatal — coach response still arrives.
            assert coach_msg["type"] == "coach"
