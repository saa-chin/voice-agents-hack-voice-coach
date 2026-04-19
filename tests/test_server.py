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
from pathlib import Path
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

    # The lifespan handler also schedules `intent_holder.ensure_loaded()`,
    # which would try to download + load FunctionGemma 270M. Stub it so
    # the WS tests run without the model present and so the per-test
    # baseline is "router not loaded → heuristic fallback active".
    srv.intent_holder.cactus = None
    srv.intent_holder.model = None
    srv.intent_holder._classifier = None
    srv.intent_holder.load_error = None
    monkeypatch.setattr(srv.intent_holder, "ensure_loaded", _noop_ensure_loaded)

    # Whisper holder — same treatment. Default state across the suite
    # is "Whisper not loaded", which means `intent_audio` requests
    # respond with a clean error frame. Tests that exercise the audio
    # path inject their own fake transcribe directly on the holder.
    srv.whisper_holder.cactus = None
    srv.whisper_holder.model = None
    srv.whisper_holder.weights_path = None
    srv.whisper_holder.load_error = None
    monkeypatch.setattr(srv.whisper_holder, "ensure_loaded", _noop_ensure_loaded)

    # Force-disable macOS `say` rendering by default so the existing
    # `coach` tests (which assert frame ordering) don't suddenly see
    # an extra `audio_reply` slipped in. The real `_say_to_wav`
    # short-circuits to None when HAS_SAY is False, so we don't need
    # to patch the function itself — direct-probe tests can still
    # exercise the real implementation by setting HAS_SAY back to True.
    monkeypatch.setattr(srv, "HAS_SAY", False)

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
    # Same for the intent holder — important because a previous test
    # may have monkeypatched a fake classifier into _classifier.
    srv.intent_holder.cactus = None
    srv.intent_holder.model = None
    srv.intent_holder._classifier = None
    srv.intent_holder.load_error = None
    # And the whisper holder.
    srv.whisper_holder.cactus = None
    srv.whisper_holder.model = None
    srv.whisper_holder.weights_path = None
    srv.whisper_holder.load_error = None


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
            # Capability flags must be present so the client can pick
            # exactly one TTS source (server WAV or browser) without
            # racing. False here because the fixture forces HAS_SAY off.
            assert msg["tts_available"] is False
            assert msg["intent_loaded"] is False
            assert msg["whisper_loaded"] is False


def test_ws_ready_reports_tts_available_when_say_present(server, monkeypatch):
    """When `say` is present on the host, the ready frame says so so
    the client can suppress its own speechSynthesis and avoid the
    double-voice playback."""
    from app import main as srv
    monkeypatch.setattr(srv, "HAS_SAY", True)
    with TestClient(server["app"]) as client:
        with client.websocket_connect("/ws/coach") as ws:
            msg = ws.receive_json()
            assert msg["type"] == "ready"
            assert msg["tts_available"] is True


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
            # vl_1 default exercise has 4 phases (warmup → glide → counting → main_task)
            assert drill0["total"] >= 4
            assert drill0["stage"] == "warmup"
            # Richer context fields are now sent on every drill message.
            assert drill0["exercise_id"] == "vl_1"
            assert drill0["exercise_name"] == "Sustained Vowel Power"
            assert drill0["category_id"] == "voice_loudness"

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


def test_ws_completes_session_after_last_drill(server, monkeypatch):
    """Walking through all drills should end with session_done containing summary.

    Scoped to a single exercise (vl_1, 4 drills) via env var so the test
    stays fast and focused on the protocol — the full 69-drill walk is
    already covered structurally by test_content.TestAllDrills.
    """
    monkeypatch.setenv("VOICE_COACH_EXERCISE", "vl_1")
    import content
    total = len(content.default_drill_set())
    assert total == 4

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


def test_ws_intent_skip_dispatches_skip_command(server):
    """Intent message routed by the heuristic (no FunctionGemma loaded)
    must produce both an intent_result frame AND the same drill-loop
    behaviour as a manual `command:skip`."""
    with TestClient(server["app"]) as client:
        with client.websocket_connect("/ws/coach") as ws:
            ws.receive_json()  # ready
            ws.send_json({"type": "start_session"})
            d0 = ws.receive_json()
            ws.send_json({"type": "intent", "utterance": "skip this one"})
            res = ws.receive_json()
            assert res["type"] == "intent_result"
            assert res["action"] == "skip"
            assert res["source"] == "heuristic"
            assert res["intent_model_loaded"] is False
            assert res["confidence"] >= 0.55
            assert isinstance(res["latency_ms"], int)
            # Typed path: no Whisper, transcript echoes utterance.
            assert res["transcribe_source"] == "client"
            assert res["transcribe_latency_ms"] is None
            assert res["transcript"] == "skip this one"
            # Same drill-loop side effect as a manual skip.
            assert ws.receive_json()["type"] == "advance"
            d1 = ws.receive_json()
            assert d1["type"] == "drill"
            assert d1["position"] == d0["position"] + 1


def test_ws_intent_rest_ends_session(server):
    with TestClient(server["app"]) as client:
        with client.websocket_connect("/ws/coach") as ws:
            ws.receive_json()  # ready
            ws.send_json({"type": "start_session"})
            ws.receive_json()  # drill 0
            ws.send_json({"type": "intent", "utterance": "I'm tired"})
            res = ws.receive_json()
            assert res["action"] == "rest"
            done = ws.receive_json()
            assert done["type"] == "session_done"
            assert done["summary"]["rest_called"] is True


def test_ws_intent_repeat_resends_current_drill(server):
    with TestClient(server["app"]) as client:
        with client.websocket_connect("/ws/coach") as ws:
            ws.receive_json()  # ready
            ws.send_json({"type": "start_session"})
            d0 = ws.receive_json()
            ws.send_json({"type": "intent", "utterance": "say it again"})
            res = ws.receive_json()
            assert res["action"] == "repeat_prompt"
            d_again = ws.receive_json()
            assert d_again["type"] == "drill"
            assert d_again["position"] == d0["position"]


def test_ws_intent_unmatched_does_not_dispatch(server):
    """Off-topic utterance returns intent_result with action=none and
    leaves the drill state untouched. The next message back is the
    response to whatever the client does NEXT, not anything auto-fired."""
    with TestClient(server["app"]) as client:
        with client.websocket_connect("/ws/coach") as ws:
            ws.receive_json()  # ready
            ws.send_json({"type": "start_session"})
            d0 = ws.receive_json()
            ws.send_json({"type": "intent", "utterance": "the blue spot"})
            res = ws.receive_json()
            assert res["type"] == "intent_result"
            assert res["action"] == "none"
            # Verify nothing else fires by sending a real command and
            # checking we get its response next (not a stray drill).
            ws.send_json({"type": "command", "action": "skip"})
            assert ws.receive_json()["type"] == "advance"
            d1 = ws.receive_json()
            assert d1["position"] == d0["position"] + 1


def test_ws_intent_uses_funcgemma_when_loaded(server, monkeypatch):
    """When the IntentHolder has a classifier, it must be consulted
    (and its source must be reported)."""
    from app import main as srv
    import intent as intent_mod

    fake = MagicMock(spec=intent_mod.FunctionGemmaClassifier)
    fake.classify.return_value = intent_mod.IntentResult(
        intent=intent_mod.Intent.SKIP,
        confidence=0.99,
        utterance="next",
        source="functiongemma",
        latency_ms=42,
    )
    # Pretend FunctionGemma is loaded by overriding the holder for one test.
    monkeypatch.setattr(srv.intent_holder, "_classifier", fake)

    with TestClient(server["app"]) as client:
        with client.websocket_connect("/ws/coach") as ws:
            ws.receive_json()  # ready
            ws.send_json({"type": "start_session"})
            ws.receive_json()  # drill 0
            ws.send_json({"type": "intent", "utterance": "next"})
            res = ws.receive_json()
            assert res["type"] == "intent_result"
            assert res["source"] == "functiongemma"
            assert res["latency_ms"] == 42
            assert res["intent_model_loaded"] is True
            # Underlying classifier was invoked.
            fake.classify.assert_called_once_with("next")


def test_ws_intent_empty_utterance_returns_none(server):
    with TestClient(server["app"]) as client:
        with client.websocket_connect("/ws/coach") as ws:
            ws.receive_json()  # ready
            ws.send_json({"type": "start_session"})
            ws.receive_json()  # drill 0
            ws.send_json({"type": "intent", "utterance": ""})
            res = ws.receive_json()
            assert res["action"] == "none"


def test_health_reports_intent_holder_status(server):
    """Health endpoint surfaces the intent-router state alongside the
    coach model so the demo can show two independent on-device models."""
    with TestClient(server["app"]) as client:
        body = client.get("/health").json()
        assert "intent_loaded" in body
        assert body["intent_loaded"] is False  # not loaded in tests
        assert "intent_model_id" in body
        assert "functiongemma" in body["intent_model_id"]


def test_health_reports_whisper_holder_status(server):
    with TestClient(server["app"]) as client:
        body = client.get("/health").json()
        assert "whisper_loaded" in body
        assert body["whisper_loaded"] is False  # not loaded in tests
        assert "whisper" in body["whisper_model_id"]
        assert "tts_available" in body
        assert body["tts_available"] is False  # forced off in fixture


# ---- intent_audio: Cactus Whisper transcription path --------------------


def test_ws_intent_audio_when_whisper_unloaded_returns_error(server):
    """Audio command before Whisper is ready must emit a typed error
    so the UI can suggest typing the command instead. Drill state must
    not change."""
    with TestClient(server["app"]) as client:
        with client.websocket_connect("/ws/coach") as ws:
            ws.receive_json()  # ready
            ws.send_json({"type": "start_session"})
            d0 = ws.receive_json()
            ws.send_json({
                "type": "intent_audio",
                "pcm_b64": _b64_pcm(1.0),
                "sample_rate": 16000,
            })
            err = ws.receive_json()
            assert err["type"] == "error"
            assert "Whisper" in err["message"] or "whisper" in err["message"]
            # Loop continues — explicit skip still works.
            ws.send_json({"type": "command", "action": "skip"})
            assert ws.receive_json()["type"] == "advance"
            assert ws.receive_json()["position"] == d0["position"] + 1


def test_ws_intent_audio_transcribes_and_routes(server):
    """When Whisper is loaded, intent_audio runs through the holder's
    transcribe(), then through the existing classifier, then dispatches
    the matched action. The result chip carries both latencies."""
    from app import main as srv

    fake_holder_transcribe = MagicMock(return_value=("skip this one", 137))
    # Mark the holder as loaded so the WS doesn't bail out.
    srv.whisper_holder.model = MagicMock(name="whisper_handle")
    srv.whisper_holder.cactus = MagicMock()
    srv.whisper_holder.transcribe = fake_holder_transcribe

    with TestClient(server["app"]) as client:
        with client.websocket_connect("/ws/coach") as ws:
            ws.receive_json()  # ready
            ws.send_json({"type": "start_session"})
            d0 = ws.receive_json()
            ws.send_json({
                "type": "intent_audio",
                "pcm_b64": _b64_pcm(1.0),
                "sample_rate": 16000,
            })
            res = ws.receive_json()
            assert res["type"] == "intent_result"
            assert res["action"] == "skip"
            assert res["transcribe_source"] == "whisper"
            assert res["transcribe_latency_ms"] == 137
            assert res["transcript"] == "skip this one"
            # Underlying transcribe was called once with the raw PCM.
            assert fake_holder_transcribe.call_count == 1
            (sent_pcm,), _ = fake_holder_transcribe.call_args
            assert len(sent_pcm) == 32000
            # Drill-loop side effect.
            assert ws.receive_json()["type"] == "advance"
            assert ws.receive_json()["position"] == d0["position"] + 1


def test_ws_intent_audio_wrong_sample_rate_errors(server):
    from app import main as srv
    srv.whisper_holder.model = MagicMock()
    srv.whisper_holder.cactus = MagicMock()
    srv.whisper_holder.transcribe = MagicMock(return_value=("", 0))

    with TestClient(server["app"]) as client:
        with client.websocket_connect("/ws/coach") as ws:
            ws.receive_json()  # ready
            ws.send_json({"type": "start_session"})
            ws.receive_json()  # drill 0
            ws.send_json({
                "type": "intent_audio",
                "pcm_b64": _b64_pcm(1.0),
                "sample_rate": 22050,
            })
            err = ws.receive_json()
            assert err["type"] == "error"
            assert "sample_rate" in err["message"]


def test_ws_intent_audio_empty_pcm_errors(server):
    from app import main as srv
    srv.whisper_holder.model = MagicMock()
    srv.whisper_holder.cactus = MagicMock()
    srv.whisper_holder.transcribe = MagicMock()

    with TestClient(server["app"]) as client:
        with client.websocket_connect("/ws/coach") as ws:
            ws.receive_json()  # ready
            ws.send_json({"type": "start_session"})
            ws.receive_json()  # drill 0
            ws.send_json({
                "type": "intent_audio",
                "pcm_b64": "",
                "sample_rate": 16000,
            })
            err = ws.receive_json()
            assert err["type"] == "error"
            assert "empty" in err["message"]
            srv.whisper_holder.transcribe.assert_not_called()


def test_ws_intent_audio_transcribe_failure_returns_error(server):
    from app import main as srv
    srv.whisper_holder.model = MagicMock()
    srv.whisper_holder.cactus = MagicMock()
    srv.whisper_holder.transcribe = MagicMock(
        side_effect=RuntimeError("whisper boom")
    )

    with TestClient(server["app"]) as client:
        with client.websocket_connect("/ws/coach") as ws:
            ws.receive_json()  # ready
            ws.send_json({"type": "start_session"})
            ws.receive_json()  # drill 0
            ws.send_json({
                "type": "intent_audio",
                "pcm_b64": _b64_pcm(1.0),
                "sample_rate": 16000,
            })
            err = ws.receive_json()
            assert err["type"] == "error"
            assert "whisper" in err["message"].lower()


def test_ws_intent_audio_invalid_b64_errors(server):
    from app import main as srv
    srv.whisper_holder.model = MagicMock()
    srv.whisper_holder.cactus = MagicMock()
    srv.whisper_holder.transcribe = MagicMock()

    with TestClient(server["app"]) as client:
        with client.websocket_connect("/ws/coach") as ws:
            ws.receive_json()  # ready
            ws.send_json({"type": "start_session"})
            ws.receive_json()  # drill 0
            ws.send_json({
                "type": "intent_audio",
                "pcm_b64": "not~~base64@@",
                "sample_rate": 16000,
            })
            err = ws.receive_json()
            assert err["type"] == "error"


# ---- audio_reply: server-side macOS `say` TTS ----------------------------


def test_ws_audio_reply_emitted_when_tts_available(server, monkeypatch):
    """When `_say_to_wav` returns bytes, the server must emit an
    audio_reply frame right after the coach frame.

    With server-side TTS enabled the server ALSO renders drill prompts
    (so the whole session uses one consistent voice instead of
    alternating browser+say). That means `_say_to_wav` is called
    multiple times: at least once per drill prompt sent + once per
    coach reply. The test asserts the coach-reply call happened."""
    from app import main as srv
    rendered_wav = b"RIFF" + b"\x00" * 100  # plausible-looking bytes
    say_calls = []

    def fake_say(text):
        say_calls.append(text)
        return rendered_wav

    monkeypatch.setattr(srv, "HAS_SAY", True)
    monkeypatch.setattr(srv, "_say_to_wav", fake_say)

    with TestClient(server["app"]) as client:
        with client.websocket_connect("/ws/coach") as ws:
            ws.receive_json()  # ready
            ws.send_json({"type": "start_session"})
            drill0 = ws.receive_json()
            assert drill0["type"] == "drill"
            # First drill carries its own pre-rendered prompt wav so
            # the client doesn't have to use browser TTS.
            assert "prompt_wav_b64" in drill0
            ws.send_json({
                "type": "audio",
                "pcm_b64": _b64_pcm(1.0),
                "sample_rate": 16000,
            })
            ws.receive_json()  # metrics
            ws.receive_json()  # thinking
            coach_msg = ws.receive_json()
            assert coach_msg["type"] == "coach"
            audio = ws.receive_json()
            assert audio["type"] == "audio_reply"
            assert audio["source"] == "macos_say"
            # WAV is base64-encoded; round-trip back to the bytes the
            # fake renderer produced.
            decoded = base64.b64decode(audio["wav_b64"])
            assert decoded == rendered_wav
            # `say` was handed the joined ack + feedback line at some
            # point — alongside the drill prompts.
            assert any("Nice" in c for c in say_calls), say_calls


def test_ws_drill_prompt_wav_attached_when_tts_available(server, monkeypatch):
    """Drill frames must carry `prompt_wav_b64` so the client speaks
    drill prompts via the SAME engine as coach replies. Without this,
    drill prompts go through browser speechSynthesis and coach replies
    go through `say`, alternating two voices and giving the patient the
    impression that two separate TTS engines are talking."""
    from app import main as srv
    rendered_wav = b"DRILL_WAV"
    monkeypatch.setattr(srv, "HAS_SAY", True)
    monkeypatch.setattr(srv, "_say_to_wav", lambda _t: rendered_wav)

    with TestClient(server["app"]) as client:
        with client.websocket_connect("/ws/coach") as ws:
            ws.receive_json()  # ready
            ws.send_json({"type": "start_session"})
            drill0 = ws.receive_json()
            assert drill0["type"] == "drill"
            assert drill0["prompt_wav_b64"] == base64.b64encode(rendered_wav).decode("ascii")
            # Skip → next drill must also carry a wav.
            ws.send_json({"type": "command", "action": "skip"})
            assert ws.receive_json()["type"] == "advance"
            drill1 = ws.receive_json()
            assert drill1["type"] == "drill"
            assert "prompt_wav_b64" in drill1


def test_ws_drill_omits_prompt_wav_when_tts_unavailable(server, monkeypatch):
    """Without `say`, the drill frame must NOT include `prompt_wav_b64`
    so the client knows to use its own browser TTS for the prompt."""
    from app import main as srv
    monkeypatch.setattr(srv, "HAS_SAY", False)

    with TestClient(server["app"]) as client:
        with client.websocket_connect("/ws/coach") as ws:
            ws.receive_json()  # ready
            ws.send_json({"type": "start_session"})
            drill0 = ws.receive_json()
            assert drill0["type"] == "drill"
            assert "prompt_wav_b64" not in drill0


def test_ws_drill_omits_prompt_wav_when_say_renders_nothing(server, monkeypatch):
    """If `say` is present but the render fails, drill frame still ships
    (just without the wav). Client falls back to browser TTS for the
    prompt — better than dropping the prompt entirely."""
    from app import main as srv
    monkeypatch.setattr(srv, "HAS_SAY", True)
    monkeypatch.setattr(srv, "_say_to_wav", lambda _t: None)

    with TestClient(server["app"]) as client:
        with client.websocket_connect("/ws/coach") as ws:
            ws.receive_json()  # ready
            ws.send_json({"type": "start_session"})
            drill0 = ws.receive_json()
            assert drill0["type"] == "drill"
            assert "prompt_wav_b64" not in drill0


def test_build_drill_tts_handles_instruction_only_phase(server):
    """Mirrors buildDrillTTS in audio.ts. When note is empty the
    prompt IS the full instruction — speak it as-is (no 'Now: ...')."""
    from app import main as srv
    assert srv._build_drill_tts("Eee.", "") == "Eee."
    assert srv._build_drill_tts("Eee.", "   ") == "Eee."


def test_build_drill_tts_combines_note_then_prompt(server):
    """When both note (cue) and prompt (utterance) are present, the
    rendered line should drop trailing punctuation from the cue and
    join with 'Now:' — same shape as the frontend builder. The trailing
    '.' is appended unconditionally to match buildDrillTTS in audio.ts
    (which produces the same double-dot when the prompt already ended
    in punctuation; harmless to TTS engines)."""
    from app import main as srv
    out = srv._build_drill_tts("Hello there", "Read the phrase aloud.")
    assert out == "Read the phrase aloud. Now: Hello there."
    # Cue with trailing punctuation gets cleaned before the period.
    out2 = srv._build_drill_tts("Hi", "Read the phrase aloud!! ")
    assert out2 == "Read the phrase aloud. Now: Hi."


def test_ws_no_audio_reply_when_say_returns_none(server, monkeypatch):
    """When `_say_to_wav` returns None (say missing or failed), the
    server must NOT emit an audio_reply — the next message should be
    the normal advance/drill cue."""
    from app import main as srv
    monkeypatch.setattr(srv, "HAS_SAY", True)
    monkeypatch.setattr(srv, "_say_to_wav", lambda _t: None)

    with TestClient(server["app"]) as client:
        with client.websocket_connect("/ws/coach") as ws:
            ws.receive_json()  # ready
            ws.send_json({"type": "start_session"})
            ws.receive_json()  # drill 0
            ws.send_json({
                "type": "audio",
                "pcm_b64": _b64_pcm(1.0),
                "sample_rate": 16000,
            })
            ws.receive_json()  # metrics
            ws.receive_json()  # thinking
            assert ws.receive_json()["type"] == "coach"
            # Default canned reply has next_action=advance — that's
            # what we should see next, with no audio_reply in between.
            assert ws.receive_json()["type"] == "advance"


def test_say_to_wav_returns_none_without_say_binary(server, monkeypatch):
    """Direct probe of the helper. Pretend `say` is missing — it
    should return None without raising or shelling out."""
    from app import main as srv
    monkeypatch.setattr(srv, "HAS_SAY", False)
    assert srv._say_to_wav("hello world") is None


def test_say_to_wav_returns_none_for_empty_text(server, monkeypatch):
    from app import main as srv
    monkeypatch.setattr(srv, "HAS_SAY", True)
    # Empty input must short-circuit BEFORE shelling out.
    assert srv._say_to_wav("") is None
    assert srv._say_to_wav("   ") is None


def test_say_to_wav_runs_subprocess_and_returns_bytes(server, monkeypatch):
    """Cover the happy subprocess path. We mock subprocess.run to
    pretend `say` ran and wrote the wav file we expected at the
    --output path it received."""
    from app import main as srv
    monkeypatch.setattr(srv, "HAS_SAY", True)
    monkeypatch.setattr(srv, "SAY_VOICE", "Samantha")  # exercise -v branch

    expected_bytes = b"RIFF\x00\x00\x00\x00WAVEfake-pcm-data"
    captured_args: list[list[str]] = []

    def fake_run(args, **_kw):
        captured_args.append(args)
        out_path = Path(args[args.index("-o") + 1])
        out_path.write_bytes(expected_bytes)
        return MagicMock(returncode=0, stderr=b"")

    monkeypatch.setattr("subprocess.run", fake_run)
    result = srv._say_to_wav("hello world")
    assert result == expected_bytes
    # `say` was invoked with the data-format flag, optional voice,
    # an output path, then the text after `--`.
    assert len(captured_args) == 1
    args = captured_args[0]
    assert args[0] == "say"
    assert "--data-format=LEI16@22050" in args
    assert "-v" in args and "Samantha" in args
    assert "--" in args
    assert "hello world" in args


def test_say_to_wav_returns_none_when_subprocess_fails(server, monkeypatch):
    from app import main as srv
    monkeypatch.setattr(srv, "HAS_SAY", True)
    monkeypatch.setattr(
        "subprocess.run",
        lambda *_a, **_kw: MagicMock(returncode=1, stderr=b"boom"),
    )
    assert srv._say_to_wav("hello") is None


def test_say_to_wav_returns_none_when_subprocess_times_out(server, monkeypatch):
    from app import main as srv
    import subprocess as _sp
    monkeypatch.setattr(srv, "HAS_SAY", True)
    def boom(*_a, **_kw):
        raise _sp.TimeoutExpired(cmd="say", timeout=10)
    monkeypatch.setattr("subprocess.run", boom)
    assert srv._say_to_wav("hello") is None


def test_say_to_wav_returns_none_when_output_file_empty(server, monkeypatch):
    """If `say` exits 0 but writes a zero-byte file, we treat that as
    a failure rather than shipping an empty WAV the browser will
    reject."""
    from app import main as srv
    monkeypatch.setattr(srv, "HAS_SAY", True)

    def fake_run(args, **_kw):
        out_path = Path(args[args.index("-o") + 1])
        out_path.write_bytes(b"")
        return MagicMock(returncode=0, stderr=b"")

    monkeypatch.setattr("subprocess.run", fake_run)
    assert srv._say_to_wav("hello") is None


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
