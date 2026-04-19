from __future__ import annotations

import io
import math
import os
import tempfile
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import parselmouth
import soundfile as sf
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Parkinson's Voice MVP API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

REFERENCE_PHRASE = "The quick brown fox jumps over the lazy dog"
TARGET_DURATION_SECONDS = 4.0
TARGET_PITCH_STD_HZ = 35.0
TARGET_RMS = 0.06


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        numeric = float(value)
        if math.isnan(numeric) or math.isinf(numeric):
            return default
        return numeric
    except Exception:
        return default


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(value, maximum))


def normalize_score(value: float, target: float, tolerance: float) -> float:
    if target <= 0:
        return 0.0
    delta = abs(value - target)
    score = 100.0 * (1.0 - delta / max(tolerance, 1e-6))
    return clamp(score, 0.0, 100.0)


def compute_speech_rate_proxy(y: np.ndarray, sr: int, hop_length: int = 512) -> float:
    frame_length = 2048
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    threshold = max(np.percentile(rms, 60), 0.01)
    active = rms > threshold
    segments = 0
    prev = False
    for value in active:
        if value and not prev:
            segments += 1
        prev = bool(value)

    duration = librosa.get_duration(y=y, sr=sr)
    if duration <= 0:
        return 0.0

    syllable_like_units = max(segments, 1)
    return safe_float(syllable_like_units / duration)


def decode_audio_bytes(audio_bytes: bytes, filename: str | None = None) -> tuple[np.ndarray, int]:
    try:
        audio_buffer = io.BytesIO(audio_bytes)
        y, sr = sf.read(audio_buffer, dtype="float32", always_2d=False)
        return y, int(sr)
    except Exception as soundfile_exc:
        temp_path: str | None = None
        try:
            suffix = Path(filename or "upload.wav").suffix or ".wav"
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp_file:
                temp_file.write(audio_bytes)
                temp_path = temp_file.name

            y, sr = librosa.load(temp_path, sr=None, mono=False)
            return y, int(sr)
        except Exception as fallback_exc:
            raise HTTPException(
                status_code=400,
                detail=f"Could not decode audio file: soundfile={soundfile_exc}; fallback={fallback_exc}",
            ) from fallback_exc
        finally:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)


def to_mono(y: np.ndarray) -> np.ndarray:
    if isinstance(y, np.ndarray) and y.ndim > 1:
        if y.shape[0] <= y.shape[1]:
            return np.mean(y, axis=0)
        return np.mean(y, axis=1)
    return y


def analyze_audio_bytes(audio_bytes: bytes, filename: str | None = None) -> dict[str, Any]:
    y, sr = decode_audio_bytes(audio_bytes, filename=filename)

    if y is None or len(y) == 0:
        raise HTTPException(status_code=400, detail="Uploaded audio is empty.")

    y = to_mono(y)

    y = librosa.util.normalize(y)
    duration = safe_float(librosa.get_duration(y=y, sr=sr))

    snd = parselmouth.Sound(y, sampling_frequency=sr)
    pitch = snd.to_pitch(time_step=0.01, pitch_floor=75, pitch_ceiling=400)
    pitch_values = pitch.selected_array["frequency"]
    voiced_pitch = pitch_values[pitch_values > 0]

    mean_pitch = safe_float(np.mean(voiced_pitch), 0.0) if voiced_pitch.size else 0.0
    pitch_std = safe_float(np.std(voiced_pitch), 0.0) if voiced_pitch.size else 0.0

    point_process = parselmouth.praat.call(snd, "To PointProcess (periodic, cc)", 75, 400)
    try:
        jitter_local = safe_float(
            parselmouth.praat.call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3),
            0.0,
        )
    except Exception:
        jitter_local = 0.0

    try:
        shimmer_local = safe_float(
            parselmouth.praat.call(
                [snd, point_process],
                "Get shimmer (local)",
                0,
                0,
                0.0001,
                0.02,
                1.3,
                1.6,
            ),
            0.0,
        )
    except Exception:
        shimmer_local = 0.0

    try:
        harmonicity = snd.to_harmonicity_cc()
        hnr = safe_float(parselmouth.praat.call(harmonicity, "Get mean", 0, 0), 0.0)
    except Exception:
        hnr = 0.0

    rms = librosa.feature.rms(y=y)[0]
    loudness_mean = safe_float(np.mean(rms), 0.0)
    loudness_std = safe_float(np.std(rms), 0.0)
    speech_rate_proxy = compute_speech_rate_proxy(y, sr)

    loudness_score = normalize_score(loudness_mean, TARGET_RMS, tolerance=TARGET_RMS)
    pitch_variation_score = normalize_score(pitch_std, TARGET_PITCH_STD_HZ, tolerance=TARGET_PITCH_STD_HZ)
    duration_score = normalize_score(duration, TARGET_DURATION_SECONDS, tolerance=TARGET_DURATION_SECONDS)
    stability_penalty = clamp((jitter_local * 1500.0) + (shimmer_local * 350.0), 0.0, 100.0)
    stability_score = 100.0 - stability_penalty

    overall_score = round(
        0.30 * loudness_score
        + 0.25 * pitch_variation_score
        + 0.20 * duration_score
        + 0.25 * stability_score,
        1,
    )

    feedback: list[str] = []
    if loudness_mean < 0.04:
        feedback.append("Speak a little louder to improve vocal intensity.")
    if pitch_std < 20:
        feedback.append("Try adding more pitch variation so the voice sounds less monotone.")
    if duration < 2.5:
        feedback.append("Slow down slightly and stretch each word a bit more clearly.")
    if jitter_local > 0.02 or shimmer_local > 0.08:
        feedback.append("Your voice stability looks reduced. Try taking a breath and repeating the phrase steadily.")
    if not feedback:
        feedback.append("Nice job. The sample looks reasonably strong for this simple screening exercise.")

    return {
        "referencePhrase": REFERENCE_PHRASE,
        "overallScore": overall_score,
        "metrics": {
            "durationSeconds": round(duration, 2),
            "meanPitchHz": round(mean_pitch, 2),
            "pitchVariationHz": round(pitch_std, 2),
            "loudnessRms": round(loudness_mean, 4),
            "loudnessStd": round(loudness_std, 4),
            "speechRateProxy": round(speech_rate_proxy, 2),
            "jitterLocal": round(jitter_local, 4),
            "shimmerLocal": round(shimmer_local, 4),
            "hnr": round(hnr, 2),
        },
        "scores": {
            "loudness": round(loudness_score, 1),
            "pitchVariation": round(pitch_variation_score, 1),
            "duration": round(duration_score, 1),
            "stability": round(stability_score, 1),
        },
        "feedback": feedback,
        "disclaimer": "This MVP is for speech practice and monitoring only. It is not a medical diagnosis tool.",
    }


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)) -> dict[str, Any]:
    audio_bytes = await file.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Uploaded audio is empty.")

    return analyze_audio_bytes(audio_bytes, filename=file.filename)
