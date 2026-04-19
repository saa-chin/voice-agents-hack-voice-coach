# Parkinson's Voice MVP

A simple full-stack starter repo for recording one voice sample in the browser and analyzing it with a FastAPI backend.

## What it does

- Records audio in the browser
- Sends the sample to a FastAPI API
- Extracts simple voice markers that are commonly discussed in Parkinson's speech analysis:
  - duration
  - mean pitch
  - pitch variation
  - loudness
  - jitter
  - shimmer
  - HNR
- Returns a lightweight score and coaching feedback

## Important note

This project is for **speech practice and monitoring only**. It is **not** a medical diagnosis tool.

---

## Repo structure

```text
parkinsons-voice-mvp/
  backend/
    app/main.py
    requirements.txt
  frontend/
    app/
    components/
    package.json
  README.md
```

---

## 1. Run the backend

### Requirements

- Python 3.11+
- macOS / Linux recommended

### Setup

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

Backend should now be running at:

```text
http://localhost:8000
```

Health check:

```bash
curl http://localhost:8000/health
```

---

## 2. Run the frontend

### Requirements

- Node.js 18+

### Setup

```bash
cd frontend
npm install
npm run dev
```

Frontend should now be running at:

```text
http://localhost:3000
```

---

## 3. Configure frontend → backend URL

By default the frontend calls:

```text
http://localhost:8000
```

If you want to change that, create this file:

```bash
frontend/.env.local
```

With:

```env
NEXT_PUBLIC_API_BASE_URL=http://localhost:8000
```

---

## 4. How to demo it

1. Open the frontend in a browser.
2. Allow microphone access.
3. Read the reference phrase aloud:

```text
The quick brown fox jumps over the lazy dog
```

4. Stop recording.
5. Click **Analyze recording**.
6. Review the score, feedback, and raw voice metrics.

---

## Suggested next improvements

### Better clinical usefulness
- add repeated vowel tasks like sustained "ahhh"
- add separate exercise modes for loudness, prosody, and articulation
- track sessions over time in a database

### Better accuracy
- add Whisper or faster-whisper for transcription
- compare spoken text vs reference phrase
- store baseline samples per patient
- add per-user longitudinal charts

### Better product demo
- add waveform display
- add severity trend over time
- add coach voice prompts with ElevenLabs
- add authentication and patient history

---

## API

### `GET /health`
Returns:

```json
{ "status": "ok" }
```

### `POST /analyze`
Form-data:
- `file`: audio upload

Returns JSON like:

```json
{
  "referencePhrase": "The quick brown fox jumps over the lazy dog",
  "overallScore": 78.4,
  "metrics": {
    "durationSeconds": 3.61,
    "meanPitchHz": 172.44,
    "pitchVariationHz": 28.51,
    "loudnessRms": 0.0543,
    "loudnessStd": 0.0204,
    "speechRateProxy": 1.83,
    "jitterLocal": 0.0118,
    "shimmerLocal": 0.0551,
    "hnr": 16.37
  },
  "scores": {
    "loudness": 90.5,
    "pitchVariation": 81.4,
    "duration": 90.2,
    "stability": 63.0
  },
  "feedback": [
    "Try adding more pitch variation so the voice sounds less monotone."
  ],
  "disclaimer": "This MVP is for speech practice and monitoring only. It is not a medical diagnosis tool."
}
```

---

## Hackathon framing

A good one-line pitch for this build:

> An AI-assisted speech practice tool for Parkinson's patients that records one phrase, analyzes voice stability and expressiveness, and gives simple coaching feedback.
