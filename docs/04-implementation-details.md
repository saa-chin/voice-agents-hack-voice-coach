# Implementation Details

This document is the engineering blueprint for the initial MVP. It maps every component of the architecture to a concrete module, a concrete library, an estimated cost, and an owner.

## 1. Repository layout

```
voice-agents-hack-voice-coach/
├── docs/                       ← This documentation set
├── mobile/                     ← React Native 0.85 app (iOS + Android)
│   ├── App.tsx                 ← Navigation root
│   ├── src/
│   │   ├── screens/            ← Home, Drill, Summary, History
│   │   ├── components/         ← LoudnessMeter, PitchMeter, TrendChart
│   │   ├── audio/              ← Native bridge: capture, DSP, ring buffer
│   │   ├── inference/          ← Cactus SDK wrapper, prompt builders
│   │   ├── storage/            ← SQLite schema + repository layer
│   │   ├── drills/             ← Drill content + protocol logic
│   │   └── theme/              ← Accessibility-first design tokens
│   ├── ios/                    ← Cactus xcframework + audio engine
│   └── android/                ← Cactus AAR + AudioRecord bridge
└── assets/                     ← Models (downloaded on first launch), images
```

## 2. Component-to-technology map

| Component | Technology | Notes |
| --- | --- | --- |
| App shell | React Native 0.85, TypeScript 5.8 | Already scaffolded |
| Navigation | `@react-navigation/native` (stack) | Three screens: Home → Drill → Summary |
| Audio capture (iOS) | `AVAudioEngine` via native module | 16 kHz mono PCM, 50 ms frames |
| Audio capture (Android) | `AudioRecord` via native module | Same format for parity |
| DSP — loudness | RMS over 50 ms window → dBFS → calibrated dB SPL estimate | Pure C/Swift/Kotlin, no model |
| DSP — pitch | YIN algorithm, 25 ms hop | Returns F0 in Hz, voicing flag |
| DSP — pace | Energy-based syllable nucleus detector | Rolling syllables/sec |
| On-device router | FunctionGemma 270M via `cactus-react-native` | Structured tool calls |
| On-device coach | Gemma 4 E2B (INT4) via `cactus-react-native` | Native audio input |
| TTS output | `AVSpeechSynthesizer` (iOS) / `TextToSpeech` (Android) | Free, fast, decent quality |
| Local storage | `react-native-quick-sqlite` | Sync, fast, works on Hermes |
| Charts | `victory-native` or `react-native-svg` hand-rolled | Simple 7-day line |
| Cloud (opt-in) | `google-genai` SDK → Gemini 2.5 | Text only, sanitized |
| State | Zustand | Tiny, no boilerplate |
| Logging | `react-native-logs` to a local rolling file | No telemetry sent anywhere |

## 3. Data model (SQLite)

```mermaid
erDiagram
    SESSION ||--o{ TURN : contains
    SESSION {
        uuid id PK
        datetime started_at
        datetime ended_at
        text drill_set
        int turns_completed
        real avg_loudness_db
        real avg_pitch_range_st
        real avg_pace_sps
    }
    TURN {
        uuid id PK
        uuid session_id FK
        int turn_index
        text prompt_text
        text drill_type
        real target_loudness_db
        real achieved_loudness_db
        real pitch_range_st
        real pace_sps
        text coach_feedback
        text next_action
        datetime created_at
    }
    SETTINGS {
        text key PK
        text value
    }
```

No user table. No account. No PII beyond what the patient types into a free-text "name" field stored locally.

## 4. The inference contract (Gemma 4 prompt schema)

Every coaching turn sends Gemma 4 a small system prompt, the audio buffer, and the DSP-derived numerics. The model is instructed to return strict JSON.

```text
SYSTEM:
You are a warm, patient speech coach for adults with motor speech disorders
(Parkinson's, post-stroke dysarthria). The user just attempted to say:
"<prompt>".
Their target loudness was <target> dB SPL. They reached <achieved> dB SPL.
Their pitch range was <range> semitones. Their pace was <sps> syllables/sec.
You will hear their attempt as audio.

Listen for: vocal effort, breath support, pitch monotony, trailing off,
rushed pace, slurred articulation. Be specific but never discouraging.

Respond with JSON only:
{
  "ack": "<one short warm acknowledgement, 3–6 words>",
  "feedback": "<one specific, actionable cue, 8–18 words>",
  "next_action": "retry" | "advance" | "rest",
  "metrics_observed": {
    "loudness_ok": bool,
    "pitch_range_ok": bool,
    "pace_ok": bool,
    "articulation_ok": bool
  }
}
```

The structured output is what makes the system reliable: the UI advances or retries based on `next_action`, the chart updates from `metrics_observed`, and only `ack + feedback` are spoken.

## 5. Drill protocol (encoded in `src/drills/`)

```mermaid
stateDiagram-v2
    [*] --> Warmup
    Warmup --> WarmupAttempt
    WarmupAttempt --> Warmup: retry (n<3)
    WarmupAttempt --> Phrases: advance
    Phrases --> PhraseAttempt
    PhraseAttempt --> Phrases: retry (n<2)
    PhraseAttempt --> NextPhrase: advance
    NextPhrase --> Phrases: more phrases
    NextPhrase --> Conversation: phrases done
    Conversation --> ConvoTurn
    ConvoTurn --> Conversation: continue (t<60s)
    ConvoTurn --> Summary: time up
    Summary --> [*]
    Warmup --> Rest: user requests rest
    Phrases --> Rest: user requests rest
    Conversation --> Rest: user requests rest
    Rest --> Summary
```

## 6. Task breakdown (initial MVP)

Tasks are sized so a small team can execute in parallel. **P0** is required for the first usable end-to-end build. **P1** is highly desirable. **P2** is stretch.

> Status legend: ☐ not started · ◐ in progress · ☑ done. Edit the glyph in place to update.

| # | Status | Task | Owner track | Priority | Est. time | Depends on |
|---|---|---|---|---|---|---|
| T01 | ☐ | Add navigation, theme tokens, screen shells (Home, Drill, Summary, History) | Frontend | P0 | 1.5h | — |
| T02 | ☐ | Install and configure `cactus-react-native`, run `pod install`, verify import on device | Native | P0 | 2h | — |
| T03 | ☐ | Download Gemma 4 E2B and FunctionGemma 270M weights, bundle or first-launch download | Native | P0 | 1h | T02 |
| T04 | ☐ | iOS audio capture native module (`AVAudioEngine` → 16 kHz PCM event stream to JS) | Native | P0 | 3h | — |
| T05 | ☐ | Android audio capture native module (`AudioRecord` parity) | Native | P1 | 3h | — |
| T06 | ☐ | DSP module: RMS loudness, dBFS → dB SPL calibration helper | Native | P0 | 2h | T04 |
| T07 | ☐ | DSP module: YIN pitch tracker, voicing flag, semitone range over window | Native | P1 | 3h | T04 |
| T08 | ☐ | DSP module: syllable-nucleus pace estimator | Native | P2 | 3h | T04 |
| T09 | ☐ | `LoudnessMeter` React component (animated bar, target line, accessible) | Frontend | P0 | 1.5h | T06 |
| T10 | ☐ | `PitchMeter` React component | Frontend | P1 | 1h | T07 |
| T11 | ☐ | Drill protocol state machine in `src/drills/` (warm-up → phrases → convo → summary) | Frontend | P0 | 2h | T01 |
| T12 | ☐ | Phrase content set: 20 functional phrases, 8 warm-up vowels, 10 conversation prompts | Content | P0 | 1h | — |
| T13 | ☐ | Cactus SDK wrapper: load model, pass audio buffer + system prompt, parse JSON response | Inference | P0 | 2h | T02, T03 |
| T14 | ☐ | Prompt builder: assemble system prompt with drill context and DSP metrics | Inference | P0 | 1h | T13 |
| T15 | ☐ | TTS playback wrapper (iOS + Android), barge-in handling | Frontend | P0 | 1h | — |
| T16 | ☐ | SQLite schema migration, repository functions for `session`, `turn`, `settings` | Storage | P0 | 1.5h | — |
| T17 | ☐ | Wire drill state machine to Cactus calls and SQLite writes | Inference | P0 | 2h | T11, T13, T16 |
| T18 | ☐ | Session summary screen: 3 numbers + 1 encouragement + 7-day mini-chart | Frontend | P0 | 2h | T16 |
| T19 | ☐ | History screen: list of past sessions, tap to view detail | Frontend | P1 | 1.5h | T16 |
| T20 | ☐ | FunctionGemma router for "retry / rest / repeat / done" intents from voice | Inference | P1 | 2h | T13 |
| T21 | ☐ | Accessibility pass: large text mode, high contrast, hit-target sizes, screen reader labels | Frontend | P0 | 1.5h | T01 |
| T22 | ☐ | Empty / first-run / model-downloading states with a clear progress UI | Frontend | P0 | 1h | T03 |
| T23 | ☐ | Cloud (opt-in) weekly PDF report via Gemini — sanitized metrics only | Cloud | P2 | 3h | T16 |
| T24 | ☐ | Cloud (opt-in) personalized phrase generator via Gemini | Cloud | P2 | 2h | T12 |
| T25 | ☐ | App icon, splash screen, Home-screen polish | Frontend | P0 | 1h | T01 |

**P0 total: ~26 hours of focused engineering work, parallelizable across two people.**

### Progress tracker

Quick checklist view, grouped by priority. Tick boxes as work lands.

**P0 — required for first usable end-to-end build (16 tasks)**

- [ ] T01 — Navigation, theme tokens, screen shells (Home, Drill, Summary, History)
- [ ] T02 — Install and configure `cactus-react-native`, verify on device
- [ ] T03 — Download Gemma 4 E2B + FunctionGemma 270M weights
- [ ] T04 — iOS audio capture native module (`AVAudioEngine`)
- [ ] T06 — DSP: RMS loudness + dBFS → dB SPL calibration
- [ ] T09 — `LoudnessMeter` React component
- [ ] T11 — Drill protocol state machine
- [ ] T12 — Phrase content set (warm-ups, phrases, conversation prompts)
- [ ] T13 — Cactus SDK wrapper (load model, audio + prompt, parse JSON)
- [ ] T14 — Prompt builder with drill context + DSP metrics
- [ ] T15 — TTS playback wrapper (iOS + Android), barge-in
- [ ] T16 — SQLite schema + repository for `session`, `turn`, `settings`
- [ ] T17 — Wire drill state machine to Cactus calls and SQLite
- [ ] T18 — Session summary screen (3 numbers + encouragement + 7-day chart)
- [ ] T21 — Accessibility pass (large text, contrast, hit targets, screen reader)
- [ ] T22 — First-run / model-downloading progress UI
- [ ] T25 — App icon, splash screen, Home-screen polish

**P1 — highly desirable (5 tasks)**

- [ ] T05 — Android audio capture native module (`AudioRecord`)
- [ ] T07 — DSP: YIN pitch tracker + semitone range
- [ ] T10 — `PitchMeter` React component
- [ ] T19 — History screen (list of past sessions, detail view)
- [ ] T20 — FunctionGemma router for voice intents (retry / rest / repeat / done)

**P2 — stretch (3 tasks)**

- [ ] T08 — DSP: syllable-nucleus pace estimator
- [ ] T23 — Cloud weekly PDF report via Gemini (opt-in)
- [ ] T24 — Cloud personalized phrase generator via Gemini (opt-in)

## 7. Supporting work tracks (clinical, content, design, ops)

The engineering tasks above are necessary but not sufficient. The app's value depends on clinically defensible targets, well-written prompts, plain-language copy, and real conversations with patients, caregivers, and SLPs. These tracks run in parallel with engineering and feed into it.

| # | Status | Task | Track | Priority | Est. time | Feeds into |
|---|---|---|---|---|---|---|
| R01 | ☐ | Literature review on motor speech therapy (LSVT LOUD, SPEAK OUT!, evidence on loudness-led approaches) → write a 1-page summary with citations | Clinical | P0 | 4h | T11, T14 |
| R02 | ☐ | Defensible default targets: loudness in dB SPL, pitch range in semitones, pace in syllables/sec — by drill type and severity tier | Clinical | P0 | 2h | T06, T14, T17 |
| R03 | ☐ | Microphone/SPL calibration table: measure 3+ phones at 30 cm against a reference SPL meter, produce offset constants | Research | P0 | 3h | T06 |
| R04 | ☐ | Recruit 2–3 SLP advisors for a 30-min protocol review; capture written feedback | Clinical | P0 | 4h (calendar: 1 week) | T11, R10 |
| R05 | ☐ | Recruit 5 participants (PD and/or post-stroke) for 15-min usability sessions on a TestFlight build | Research | P1 | 6h (calendar: 2 weeks) | T18, T21 |
| R06 | ☐ | Caregiver interviews: 3 short calls on at-home practice barriers and what "helpful" looks like | Research | P1 | 3h | T18, R12 |
| R07 | ☐ | Competitive landscape one-pager (SmallTalk, Constant Therapy, Tactus, Voice Aerobics, generic loudness apps) — what we do differently | Research | P1 | 3h | Positioning, R16 |
| R08 | ☐ | Functional phrase library: 20 phrases × 5 daily-life categories (greetings, café/restaurant, pharmacy, family, safety) | Content | P0 | 3h | T12 |
| R09 | ☐ | Warm-up set: 8 sustained vowels + 6 carrier phrases with marked stress patterns | Content | P0 | 1.5h | T12 |
| R10 | ☐ | Conversation prompts across 3 difficulty tiers (one-word answer, full sentence, multi-turn) — 10 each | Content | P0 | 2h | T12 |
| R11 | ☐ | Coach voice & tone style guide (warm, specific, never patronizing) — 2 pages with do/don't and 20 sample lines | Copy | P0 | 3h | T14, T15 |
| R12 | ☐ | Encouragement copy library: 30 ack variants + 30 feedback variants, calibrated to never overclaim or sound clinical | Copy | P0 | 2h | T14, T18 |
| R13 | ☐ | Onboarding copy: 3 screens at a 6th-grade reading level, including consent for local audio processing | Copy | P0 | 2h | T22 |
| R14 | ☐ | Plain-language safety disclaimer and "not a medical device" framing, reviewed by an SLP advisor | Compliance | P0 | 2h | T22, R04 |
| R15 | ☐ | Privacy policy + in-app data disclosure (local-first by default, explicit opt-in for cloud) | Compliance | P0 | 3h | T22 |
| R16 | ☐ | App Store and Play Store listing copy + screenshot storyboard (5 screens, captions, accessible alt text) | Marketing | P1 | 3h | T25 |
| R17 | ☐ | 90-second demo script with an explicit airplane-mode beat to prove on-device inference | Marketing | P0 | 1h | T25 |
| R18 | ☐ | Brand pass: name lock, low-vision-friendly palette (≥7:1 contrast on key surfaces), logo brief | Design | P0 | 3h | T01, T25 |
| R19 | ☐ | WCAG 2.2 AA checklist mapped to each RN screen, with pass/fail per criterion | Design | P0 | 2h | T21 |
| R20 | ☐ | One-page consent form for in-person user testing (plain language, no jargon) | Compliance | P1 | 1h | R05 |
| R21 | ☐ | Recruitment outreach plan: PD support groups, stroke recovery communities, local SLP clinics — message templates + target list | Ops | P1 | 2h | R04, R05, R06 |
| R22 | ☐ | Post-MVP feature shortlist synthesized from R04–R06 interviews; ranked by impact vs effort | Research | P2 | 2h | Roadmap |

### Supporting tracks — progress checklist

**P0 — must land alongside the MVP build (12 tasks)**

- [ ] R01 — Motor speech therapy literature review (LSVT LOUD, SPEAK OUT!)
- [ ] R02 — Defensible default targets per drill and severity tier
- [ ] R03 — Microphone / dB SPL calibration table (3+ phones)
- [ ] R04 — Recruit 2–3 SLP advisors and run protocol review
- [ ] R08 — Functional phrase library (20 × 5 categories)
- [ ] R09 — Warm-up vowels + carrier phrases with stress marks
- [ ] R10 — Conversation prompts across 3 difficulty tiers
- [ ] R11 — Coach voice & tone style guide
- [ ] R12 — Encouragement copy library (ack + feedback variants)
- [ ] R13 — Onboarding copy at 6th-grade reading level
- [ ] R14 — Safety disclaimer and "not a medical device" framing
- [ ] R15 — Privacy policy + in-app data disclosure
- [ ] R17 — 90-second demo script with airplane-mode proof
- [ ] R18 — Brand: name, low-vision palette, logo brief
- [ ] R19 — WCAG 2.2 AA checklist per screen

**P1 — strongly recommended (5 tasks)**

- [ ] R05 — 5-participant usability sessions on TestFlight
- [ ] R06 — Caregiver interviews (×3)
- [ ] R07 — Competitive landscape one-pager
- [ ] R16 — App Store / Play Store listing copy + screenshot storyboard
- [ ] R20 — Plain-language consent form for user testing
- [ ] R21 — Recruitment outreach plan + message templates

**P2 — stretch (1 task)**

- [ ] R22 — Post-MVP feature shortlist from interview synthesis

## 8. Build and run

```bash
git clone <this-repo>
cd voice-agents-hack-voice-coach/mobile
npm install
npx pod-install ios
npm run ios       # or: npm run android
```

The first launch downloads Gemma 4 E2B (~1.6 GB, INT4) and FunctionGemma 270M (~150 MB) and caches them under the app's documents directory. Subsequent launches are instant.

## 9. How to verify the implementation

A reviewer with a build of the app on a physical device can confirm the architectural claims in under a minute:

1. **Local-first.** Put the device in airplane mode, launch the app, and complete a full drill end-to-end. Nothing in the flow requires the network.
2. **DSP, not LLM, in the hot loop.** The loudness meter responds to voice within a single animation frame (~16 ms). This is faster than any LLM round-trip and confirms the metering layer is pure DSP.
3. **Sub-second multimodal turn-taking.** The coach's spoken response begins less than one second after the patient stops speaking, on a recent ARM device. This validates the Gemma 4 + Cactus latency budget.
4. **Structured output, not vibes.** The session summary screen reflects the same numbers stored in the local SQLite database, viewable via the React Native debugger or any SQLite browser pointed at the app's documents directory.

That is the implementation, end to end.
