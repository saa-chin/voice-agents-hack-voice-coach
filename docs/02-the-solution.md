# The Solution

## A pocket-sized speech therapist that runs entirely on the patient's phone

We are building **Voice Coach** — a daily speech-practice companion for people with Parkinson's, post-stroke dysarthria, and other motor speech disorders. It uses **Gemma 4** running on the **Cactus** on-device runtime to give patients the one thing they have never had between clinic visits: **immediate, objective, encouraging feedback on every sentence they speak.**

No cloud. No subscription tier for the audio analysis. No data ever leaves the device. Open the app, tap once, practice for five minutes, see your trend line move.

## The product in one screen

The patient opens the app and sees a single button: **Start today's practice.** Five minutes later they have completed a structured drill modeled on the LSVT LOUD protocol:

1. A **vocal warm-up** — sustained vowel and pitch glides — to measure baseline loudness and pitch range.
2. A short set of **functional phrases** — things they actually say in real life ("I'd like a coffee, please" / "Could you repeat that?") — with a live loudness meter and a target line.
3. A **conversational drill** — a 60-second open prompt where the coach holds a real conversation and gently nudges loudness, pace, and articulation in the moment.
4. A **session summary** — three numbers (loudness, pitch range, pace), a one-line encouragement, and a 7-day trend.

The whole experience is designed for an older adult who may have a tremor, may not be a strong reader, and is using a phone propped on a kitchen table.

## What makes this possible now (and not last year)

Three things had to be true at once. They are now all true.

### 1. Gemma 4 understands voice as voice, not as text

Every prior on-device speech app has been forced to do **audio → transcript → text-LLM → response**. That pipeline throws away exactly the information a speech coach needs: *how* the words were said. Loudness, hesitation, breathiness, pitch, pace — all gone by the time the LLM sees the input.

Gemma 4 is the first on-device model whose audio encoder feeds directly into the language model. It reasons over the **raw acoustic signal**. It can tell that the patient said "I'm fine" *quietly and with a falling pitch* — and respond accordingly. That single capability is what makes a credible speech coach possible on a phone.

### 2. Cactus makes Gemma 4 fast enough to feel real

A coach that takes 4 seconds to respond is not a coach. Cactus runs Gemma 4 on the device's NPU and ARM cores with INT4 quantization, hitting **~0.3 seconds end-to-end** on a recent iPhone or M-series device. The patient speaks. The coach responds. The interaction feels human.

### 3. Privacy is no longer a tradeoff

Because everything runs locally, the audio of a patient struggling to say their grandchild's name is processed and discarded on their own device. Nothing is uploaded. Nothing is logged on a server. There is no third-party data processor. This is not a feature we added — it is a structural property of the architecture, and it is what will get us into clinics.

## The clinical anchor: LSVT LOUD-inspired, not LSVT-branded

We are not claiming to deliver LSVT LOUD therapy itself — that is a trademarked, certified clinical protocol delivered by trained pathologists. We are building the **practice companion that lives between those sessions**, using the same evidence-backed core principle: **train the patient to recalibrate their internal sense of "normal loudness" through high-effort, high-repetition daily practice.**

This positioning matters legally (we are a wellness and practice tool, not a medical device on day one) and clinically (speech-language pathologists will recommend a tool that reinforces what they already do, not one that tries to replace them).

## The hybrid model: on-device by default, cloud only when it adds value

| Function | Where it runs | Why |
| --- | --- | --- |
| Live loudness, pitch, pace metering | On-device DSP | Microsecond latency, deterministic, no model needed |
| Real-time coaching feedback (audio in, speech out) | On-device — Gemma 4 E2B via Cactus | Privacy, latency, zero marginal cost |
| Intent routing and tool calls within the app | On-device — FunctionGemma 270M | Sub-100ms decisions, no round-trip |
| Weekly progress report for clinician (PDF) | Cloud — Gemini, opt-in only | Sanitized metrics only, no audio ever uploaded |
| Personalized new drill scripts | Cloud — Gemini, opt-in only | Pure text generation, no PHI |

The default mode is fully offline. Cloud is an opt-in upgrade for users who want clinician sharing or richer drill content. Even then, audio never leaves the device.

## Why this is a venture-backable business

- **Market.** ~12M Parkinson's patients globally + ~2M with other dysarthrias + ~3M children and adults with stuttering and articulation disorders. Aging populations grow this every year.
- **Adherence is the unsolved problem in speech therapy.** Every clinician we have read or spoken to identifies between-session practice as the biggest determinant of outcomes and the biggest failure point in current care. We are attacking the exact bottleneck.
- **Distribution channel exists.** ~150,000 US speech-language pathologists are looking for tools to recommend to patients. A free, private, clinician-friendly app spreads through them.
- **Defensible.** On-device multimodal voice understanding is hard. Cloud incumbents (Google, Microsoft, Amazon Health) are structurally disincentivized to build local-first products that generate no per-call revenue.
- **Regulatory path.** Start as a wellness tool (no clearance required), graduate to FDA Software as a Medical Device (Class II, de novo) once we have the outcomes data — the same path Pear, Akili, and Big Health walked.
- **Expandable.** The same engine extends naturally to post-stroke aphasia, ALS, accent training, professional voice coaching (singers, broadcasters), and language learning. Parkinson's is the wedge, not the ceiling.

## What the experience looks like

A patient opens the app and speaks the prompt phrase too quietly to be understood. The app shows a loudness meter falling short of the target line, and the coach gently says: *"I could hear you, but only just — try again, imagine you are calling across a room."* The patient tries again. The meter hits the target. The coach says: *"That is your voice. Same energy on the next one."*

The entire interaction — microphone capture, multimodal reasoning over the audio, spoken response — happens locally on the patient's device. With the network turned off. In under a second.

That is the product.
