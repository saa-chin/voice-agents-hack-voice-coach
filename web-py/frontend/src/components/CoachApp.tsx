import { useEffect, useRef, useState } from 'react';
import {
  cancelServerSpeech,
  cancelSpeech,
  createRecognizer,
  createRecorder,
  int16ToBase64,
  isRecognitionAvailable,
  isTTSAvailable,
  playWavBase64,
  primeTTS,
  speak,
  type AudioAnalysis,
  type RecognizerHandle,
  type RecorderHandle,
} from '../lib/audio';
import {
  connect,
  type CoachMsg,
  type Connection,
  type DrillMsg,
  type IntentResultMsg,
  type MetricsMsg,
  type ServerMessage,
  type SummaryMsg,
} from '../lib/ws';

const BACKEND_PORT =
  (import.meta as any).env?.PUBLIC_BACKEND_PORT ?? '8765';

function resolveWsUrl(): string {
  const explicit = (import.meta as any).env?.PUBLIC_BACKEND_WS_URL;
  if (explicit) return explicit;
  if (typeof window !== 'undefined' && window.location?.hostname) {
    // Use whatever hostname the page was loaded with (localhost vs 127.0.0.1
    // both work, but the browser is strict about mixed-origin WebSockets).
    return `ws://${window.location.hostname}:${BACKEND_PORT}/ws/coach`;
  }
  return `ws://127.0.0.1:${BACKEND_PORT}/ws/coach`;
}

const BACKEND_WS_URL = resolveWsUrl();

type Phase =
  | 'connecting'
  | 'loading'
  | 'ready'
  | 'drill'
  | 'recording'
  | 'thinking'
  | 'feedback'
  | 'done'
  | 'error';

const STAGES = ['warmup', 'glide', 'counting', 'main_task'] as const;
type Stage = (typeof STAGES)[number];

const STAGE_LABEL: Record<Stage, string> = {
  warmup: 'Warm-up',
  glide: 'Pitch glide',
  counting: 'Counting',
  main_task: 'Main task',
};

// ---- theme ----------------------------------------------------------------
//
// Two-tone (light/dark) toggle. The initial class is set by the no-flash
// inline script in index.astro before paint; we mirror that logic here so
// the React state stays in sync with whatever the script picked.

type Theme = 'light' | 'dark';

function readInitialTheme(): Theme {
  if (typeof document === 'undefined') return 'light';
  return document.documentElement.classList.contains('dark') ? 'dark' : 'light';
}

function applyTheme(theme: Theme) {
  if (typeof document === 'undefined') return;
  const root = document.documentElement;
  if (theme === 'dark') root.classList.add('dark');
  else root.classList.remove('dark');
  try {
    localStorage.setItem('vc-theme', theme);
  } catch {
    /* ignore quota / private mode */
  }
}

export default function CoachApp() {
  const [phase, setPhase] = useState<Phase>('connecting');
  const [drill, setDrill] = useState<DrillMsg | null>(null);
  const [metrics, setMetrics] = useState<MetricsMsg | null>(null);
  const [coach, setCoach] = useState<CoachMsg | null>(null);
  const [summary, setSummary] = useState<SummaryMsg['summary'] | null>(null);
  const [errMsg, setErrMsg] = useState<string | null>(null);
  const [transientError, setTransientError] = useState<string | null>(null);
  const [autoMode, setAutoMode] = useState(true);
  const [theme, setTheme] = useState<Theme>(readInitialTheme);
  // Live transcript while recording. We update on every recognizer event,
  // not via forceTick — interim results arrive faster than 50 ms.
  const [liveTranscript, setLiveTranscript] = useState<{
    final: string;
    interim: string;
  }>({ final: '', interim: '' });
  // Intent router state. `commandPhase` reflects what the inline
  // action-bar mic is doing (idle / listening / thinking); the
  // `intentResult` chip below shows the latest FunctionGemma 270M
  // verdict and the Whisper STT latency.
  const [commandPhase, setCommandPhase] = useState<
    'idle' | 'listening' | 'thinking'
  >('idle');
  const [intentResult, setIntentResult] = useState<IntentResultMsg | null>(
    null,
  );
  const [, forceTick] = useState(0);

  const wsRef = useRef<Connection | null>(null);
  const recRef = useRef<RecorderHandle | null>(null);
  const recognizerRef = useRef<RecognizerHandle | null>(null);
  // Dedicated PCM recorder for the "say a command" channel. Independent
  // of the drill recorder so a command capture can run while a drill
  // response is in flight without sharing state. The captured PCM is
  // sent to the backend (Cactus Whisper transcribes there) — the
  // browser's SpeechRecognition is no longer used for command audio,
  // because on Chrome it routes audio through Google's cloud STT and
  // breaks the on-device privacy story.
  const cmdRecorderRef = useRef<RecorderHandle | null>(null);
  const cmdTimeoutRef = useRef<number | null>(null);
  // True while we're suppressing the drill mic auto-arm because a
  // server-rendered TTS clip is still playing. Different from the
  // browser-TTS gate (`pendingUtterances`) because Audio.play() runs
  // in its own promise lane.
  const serverTtsPlayingRef = useRef(false);
  // Whether the BACKEND will render TTS (macOS `say`) for us. Snapped
  // from the `ready` frame and never changed mid-session, so the two
  // TTS engines (browser speechSynthesis + server WAV) never race and
  // we never end up with both voices speaking the same line.
  const serverTtsAvailableRef = useRef(false);
  const levelTimer = useRef<number | null>(null);
  // Track current phase via ref so async WS handlers don't read stale state.
  const phaseRef = useRef<Phase>('connecting');
  const autoModeRef = useRef(autoMode);
  // True between clicking "End" and the session truly closing. Suppresses
  // any in-flight `coach` / `audio_reply` / `drill` / `metrics` / `thinking`
  // frames the server may still emit while it finishes the model call it
  // started before the rest command landed — without this, those frames
  // would yank the UI back to `feedback` and the End button would feel
  // broken.
  const endingRef = useRef(false);
  // Forward refs for auto-mode callbacks defined inside `createRecorder` (VAD)
  // and `speak()` (TTS onEnd). Using refs keeps the lambdas stable and avoids
  // closing over stale `phase`/`autoMode` snapshots when async callbacks fire.
  const startRecordingRef = useRef<() => Promise<void>>();
  const stopRecordingRef = useRef<() => Promise<void>>();
  // StrictMode in dev runs effects twice. Without a guard we open two WS
  // connections per mount, which the server logs show happening.
  const wsInitialized = useRef(false);

  useEffect(() => {
    phaseRef.current = phase;
  }, [phase]);
  useEffect(() => {
    autoModeRef.current = autoMode;
  }, [autoMode]);

  // ---- WS lifecycle ------------------------------------------------------

  useEffect(() => {
    if (wsInitialized.current) {
      console.log('[coach] StrictMode re-mount: skipping duplicate connect');
      return;
    }
    wsInitialized.current = true;

    const ws = connect(BACKEND_WS_URL, handleServerMessage, {
      onCloseOrError: ({ code, reason, clean }) => {
        const p = phaseRef.current;
        if (p !== 'done' && p !== 'error') {
          setErrMsg(
            `Lost connection to coach server (code ${code}${reason ? `: ${reason}` : ''}${clean ? '' : ', not clean'}).`,
          );
          setPhase('error');
        }
      },
      onSendError: (err, msg) => {
        setTransientError(`Could not send ${msg.type}: ${err.message}`);
      },
    });
    wsRef.current = ws;
    ws.ready
      .catch((e) => {
        console.error('[coach] ws.ready rejected', e);
        setErrMsg(
          `Could not reach coach server at ${BACKEND_WS_URL}. Is the backend running?`,
        );
        setPhase('error');
      });

    return () => {
      console.log('[coach] cleanup: closing WS');
      ws.close();
      cancelSpeech();
      cancelServerSpeech();
      recRef.current?.cancel();
      recognizerRef.current?.cancel();
      recognizerRef.current = null;
      cmdRecorderRef.current?.cancel();
      cmdRecorderRef.current = null;
      if (cmdTimeoutRef.current !== null) {
        window.clearTimeout(cmdTimeoutRef.current);
        cmdTimeoutRef.current = null;
      }
      if (levelTimer.current !== null) window.clearInterval(levelTimer.current);
      wsInitialized.current = false;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  function handleServerMessage(msg: ServerMessage) {
    switch (msg.type) {
      case 'loading':
        setPhase('loading');
        return;
      case 'ready': {
        // Lock in the TTS source for the whole session. Re-snapping
        // mid-session would race in-flight `coach` replies, so we
        // only ever read this on the first `ready` we see per
        // connection.
        serverTtsAvailableRef.current = msg.tts_available === true;
        const p = phaseRef.current;
        if (p !== 'drill' && p !== 'recording' && p !== 'thinking') {
          setPhase('ready');
        }
        return;
      }
      case 'drill':
        if (endingRef.current) return;
        setDrill(msg);
        setMetrics(null);
        setCoach(null);
        setLiveTranscript({ final: '', interim: '' });
        setTransientError(null);
        setPhase('drill');
        // ONE voice for the entire session. If the backend rendered
        // the prompt with `say` and shipped a wav, play THAT (same
        // engine the coach reply uses). Otherwise speak via browser
        // TTS. Mixing the two is what makes it sound like "both my
        // laptop and the on-device model are speaking" — alternating
        // engines on every turn.
        speakDrillPrompt(msg, () => {
          // Auto-arm the mic the moment the prompt finishes reading.
          // Only if we're still in 'drill' (user hasn't manually
          // started, skipped, or finished the session) and auto-mode
          // is on.
          if (!autoModeRef.current) return;
          if (phaseRef.current !== 'drill') return;
          startRecordingRef.current?.();
        });
        return;
      case 'metrics':
        if (endingRef.current) return;
        setMetrics(msg);
        return;
      case 'thinking':
        if (endingRef.current) return;
        setPhase('thinking');
        return;
      case 'coach': {
        if (endingRef.current) return;
        setCoach(msg);
        setPhase('feedback');
        // EXACTLY ONE TTS source. If the backend reported it can
        // render via `say`, we wait for the `audio_reply` frame and
        // do nothing here — no racy fallback timer that would queue
        // a SECOND voice (the browser TTS) on top. If the backend
        // can't TTS, speak immediately via the browser.
        if (!serverTtsAvailableRef.current) {
          const spoken = joinForSpeech(msg.ack, msg.feedback);
          if (spoken) speak(spoken);
        }
        return;
      }
      case 'audio_reply': {
        if (endingRef.current) return;
        // Server-rendered TTS arrived. We never queued a browser
        // utterance for this turn (server-tts mode), so there is
        // nothing to cancel — just play the WAV.
        serverTtsPlayingRef.current = true;
        playWavBase64(msg.wav_b64)
          .catch((e) => {
            console.warn('[coach] server TTS playback failed', e);
            // Last-ditch only on actual playback failure: read the
            // current coach text via the browser so the user hears
            // SOMETHING. This is the only path that can produce
            // double-voice and only fires when WAV playback errors.
            const text = joinForSpeech(coach?.ack ?? '', coach?.feedback ?? '');
            if (text) speak(text);
          })
          .finally(() => {
            serverTtsPlayingRef.current = false;
          });
        return;
      }
      case 'advance':
      case 'retry':
      case 'rest':
        // Visual cue handled by next 'drill' or 'session_done' message.
        return;
      case 'intent_result':
        // Command-channel verdict from FunctionGemma 270M (or the regex
        // fallback). The action — if any — was already dispatched
        // server-side, so we only update the visible chip + transcript.
        setIntentResult(msg);
        setCommandPhase('idle');
        if (msg.action === 'none') {
          setTransientError(
            `Didn't catch a command in "${msg.utterance || '…'}". Try "skip", "repeat", or "I'm tired".`,
          );
        }
        return;
      case 'session_done':
        // Backend's authoritative summary always wins over any
        // synthesized one set by the local End handler.
        setSummary(msg.summary);
        setPhase('done');
        cancelSpeech();
        cancelServerSpeech();
        endingRef.current = false;
        return;
      case 'error':
        // Non-fatal: server stays connected and may send next drill.
        setTransientError(msg.message);
        return;
    }
  }

  // ---- User actions ------------------------------------------------------

  const startSession = () => {
    console.log('[coach] startSession clicked, ws state =', wsRef.current?.state());
    setSummary(null);
    setErrMsg(null);
    setTransientError(null);
    endingRef.current = false;
    // Prime the speech engine while we still have a user-gesture context.
    // Subsequent speak() calls from async WS handlers will be allowed.
    primeTTS();
    if (!wsRef.current) {
      setTransientError('No WebSocket connection. Reload the page.');
      return;
    }
    wsRef.current.send({ type: 'start_session' });
  };

  const startRecording = async () => {
    // Hard interrupt any in-flight TTS — we want a clean mic recording.
    cancelSpeech();
    cancelServerSpeech();
    setTransientError(null);
    setLiveTranscript({ final: '', interim: '' });
    // Defensive: stop + drop any leftover recorder from a prior turn.
    // createRecorder is *one-shot* — its MediaStream is .stop()'d in
    // the underlying stop() and can't be restarted. Always make a fresh one.
    if (recRef.current) {
      try { recRef.current.cancel(); } catch { /* ignore */ }
      recRef.current = null;
    }
    if (recognizerRef.current) {
      try { recognizerRef.current.cancel(); } catch { /* ignore */ }
      recognizerRef.current = null;
    }
    try {
      recRef.current = await createRecorder({
        // VAD is the auto stop-recording trigger. When the user goes silent
        // for `endHangoverMs` after having spoken, fire stopRecording().
        // We always *enable* VAD in auto mode; the recorder's analysis
        // numbers (rms/pitch/bands) are still computed either way for the
        // visualisers.
        vad: {
          enabled: autoModeRef.current,
          onSpeechStart: () => console.log('[coach vad] ▶ speech detected'),
          onSpeechEnd: () => {
            console.log('[coach vad] ■ silence after speech → auto-stop');
            stopRecordingRef.current?.();
          },
        },
      });
      await recRef.current.start();
      setPhase('recording');
      // Force a re-render every 50ms so the live visualisers animate
      // smoothly (matches the analyser tick rate).
      levelTimer.current = window.setInterval(
        () => forceTick((t) => t + 1),
        50,
      ) as unknown as number;

      // Live transcript runs in parallel via the browser's SpeechRecognition.
      // Failures here are non-fatal — recording proceeds without a transcript.
      recognizerRef.current = createRecognizer({
        onUpdate: (snap) => setLiveTranscript(snap),
        onError: (err) => {
          console.warn('[coach] recognizer error', err);
        },
      });
      recognizerRef.current?.start();
    } catch (exc: any) {
      console.error('[coach] startRecording failed', exc);
      recRef.current = null;
      setTransientError(`Mic error: ${exc?.message ?? exc}`);
      setPhase('drill');
    }
  };

  const stopRecording = async () => {
    if (!recRef.current) return;
    if (levelTimer.current !== null) {
      window.clearInterval(levelTimer.current);
      levelTimer.current = null;
    }
    setPhase('thinking');
    // Stop the live recognizer alongside the recorder so we don't keep
    // streaming partials while the backend is computing its answer.
    if (recognizerRef.current) {
      try { recognizerRef.current.stop(); } catch { /* ignore */ }
      recognizerRef.current = null;
    }
    // Hand off the recorder to a local var and clear the ref BEFORE awaiting,
    // so a failed stop() leaves the next attempt with a clean slate.
    const rec = recRef.current;
    recRef.current = null;
    try {
      const { pcm, durationMs } = await rec.stop();
      if (pcm.length === 0) {
        setTransientError(
          `No audio captured (${durationMs.toFixed(0)}ms). Try recording for at least half a second.`,
        );
        setPhase('drill');
        return;
      }
      wsRef.current?.send({
        type: 'audio',
        pcm_b64: int16ToBase64(pcm),
        sample_rate: 16000,
      });
    } catch (exc: any) {
      console.error('[coach] stopRecording failed', exc);
      setTransientError(`Capture error: ${exc?.message ?? exc}`);
      setPhase('drill');
    }
  };

  // Centralised drill-prompt TTS. Reads the same `serverTtsAvailableRef`
  // gate the coach-reply path uses, so the same TTS engine speaks the
  // entire session — no alternating voices. If the server shipped a
  // pre-rendered wav, play it; if not (server doesn't have `say`, or
  // render failed for this drill), fall back to browser speechSynthesis.
  const speakDrillPrompt = (msg: DrillMsg, onDone?: () => void) => {
    const fire = () => {
      try { onDone?.(); } catch (e) { console.warn('[coach] onDone threw', e); }
    };
    if (msg.prompt_wav_b64) {
      // Cancel any browser TTS that might be queued from a prior turn.
      cancelSpeech();
      playWavBase64(msg.prompt_wav_b64)
        .catch((e) => {
          console.warn('[coach] drill prompt wav failed, using browser TTS', e);
          // Last-ditch fallback so the user still hears the prompt.
          speak(buildDrillTTS(msg), { onEnd: () => fire() });
        })
        .then(fire);
      return;
    }
    speak(buildDrillTTS(msg), {
      onEnd: ({ ok }) => {
        if (!ok) console.warn('[coach] drill TTS failed; arming mic anyway');
        fire();
      },
    });
  };

  const repeatPrompt = () => {
    if (drill) {
      speakDrillPrompt(drill, () => {
        if (autoModeRef.current && phaseRef.current === 'drill') {
          startRecordingRef.current?.();
        }
      });
    }
  };
  const skipDrill = () => wsRef.current?.send({ type: 'command', action: 'skip' });

  // End the session immediately from the client's perspective. The
  // backend may still be mid-`asyncio.to_thread(model_call, …)` and
  // can't honour a `rest` for several seconds; without this fast-path
  // the user sees no response, then the server's still-in-flight
  // `coach`/`audio_reply` frames yank the UI back to feedback —
  // making End feel completely broken. Strategy:
  //   1. Mark `endingRef = true` so subsequent server frames are
  //      ignored (see handleServerMessage). Only `session_done` and
  //      `error` get through.
  //   2. Tear down all client-side audio (mic, recogniser, voice
  //      command, browser/server TTS, level timer).
  //   3. Optimistically transition to the done view with a synthesized
  //      summary so the user gets instant feedback.
  //   4. Send `command/rest`. When the real `session_done` arrives,
  //      it overwrites the placeholder with backend-authoritative data.
  const restSession = () => {
    endingRef.current = true;
    cancelSpeech();
    cancelServerSpeech();
    if (recRef.current) {
      try { recRef.current.cancel(); } catch { /* ignore */ }
      recRef.current = null;
    }
    if (recognizerRef.current) {
      try { recognizerRef.current.cancel(); } catch { /* ignore */ }
      recognizerRef.current = null;
    }
    if (cmdRecorderRef.current) {
      try { cmdRecorderRef.current.cancel(); } catch { /* ignore */ }
      cmdRecorderRef.current = null;
    }
    if (cmdTimeoutRef.current !== null) {
      window.clearTimeout(cmdTimeoutRef.current);
      cmdTimeoutRef.current = null;
    }
    if (levelTimer.current !== null) {
      window.clearInterval(levelTimer.current);
      levelTimer.current = null;
    }
    setCommandPhase('idle');
    setTransientError(null);
    // Best-guess summary from what we know locally. Server's
    // session_done will replace this with the real numbers.
    setSummary({
      advanced: drill?.position ?? 0,
      total: drill?.total ?? (drill ? 1 : 0),
      retries: 0,
      avg_dbfs: null,
      json_failures: 0,
      rest_called: true,
      session_log: '(ending…)',
    });
    setPhase('done');
    wsRef.current?.send({ type: 'command', action: 'rest' });
  };

  // ---- Voice-command channel (Cactus Whisper + FunctionGemma 270M) -------
  //
  // Independent of the drill mic. We capture raw PCM with the same
  // pipeline the drill loop uses, ship it to /ws/coach as
  // { type: "intent_audio", pcm_b64 }, and the backend transcribes
  // on-device with Cactus Whisper before routing the text through
  // FunctionGemma 270M. NO browser SpeechRecognition: that API
  // routes audio through Google's cloud STT on Chrome and would
  // break the on-device privacy story this whole project is built on.
  //
  // Typed commands still go through `{ type: "intent", utterance }`
  // (no audio, no Whisper) for users without a mic and for demo flow.

  // Hard cap so a forgotten button never hangs the mic. Most command
  // utterances are < 2 s; VAD-driven auto-stop handles the common case.
  const CMD_CAPTURE_MS = 5000;

  const cancelVoiceCommand = () => {
    if (cmdTimeoutRef.current !== null) {
      window.clearTimeout(cmdTimeoutRef.current);
      cmdTimeoutRef.current = null;
    }
    if (cmdRecorderRef.current) {
      try { cmdRecorderRef.current.cancel(); } catch { /* ignore */ }
      cmdRecorderRef.current = null;
    }
    setCommandPhase('idle');
  };

  // Helper to stop the command recorder, base64 the PCM, and ship
  // it. Idempotent — once it fires we clear refs so VAD/timeout
  // races don't double-send.
  const finaliseVoiceCommand = async () => {
    if (cmdTimeoutRef.current !== null) {
      window.clearTimeout(cmdTimeoutRef.current);
      cmdTimeoutRef.current = null;
    }
    const rec = cmdRecorderRef.current;
    cmdRecorderRef.current = null;
    if (!rec) return;
    setCommandPhase('thinking');
    try {
      const { pcm, durationMs } = await rec.stop();
      if (pcm.length === 0 || durationMs < 250) {
        setCommandPhase('idle');
        setTransientError(
          `No command heard (${durationMs.toFixed(0)} ms captured).`,
        );
        return;
      }
      wsRef.current?.send({
        type: 'intent_audio',
        pcm_b64: int16ToBase64(pcm),
        sample_rate: 16000,
      });
    } catch (exc: any) {
      console.error('[coach cmd] capture failed', exc);
      setCommandPhase('idle');
      setTransientError(`Command mic error: ${exc?.message ?? exc}`);
    }
  };

  const startVoiceCommand = async () => {
    if (commandPhase !== 'idle') {
      cancelVoiceCommand();
      return;
    }
    setIntentResult(null);
    setTransientError(null);
    setCommandPhase('listening');
    // Same sample format the drill mic uses (16 kHz mono Int16). The
    // backend's Whisper holder expects exactly that.
    try {
      const recorder = await createRecorder({
        vad: {
          enabled: true,
          // Trip the auto-stop a bit faster than the drill mic so a
          // quick "skip" doesn't keep the mic open after the word.
          endHangoverMs: 600,
          onSpeechEnd: () => {
            console.log('[coach cmd vad] silence after speech → finalise');
            finaliseVoiceCommand();
          },
        },
      });
      await recorder.start();
      cmdRecorderRef.current = recorder;
    } catch (exc: any) {
      console.error('[coach cmd] startRecording failed', exc);
      setCommandPhase('idle');
      setTransientError(`Mic error: ${exc?.message ?? exc}`);
      return;
    }
    // Hard timeout safety net in case VAD never fires (silent room,
    // mic muted, etc.) so the panel never stays stuck on "Listening".
    cmdTimeoutRef.current = window.setTimeout(() => {
      cmdTimeoutRef.current = null;
      finaliseVoiceCommand();
    }, CMD_CAPTURE_MS) as unknown as number;
  };

  // Refresh forward refs every render so the latest closures (with current
  // state) are what the async VAD / TTS callbacks reach for.
  startRecordingRef.current = startRecording;
  stopRecordingRef.current = stopRecording;

  const toggleTheme = () => {
    setTheme((prev) => {
      const next: Theme = prev === 'dark' ? 'light' : 'dark';
      applyTheme(next);
      return next;
    });
  };

  // ---- Render ------------------------------------------------------------

  const analysis: AudioAnalysis | null =
    phase === 'recording' ? recRef.current?.analysis() ?? null : null;
  const liveLevel = analysis?.rms != null ? Math.min(1, analysis.rms * 4) : 0;
  const liveScore = computeLiveScore(analysis, drill?.target_dbfs ?? -25);

  const inSession =
    phase === 'drill' ||
    phase === 'recording' ||
    phase === 'thinking' ||
    phase === 'feedback';

  return (
    <main
      className={
        'relative z-10 mx-auto flex min-h-screen w-full flex-col px-6 py-10 ' +
        (inSession ? 'max-w-6xl' : 'max-w-2xl')
      }
    >
      <Header theme={theme} onToggleTheme={toggleTheme} />

      <section className="flex flex-1 flex-col gap-4">
        {!isTTSAvailable() && (
          <div className="rounded-xl border border-[var(--warning-soft)] bg-[var(--warning-soft)] px-4 py-2 text-xs text-[var(--warning)]">
            This browser doesn't expose <code>speechSynthesis</code>. The coach
            will be silent — try Chrome, Edge, or Safari.
          </div>
        )}
        {phase === 'connecting' && (
          <StatusCard label="Connecting to local coach…" spinner />
        )}
        {phase === 'loading' && (
          <StatusCard
            label="Loading Gemma 4 on your machine…"
            hint="One-time, takes ~6 seconds. The model never leaves your device."
            spinner
          />
        )}
        {phase === 'ready' && <StartCard onStart={startSession} />}

        {inSession && drill && (
          <SessionView
            drill={drill}
            metrics={metrics}
            coach={coach}
            phase={phase}
            level={liveLevel}
            analysis={analysis}
            liveScore={liveScore}
            transcript={liveTranscript}
            autoMode={autoMode}
            onToggleAuto={() => setAutoMode((v) => !v)}
            transientError={transientError}
            onStartRec={startRecording}
            onStopRec={stopRecording}
            onRepeat={repeatPrompt}
            onSkip={skipDrill}
            onRest={restSession}
            commandPhase={commandPhase}
            intentResult={intentResult}
            onStartCommand={startVoiceCommand}
            onCancelCommand={cancelVoiceCommand}
          />
        )}

        {phase === 'done' && summary && (
          <SummaryCard summary={summary} onRestart={startSession} />
        )}
        {phase === 'error' && <ErrorCard message={errMsg ?? 'Unknown error.'} />}
      </section>

      <Footer />
    </main>
  );
}

// ---- Sub-components -------------------------------------------------------

function Header({
  theme,
  onToggleTheme,
}: {
  theme: Theme;
  onToggleTheme: () => void;
}) {
  return (
    <header className="relative mb-10 flex items-start justify-between gap-3">
      <div>
        <h1 className="text-4xl font-extrabold tracking-tight text-[var(--text)]">
          Voice Coach
        </h1>
        <p className="mt-1.5 text-sm text-[var(--text-muted)]">
          On-device speech practice · Gemma 4 + Cactus
        </p>
      </div>
      <div className="mt-2 flex items-center gap-2">
        <ThemeToggle theme={theme} onToggle={onToggleTheme} />
      </div>
    </header>
  );
}

function ThemeToggle({
  theme,
  onToggle,
}: {
  theme: Theme;
  onToggle: () => void;
}) {
  const isDark = theme === 'dark';
  return (
    <button
      type="button"
      onClick={onToggle}
      aria-label={isDark ? 'Switch to light theme' : 'Switch to dark theme'}
      title={isDark ? 'Switch to light theme' : 'Switch to dark theme'}
      className="inline-flex h-9 w-9 items-center justify-center rounded-full border border-[var(--border)] bg-[var(--surface-2)] text-[var(--text-muted)] backdrop-blur transition hover:bg-[var(--surface)] hover:text-[var(--text)] focus:outline-none focus:ring-2 focus:ring-[var(--accent-ring)]"
    >
      {isDark ? (
        // Sun icon — clicking returns to light
        <svg
          xmlns="http://www.w3.org/2000/svg"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="2"
          strokeLinecap="round"
          strokeLinejoin="round"
          className="h-4 w-4"
          aria-hidden="true"
        >
          <circle cx="12" cy="12" r="4" />
          <path d="M12 2v2" />
          <path d="M12 20v2" />
          <path d="m4.93 4.93 1.41 1.41" />
          <path d="m17.66 17.66 1.41 1.41" />
          <path d="M2 12h2" />
          <path d="M20 12h2" />
          <path d="m6.34 17.66-1.41 1.41" />
          <path d="m19.07 4.93-1.41 1.41" />
        </svg>
      ) : (
        // Moon icon — clicking switches to dark
        <svg
          xmlns="http://www.w3.org/2000/svg"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="2"
          strokeLinecap="round"
          strokeLinejoin="round"
          className="h-4 w-4"
          aria-hidden="true"
        >
          <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z" />
        </svg>
      )}
    </button>
  );
}

function Footer() {
  return (
    <footer className="mt-8 border-t border-[var(--border)] pt-4 text-xs text-[var(--text-faint)]">
      Audio is processed entirely on your machine. Nothing is sent to any cloud
      service.
    </footer>
  );
}

function StatusCard({
  label,
  hint,
  spinner,
}: {
  label: string;
  hint?: string;
  spinner?: boolean;
}) {
  return (
    <div className="card p-8 text-center">
      <div className="text-lg font-semibold text-[var(--text)]">{label}</div>
      {hint && (
        <div className="mt-2 text-sm text-[var(--text-muted)]">{hint}</div>
      )}
      {spinner && (
        <div className="mt-6 flex items-center justify-center gap-2">
          <div className="h-2.5 w-2.5 animate-bounce rounded-full bg-[var(--accent)] [animation-delay:-0.3s]" />
          <div className="h-2.5 w-2.5 animate-bounce rounded-full bg-[var(--warning)] [animation-delay:-0.15s]" />
          <div className="h-2.5 w-2.5 animate-bounce rounded-full bg-[var(--violet)]" />
        </div>
      )}
    </div>
  );
}

function StartCard({ onStart }: { onStart: () => void }) {
  // 11 bars, varying max heights + staggered delays so the wave never
  // looks like a metronome. Heights are %s of the SVG viewport height.
  const bars = [
    { h: 30, delay: '0.00s' },
    { h: 60, delay: '0.10s' },
    { h: 95, delay: '0.20s' },
    { h: 75, delay: '0.30s' },
    { h: 100, delay: '0.40s' },
    { h: 55, delay: '0.50s' },
    { h: 85, delay: '0.45s' },
    { h: 100, delay: '0.35s' },
    { h: 70, delay: '0.25s' },
    { h: 50, delay: '0.15s' },
    { h: 30, delay: '0.05s' },
  ];

  const steps = [
    {
      label: 'Warm up',
      detail: 'Easy vowels & glides',
      tone: 'accent' as const,
    },
    {
      label: 'Practice',
      detail: 'Phrases & counting',
      tone: 'violet' as const,
    },
    {
      label: 'Converse',
      detail: 'Real-world prompts',
      tone: 'warning' as const,
    },
  ];

  return (
    <div className="card relative overflow-hidden p-10 text-center">
      {/* ambient floating blobs — gentle vertical drift adds life
          without distracting from the CTA. */}
      <div className="blob-float pointer-events-none absolute -left-20 -top-20 h-64 w-64 rounded-full bg-[var(--accent-soft)] blur-3xl" />
      <div
        className="blob-float pointer-events-none absolute -bottom-24 -right-20 h-72 w-72 rounded-full bg-[var(--violet-soft)] blur-3xl"
        style={{ animationDelay: '-3s' }}
      />
      <div
        className="blob-float pointer-events-none absolute right-1/3 top-4 h-32 w-32 rounded-full bg-[var(--warning-soft)] blur-2xl"
        style={{ animationDelay: '-5s' }}
      />

      <div className="relative">
        {/* Animated waveform stand-in for the old mic glyph. Same
            silhouette as the live LiveAnalyzer so the start screen
            previews the experience. */}
        <div
          className="mx-auto mb-5 flex h-20 w-44 items-end justify-center gap-1.5 rounded-2xl bg-[var(--accent-soft)] px-5 py-3 ring-1 ring-[var(--accent-ring)]"
          aria-hidden="true"
        >
          {bars.map((b, i) => (
            <div
              key={i}
              className="wave-bar w-1.5 rounded-full bg-[var(--accent)]"
              style={{
                height: `${b.h}%`,
                animationDelay: b.delay,
              }}
            />
          ))}
        </div>

        <h2 className="text-4xl font-extrabold leading-tight tracking-tight text-[var(--text)] sm:text-5xl">
          Let&apos;s find{' '}
          <span className="text-gradient-accent">your voice</span>.
        </h2>
        <p className="mx-auto mt-4 max-w-md text-base leading-relaxed text-[var(--text-muted)]">
          Ten quick drills, one private coach, zero cloud. Stretch your vocal
          cords, warm up your speech, and have a little fun while you do it.
        </p>

        {/* Three "what's coming" tiles — gives the user a sense of the
            arc instead of a wall of prose. */}
        <div className="mx-auto mt-8 grid max-w-md grid-cols-3 gap-2 text-left">
          {steps.map((s, i) => {
            const tones: Record<typeof s.tone, string> = {
              accent:
                'bg-[var(--accent-soft)] text-[var(--accent)] border-[var(--accent-ring)]',
              violet:
                'bg-[var(--violet-soft)] text-[var(--violet)] border-[var(--violet-soft)]',
              warning:
                'bg-[var(--warning-soft)] text-[var(--warning)] border-[var(--warning-soft)]',
            };
            return (
              <div
                key={s.label}
                className="card-soft flex flex-col items-start gap-1 p-3"
              >
                <span
                  className={
                    'inline-flex h-6 w-6 items-center justify-center rounded-full border text-[11px] font-bold ' +
                    tones[s.tone]
                  }
                >
                  {i + 1}
                </span>
                <span className="text-sm font-semibold text-[var(--text)]">
                  {s.label}
                </span>
                <span className="text-[11px] leading-snug text-[var(--text-faint)]">
                  {s.detail}
                </span>
              </div>
            );
          })}
        </div>

        <button
          onClick={onStart}
          className="cta-shimmer group relative mt-8 inline-flex items-center justify-center gap-2 overflow-hidden rounded-full bg-[var(--accent)] px-12 py-4 text-base font-bold tracking-wide text-[var(--accent-fg)] shadow-xl shadow-[var(--accent-soft)] transition-all duration-200 hover:scale-[1.03] hover:bg-[var(--accent-strong)] hover:shadow-2xl active:scale-[0.97]"
        >
          <span className="relative">Start session</span>
          <svg
            xmlns="http://www.w3.org/2000/svg"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2.5"
            strokeLinecap="round"
            strokeLinejoin="round"
            className="relative h-4 w-4 transition-transform duration-200 group-hover:translate-x-1"
            aria-hidden="true"
          >
            <path d="M5 12h14" />
            <path d="m13 6 6 6-6 6" />
          </svg>
        </button>

        <div className="mt-4 flex flex-wrap items-center justify-center gap-x-3 gap-y-1 text-[11px] text-[var(--text-faint)]">
          <span className="inline-flex items-center gap-1">
            <span className="h-1.5 w-1.5 rounded-full bg-[var(--accent)]" />
            100% on-device
          </span>
          <span className="inline-flex items-center gap-1">
            <span className="h-1.5 w-1.5 rounded-full bg-[var(--violet)]" />
            Gemma 4
          </span>
          <span className="inline-flex items-center gap-1">
            <span className="h-1.5 w-1.5 rounded-full bg-[var(--warning)]" />
            ~10 min
          </span>
        </div>
      </div>
    </div>
  );
}

function SessionView({
  drill,
  metrics,
  coach,
  phase,
  level,
  analysis,
  liveScore,
  transcript,
  autoMode,
  onToggleAuto,
  transientError,
  onStartRec,
  onStopRec,
  onRepeat,
  onSkip,
  onRest,
  commandPhase,
  intentResult,
  onStartCommand,
  onCancelCommand,
}: {
  drill: DrillMsg;
  metrics: MetricsMsg | null;
  coach: CoachMsg | null;
  phase: Phase;
  level: number;
  analysis: AudioAnalysis | null;
  liveScore: LiveScore | null;
  transcript: { final: string; interim: string };
  autoMode: boolean;
  onToggleAuto: () => void;
  transientError: string | null;
  onStartRec: () => void;
  onStopRec: () => void;
  onRepeat: () => void;
  onSkip: () => void;
  onRest: () => void;
  commandPhase: 'idle' | 'listening' | 'thinking';
  intentResult: IntentResultMsg | null;
  onStartCommand: () => void;
  onCancelCommand: () => void;
}) {
  // Show the transcript card whenever there's anything to show, OR while
  // recording (so users see the placeholder calibrate as they begin to talk).
  const showTranscript =
    isRecognitionAvailable() &&
    (phase === 'recording' ||
      phase === 'thinking' ||
      phase === 'feedback') &&
    (transcript.final || transcript.interim || phase === 'recording');
  return (
    <div className="flex flex-col gap-4">
      {/* Stage indicator stays full-width so progress through the
          warmup → glide → counting → main_task pipeline reads as
          one banner across the top of the workspace. */}
      <StageIndicator drill={drill} />
      {/* Two columns on lg+ screens: left side is the "doing"
          workspace (prompt + mic + controls), right side is the
          "watching" workspace (live signals, transcript, metrics,
          coach feedback). Below lg we collapse back to a single
          stack so phones still read top-to-bottom. */}
      <div className="grid grid-cols-1 gap-4 lg:grid-cols-2 lg:items-start">
        <div className="flex flex-col gap-4">
          <PromptCard drill={drill} onRepeat={onRepeat} />
          <MicButton
            phase={phase}
            level={level}
            autoMode={autoMode}
            speaking={analysis?.speaking ?? false}
            onStart={onStartRec}
            onStop={onStopRec}
          />
          {/* One thin action bar inline with the mic — no separate
              panel at the bottom. Voice commands invoke the same
              actions as the pills; chip below shows the routed
              verdict + latencies. */}
          <ActionBar
            commandPhase={commandPhase}
            intentResult={intentResult}
            onRepeat={onRepeat}
            onSkip={onSkip}
            onRest={onRest}
            onStartCommand={onStartCommand}
            onCancelCommand={onCancelCommand}
            disabled={phase === 'recording' || phase === 'thinking'}
          />
          <AutoModeRow
            autoMode={autoMode}
            onToggle={onToggleAuto}
            phase={phase}
          />
        </div>
        <div className="flex flex-col gap-4">
          <LiveAnalyzer
            analysis={analysis}
            target_dbfs={drill.target_dbfs}
            phase={phase}
            liveScore={liveScore}
          />
          {showTranscript && (
            <LiveTranscript transcript={transcript} phase={phase} />
          )}
          {metrics && (
            <MetricsLine metrics={metrics} target={drill.target_dbfs} />
          )}
          {coach && <CoachCard coach={coach} />}
        </div>
      </div>
      {transientError && <TransientError message={transientError} />}
    </div>
  );
}

function StageIndicator({ drill }: { drill: DrillMsg }) {
  return (
    <div className="flex flex-col gap-2 text-xs">
      {drill.exercise_name && (
        <div className="flex items-center justify-between text-[11px] uppercase tracking-wider text-[var(--text-faint)]">
          <span>
            {drill.category_name && (
              <>
                <span className="text-[var(--text-muted)]">
                  {drill.category_name}
                </span>
                <span className="mx-1 text-[var(--text-faint)]">›</span>
              </>
            )}
            <span className="font-medium text-[var(--text)]">
              {drill.exercise_name}
            </span>
          </span>
          <span>
            step {drill.position + 1} / {drill.total}
          </span>
        </div>
      )}
      <div className="flex flex-wrap gap-1.5">
        {STAGES.map((s, i) => {
          const active = drill.stage === s;
          const done = STAGES.indexOf(drill.stage as Stage) > i;
          return (
            <span
              key={s}
              className={
                'rounded-full border px-3 py-1 text-xs transition-all duration-300 ' +
                (active
                  ? 'border-transparent bg-[var(--accent)] text-[var(--accent-fg)] font-semibold shadow-sm'
                  : done
                  ? 'border-[var(--border)] bg-[var(--surface-2)] text-[var(--text-faint)] line-through'
                  : 'border-[var(--border)] bg-[var(--surface-2)] text-[var(--text-muted)]')
              }
            >
              {STAGE_LABEL[s]}
            </span>
          );
        })}
      </div>
    </div>
  );
}

function PromptCard({
  drill,
  onRepeat,
}: {
  drill: DrillMsg;
  onRepeat: () => void;
}) {
  // For instruction-only phases (warmup, glide), `note` is empty and the
  // `prompt` IS the full instruction. Render it as the cue rather than as
  // an "expected utterance" in quotes.
  const isInstructionOnly = !drill.note?.trim();
  const reps = drill.target_repetitions ?? 0;
  const dur = drill.target_duration_sec ?? 0;
  return (
    <div className="card p-6">
      <div className="text-xs font-semibold uppercase tracking-wider text-[var(--accent)]">
        {isInstructionOnly ? 'Phase cue' : 'Say this'}
      </div>
      {isInstructionOnly ? (
        <div className="mt-2 text-2xl font-bold leading-snug text-[var(--text)]">
          {drill.prompt}
        </div>
      ) : (
        <>
          <div className="mt-2 text-3xl font-bold leading-tight text-[var(--text)]">
            “{drill.prompt}”
          </div>
          <div className="mt-3 text-sm text-[var(--text-muted)]">
            {drill.note}
          </div>
        </>
      )}
      {drill.focus && (
        <div className="mt-3 inline-block rounded-full bg-[var(--warning-soft)] px-2.5 py-1 text-xs font-medium text-[var(--warning)]">
          focus: {drill.focus}
        </div>
      )}
      {(reps > 0 || dur > 0) && (
        <div className="mt-2 font-mono text-[11px] text-[var(--text-faint)]">
          {reps > 1 && `target ${reps} reps`}
          {reps > 1 && dur > 0 && ' · '}
          {dur > 0 && `~${dur}s each`}
        </div>
      )}
      <button
        onClick={onRepeat}
        className="mt-4 text-xs font-medium text-[var(--accent)] underline-offset-4 hover:underline"
      >
        ▶︎ hear it again
      </button>
    </div>
  );
}

function MicButton({
  phase,
  level,
  autoMode,
  speaking,
  onStart,
  onStop,
}: {
  phase: Phase;
  level: number;
  autoMode: boolean;
  speaking: boolean;
  onStart: () => void;
  onStop: () => void;
}) {
  const isRecording = phase === 'recording';
  const isThinking = phase === 'thinking';
  const isDrill = phase === 'drill';
  const disabled = isThinking;

  const ringScale = isRecording ? 1 + level * 0.5 : 1;

  return (
    <div className="my-4 flex flex-col items-center gap-2">
      <div className="relative flex items-center justify-center">
        {/* animated pulse ring */}
        {isRecording && (
          <>
            <div
              className={
                'absolute h-24 w-24 rounded-full mic-ring-anim ' +
                (speaking ? 'bg-[var(--accent-soft)]' : 'bg-[var(--danger-soft)]')
              }
            />
            <div
              className={
                'absolute h-24 w-24 rounded-full transition-transform duration-100 ' +
                (speaking
                  ? 'bg-[var(--accent-soft)]'
                  : 'bg-[var(--danger-soft)]')
              }
              style={{ transform: `scale(${ringScale + 0.3})` }}
            />
          </>
        )}
        <button
          onClick={isRecording ? onStop : onStart}
          disabled={disabled}
          className={
            'relative flex h-24 w-24 items-center justify-center rounded-full text-3xl font-semibold transition-all duration-200 active:scale-95 ' +
            (isRecording
              ? speaking
                ? 'bg-[var(--accent)] text-[var(--accent-fg)] glow-accent'
                : 'bg-[var(--danger)] text-white glow-rose'
              : isThinking
              ? 'bg-[var(--surface-inset)] text-[var(--text-faint)] cursor-not-allowed'
              : 'bg-[var(--accent)] text-[var(--accent-fg)] shadow-xl shadow-[var(--accent-soft)] hover:bg-[var(--accent-strong)]')
          }
          aria-label={isRecording ? 'Stop recording' : 'Start recording'}
        >
          {isThinking ? (
            <span className="inline-block h-6 w-6 animate-spin rounded-full border-2 border-[var(--border-strong)] border-t-[var(--accent)]" />
          ) : isRecording ? (
            '■'
          ) : (
            '🎤'
          )}
        </button>
      </div>
      <div className="text-xs text-[var(--text-muted)]">
        {isRecording
          ? speaking
            ? autoMode
              ? 'Listening… stop talking when finished'
              : 'Recording — tap to stop'
            : autoMode
            ? 'Mic open — start speaking'
            : 'Recording — tap to stop'
          : isThinking
          ? 'Coach is thinking…'
          : isDrill && autoMode
          ? 'Coach is reading the prompt — mic will arm automatically'
          : 'Tap to record your attempt'}
      </div>
    </div>
  );
}

function MetricsLine({
  metrics,
  target,
}: {
  metrics: MetricsMsg;
  target: number;
}) {
  return (
    <div className="card-soft px-4 py-2 font-mono text-xs text-[var(--text-muted)]">
      captured {metrics.duration_s.toFixed(1)}s · loudness{' '}
      {metrics.dbfs == null ? '—' : `${metrics.dbfs.toFixed(1)} dBFS`} (target{' '}
      {target.toFixed(1)} dBFS)
    </div>
  );
}

// ---- Live visualisers ----------------------------------------------------
//
// Renders four parameters in real-time while the mic is open:
//   • a horizontal loudness meter with the drill's target dBFS marker
//   • per-band spectrum bars (8 bands, log-spaced across the voice range)
//   • dominant-pitch readout (F0 in 80-400 Hz)
//   • voice-activity lamp (driven by the same hysteresis VAD that auto-stops)
//
// The component is only meaningful while `phase === 'recording'` (since
// that's when `analysis` is non-null), but it renders a calm "idle" state
// during 'drill' / 'thinking' / 'feedback' so the layout doesn't jump.

const BAND_LABELS = ['80', '160', '320', '500', '800', '1.3k', '2.1k', '3.3k'];

function LiveAnalyzer({
  analysis,
  target_dbfs,
  phase,
  liveScore,
}: {
  analysis: AudioAnalysis | null;
  target_dbfs: number;
  phase: Phase;
  liveScore: LiveScore | null;
}) {
  const recording = phase === 'recording';
  // Map dBFS to a 0..1 bar position. -60 dBFS reads as 0, 0 dBFS as 1.
  // Below -60 dBFS we treat as silence (bar pinned to 0). The target marker
  // uses the same mapping so it sits at the right spot on the meter.
  const dbfsToFrac = (db: number) => {
    if (!isFinite(db)) return 0;
    return Math.max(0, Math.min(1, (db + 60) / 60));
  };

  const dbfs = analysis?.dbfs ?? -Infinity;
  const loudFrac = dbfsToFrac(dbfs);
  const targetFrac = dbfsToFrac(target_dbfs);
  const onTarget =
    isFinite(dbfs) && Math.abs(dbfs - target_dbfs) <= 4 && recording;

  const bands = analysis?.bands ?? new Array(8).fill(0);
  const pitchHz = analysis?.pitchHz ?? null;
  const speaking = (analysis?.speaking ?? false) && recording;

  return (
    <div className="card p-4">
      <div className="mb-3 flex items-center justify-between text-[10px] uppercase tracking-wider text-[var(--text-faint)]">
        <span>Live signal</span>
        <span className="flex items-center gap-1.5">
          <span
            className={
              'inline-block h-2 w-2 rounded-full transition ' +
              (speaking
                ? 'bg-[var(--accent)] shadow-[0_0_6px_2px_var(--accent-ring)]'
                : recording
                ? 'bg-[var(--text-faint)]'
                : 'bg-[var(--border-strong)]')
            }
          />
          {speaking ? 'voice' : recording ? 'silence' : 'idle'}
        </span>
      </div>

      <div className="grid grid-cols-[1fr_auto] gap-4">
        <div className="flex flex-col gap-3">
          <LoudnessMeter
            loudFrac={loudFrac}
            targetFrac={targetFrac}
            dbfs={dbfs}
            target_dbfs={target_dbfs}
            onTarget={onTarget}
            active={recording}
          />
          <div className="grid grid-cols-[1fr_auto] gap-4">
            <SpectrumBars bands={bands} active={recording} />
            <PitchGauge pitchHz={pitchHz} active={recording && speaking} />
          </div>
        </div>
        <ScoreGauge score={liveScore} active={recording} />
      </div>
    </div>
  );
}

interface LiveScore {
  /** 0..100 composite score, latest frame. */
  total: number;
  /** 0..100 sub-score: how close current dBFS is to target_dbfs. */
  loudness: number;
  /** 0..100 sub-score: % of frames since recording-start that were voiced. */
  voicing: number;
}

function computeLiveScore(
  analysis: AudioAnalysis | null,
  target_dbfs: number,
): LiveScore | null {
  if (!analysis) return null;
  // Loudness sub-score: gaussian falloff from target. Within ±3 dB ≈ 100,
  // ±10 dB ≈ 50, ±20 dB ≈ 6. Treats silence (no finite dBFS) as 0.
  let loudness = 0;
  if (isFinite(analysis.dbfs)) {
    const diff = analysis.dbfs - target_dbfs;
    loudness = 100 * Math.exp(-Math.pow(diff / 10, 2));
  }
  // Voicing sub-score: cumulative voiced ratio. Cap at 70% (≈ realistic for
  // sustained speech with natural pauses) so the user doesn't need 100%
  // voicing to score full marks.
  const voicing = Math.min(100, (analysis.voicedRatio / 0.7) * 100);
  const total = 0.6 * loudness + 0.4 * voicing;
  return {
    total: Math.round(total),
    loudness: Math.round(loudness),
    voicing: Math.round(voicing),
  };
}

function ScoreGauge({
  score,
  active,
}: {
  score: LiveScore | null;
  active: boolean;
}) {
  // SVG circular meter. Stroke-dasharray trick: full circumference = 220, the
  // dash length tracks the score percentage. Color tier matches the bar
  // gradient: rose < 40 < amber < 70 < emerald.
  const value = active ? score?.total ?? 0 : 0;
  const SIZE = 96;
  const STROKE = 8;
  const R = (SIZE - STROKE) / 2;
  const C = 2 * Math.PI * R;
  const dash = (value / 100) * C;
  const ringStroke = !active
    ? 'var(--border-strong)'
    : value >= 70
    ? 'var(--accent)'
    : value >= 40
    ? 'var(--warning)'
    : 'var(--danger)';
  return (
    <div className="flex w-28 flex-col items-center">
      <div className="text-[10px] uppercase tracking-wider text-[var(--text-faint)]">
        Score
      </div>
      <div className="relative mt-2">
        <svg width={SIZE} height={SIZE} className="-rotate-90">
          <circle
            cx={SIZE / 2}
            cy={SIZE / 2}
            r={R}
            strokeWidth={STROKE}
            fill="none"
            stroke="var(--border)"
          />
          <circle
            cx={SIZE / 2}
            cy={SIZE / 2}
            r={R}
            strokeWidth={STROKE}
            strokeLinecap="round"
            strokeDasharray={`${dash.toFixed(2)} ${C.toFixed(2)}`}
            fill="none"
            stroke={ringStroke}
            className="transition-[stroke-dasharray,stroke] duration-150"
          />
        </svg>
        <div className="pointer-events-none absolute inset-0 flex flex-col items-center justify-center">
          <span
            className={
              'text-2xl font-semibold tabular-nums ' +
              (active ? 'text-[var(--text)]' : 'text-[var(--text-faint)]')
            }
          >
            {active ? value : '—'}
          </span>
          <span className="text-[9px] uppercase tracking-wider text-[var(--text-faint)]">
            / 100
          </span>
        </div>
      </div>
      {active && score && (
        <div className="mt-1 grid w-full grid-cols-2 gap-1 text-center font-mono text-[9px] text-[var(--text-faint)]">
          <span>L {score.loudness}</span>
          <span>V {score.voicing}</span>
        </div>
      )}
    </div>
  );
}

function LiveTranscript({
  transcript,
  phase,
}: {
  transcript: { final: string; interim: string };
  phase: Phase;
}) {
  const recording = phase === 'recording';
  const empty = !transcript.final && !transcript.interim;
  return (
    <div className="card p-4">
      <div className="mb-2 flex items-center justify-between text-[10px] uppercase tracking-wider text-[var(--text-faint)]">
        <span>Live transcript</span>
        <span className="flex items-center gap-1.5">
          <span
            className={
              'inline-block h-1.5 w-1.5 rounded-full ' +
              (recording
                ? 'animate-pulse bg-[var(--accent)]'
                : phase === 'thinking'
                ? 'bg-[var(--warning)]'
                : 'bg-[var(--border-strong)]')
            }
          />
          {recording ? 'live' : phase === 'thinking' ? 'finalising' : 'captured'}
        </span>
      </div>
      <div className="min-h-[3.5rem] text-base leading-snug">
        {empty ? (
          <span className="text-sm italic text-[var(--text-faint)]">
            {recording
              ? 'Start speaking — your words will appear here.'
              : 'Nothing transcribed.'}
          </span>
        ) : (
          <>
            <span className="text-[var(--text)]">{transcript.final}</span>
            {transcript.interim && (
              <>
                {transcript.final && ' '}
                <span className="italic text-[var(--text-muted)]">
                  {transcript.interim}
                </span>
              </>
            )}
          </>
        )}
      </div>
    </div>
  );
}

function LoudnessMeter({
  loudFrac,
  targetFrac,
  dbfs,
  target_dbfs,
  onTarget,
  active,
}: {
  loudFrac: number;
  targetFrac: number;
  dbfs: number;
  target_dbfs: number;
  onTarget: boolean;
  active: boolean;
}) {
  // Visual ramp: dim → accent → warning when over-driven. The ramp is purely
  // the bar's gradient; the absolute value still comes from `loudFrac`.
  const barColor = !active
    ? 'var(--border-strong)'
    : loudFrac > 0.85
    ? 'var(--danger)'
    : onTarget
    ? 'var(--accent)'
    : 'var(--accent-strong)';
  return (
    <div>
      <div className="flex items-center justify-between text-[10px] uppercase tracking-wider text-[var(--text-faint)]">
        <span>Loudness</span>
        <span className="font-mono normal-case tracking-normal text-[var(--text-muted)]">
          {active && isFinite(dbfs) ? `${dbfs.toFixed(1)} dBFS` : '—'}
          <span className="ml-2 text-[var(--text-faint)]">
            target {target_dbfs.toFixed(0)}
          </span>
        </span>
      </div>
      <div className="relative mt-1.5 h-3 w-full overflow-hidden rounded-full bg-[var(--surface-inset)]">
        <div
          className="h-full rounded-full transition-[width] duration-75"
          style={{
            width: `${(loudFrac * 100).toFixed(1)}%`,
            background: barColor,
          }}
        />
        {/* Target marker: vertical line at targetFrac. */}
        <div
          className="pointer-events-none absolute top-[-2px] h-[calc(100%+4px)] w-0.5 bg-[var(--text)]/60"
          style={{ left: `calc(${(targetFrac * 100).toFixed(1)}% - 1px)` }}
          title={`target ${target_dbfs.toFixed(1)} dBFS`}
        />
      </div>
    </div>
  );
}

function SpectrumBars({ bands, active }: { bands: number[]; active: boolean }) {
  return (
    <div>
      <div className="text-[10px] uppercase tracking-wider text-[var(--text-faint)]">
        Spectrum (Hz)
      </div>
      <div className="mt-2 flex h-20 items-end gap-1.5">
        {bands.map((v, i) => {
          // Apply a gentle non-linear curve so quiet bands still register
          // visually. The raw values from the analyser cluster low when
          // speech is soft; sqrt() gives a friendlier shape.
          const height = active ? Math.sqrt(Math.max(0, Math.min(1, v))) : 0;
          return (
            <div
              key={i}
              className="flex flex-1 flex-col items-center justify-end gap-1"
            >
              <div className="flex h-full w-full items-end">
                <div
                  className="w-full rounded-sm transition-[height] duration-75"
                  style={{
                    height: `${(height * 100).toFixed(1)}%`,
                    background: active
                      ? 'var(--accent)'
                      : 'var(--surface-inset)',
                  }}
                />
              </div>
              <div className="text-[9px] text-[var(--text-faint)]">
                {BAND_LABELS[i]}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

function PitchGauge({
  pitchHz,
  active,
}: {
  pitchHz: number | null;
  active: boolean;
}) {
  // Fixed gauge range covering typical adult speech F0.
  const MIN = 80;
  const MAX = 400;
  const value = pitchHz != null ? Math.max(MIN, Math.min(MAX, pitchHz)) : null;
  const frac = value != null ? (value - MIN) / (MAX - MIN) : 0;
  return (
    <div className="flex w-24 flex-col items-stretch">
      <div className="text-[10px] uppercase tracking-wider text-[var(--text-faint)]">
        Pitch
      </div>
      <div className="relative mt-2 flex h-20 w-full items-end justify-center rounded-md border border-[var(--border)] bg-[var(--surface-inset)]">
        {/* Vertical pitch column (bottom = 80 Hz, top = 400 Hz). */}
        <div className="relative h-full w-2 rounded-full bg-[var(--surface-inset)]">
          {value != null && (
            <div
              className="absolute left-1/2 h-3 w-3 -translate-x-1/2 -translate-y-1/2 rounded-full bg-[var(--info)] shadow-[0_0_6px_2px_var(--info-soft)] transition-[bottom] duration-100"
              style={{ bottom: `${(frac * 100).toFixed(1)}%` }}
            />
          )}
        </div>
      </div>
      <div className="mt-1 text-center font-mono text-[11px] text-[var(--text-muted)]">
        {active && value != null ? `${Math.round(value)} Hz` : '—'}
      </div>
    </div>
  );
}

function AutoModeRow({
  autoMode,
  onToggle,
  phase,
}: {
  autoMode: boolean;
  onToggle: () => void;
  phase: Phase;
}) {
  // Toggling mid-recording would leave the VAD in an inconsistent state.
  // It's harmless but confusing — disable while a turn is in flight.
  const disabled = phase === 'recording' || phase === 'thinking';
  return (
    <div className="card-soft flex items-center justify-between px-4 py-2 text-xs text-[var(--text-muted)]">
      <span>
        Auto mic{' '}
        <span className="text-[var(--text-faint)]">
          (arms after the prompt, stops when you go quiet)
        </span>
      </span>
      <button
        onClick={onToggle}
        disabled={disabled}
        className={
          'relative inline-flex h-6 w-11 shrink-0 items-center rounded-full transition disabled:opacity-60 disabled:pointer-events-none ' +
          (autoMode ? 'bg-[var(--accent)]' : 'bg-[var(--border-strong)]')
        }
        aria-pressed={autoMode}
        aria-label="Toggle auto mic"
      >
        <span
          className={
            'inline-block h-4 w-4 transform rounded-full bg-white shadow transition ' +
            (autoMode ? 'translate-x-6' : 'translate-x-1')
          }
        />
      </button>
    </div>
  );
}

function CoachCard({ coach }: { coach: CoachMsg }) {
  const matched = coach.matched_prompt !== false;
  const actionStyles: Record<string, string> = {
    advance:
      'bg-[var(--accent-soft)] text-[var(--accent)] border-[var(--accent-ring)]',
    retry:
      'bg-[var(--warning-soft)] text-[var(--warning)] border-[var(--warning-soft)]',
    rest:
      'bg-[var(--info-soft)] text-[var(--info)] border-[var(--info-soft)]',
  };
  return (
    <div className="card p-5">
      <div className="flex items-center justify-between text-xs uppercase tracking-wider text-[var(--text-faint)]">
        <span>Coach</span>
        <span className="font-mono text-[10px] text-[var(--text-faint)]">
          {coach.latency_s.toFixed(1)}s
        </span>
      </div>

      {coach.heard && (
        <div className="mt-3 text-sm text-[var(--text-muted)]">
          <span className="text-[var(--text-faint)]">heard:</span>{' '}
          <span className="font-medium text-[var(--text)]">
            “{coach.heard}”
          </span>{' '}
          {!matched && (
            <span className="ml-1 rounded-full bg-[var(--danger-soft)] px-2 py-0.5 text-[10px] uppercase tracking-wider text-[var(--danger)]">
              mismatch
            </span>
          )}
        </div>
      )}

      {coach.ack && (
        <div className="mt-2 text-base text-[var(--text)]">{coach.ack}</div>
      )}
      {coach.feedback && (
        <div className="mt-1 text-base text-[var(--text-muted)]">
          {coach.feedback}
        </div>
      )}

      <div className="mt-4 flex flex-wrap items-center gap-2 text-[11px]">
        <span
          className={
            'rounded-full border px-2 py-0.5 uppercase tracking-wider ' +
            (actionStyles[coach.next_action] ??
              'border-[var(--border)] text-[var(--text-muted)]')
          }
        >
          → {coach.next_action}
        </span>
        {Object.entries(coach.metrics_observed)
          .filter(([k]) => k !== 'matched_prompt')
          .map(([k, v]) => (
            <span
              key={k}
              className={
                'rounded-full border px-2 py-0.5 ' +
                (v
                  ? 'border-[var(--accent-ring)] bg-[var(--accent-soft)] text-[var(--accent)]'
                  : 'border-[var(--border)] bg-[var(--surface-2)] text-[var(--text-faint)]')
              }
            >
              {k.replace(/_/g, ' ').replace(/ ok$/, '')} {v ? '✓' : '·'}
            </span>
          ))}
      </div>
    </div>
  );
}

function TransientError({ message }: { message: string }) {
  return (
    <div className="rounded-xl border border-[var(--danger-soft)] bg-[var(--danger-soft)] px-4 py-2 text-sm text-[var(--danger)]">
      {message}
    </div>
  );
}

// ---- ActionBar: single inline row of controls below the mic --------------
//
// One thin toolbar instead of a chunky "Commands" card at the bottom.
// Three quick-action chips for direct taps + one mic chip that captures
// PCM and ships it to Cactus Whisper → FunctionGemma 270M. The intent
// result (with Whisper + router latencies) appears as a small line
// underneath the bar so judges can still see the on-device routing
// happening in real time.

function ActionBar({
  commandPhase,
  intentResult,
  onRepeat,
  onSkip,
  onRest,
  onStartCommand,
  onCancelCommand,
  disabled,
}: {
  commandPhase: 'idle' | 'listening' | 'thinking';
  intentResult: IntentResultMsg | null;
  onRepeat: () => void;
  onSkip: () => void;
  onRest: () => void;
  onStartCommand: () => void;
  onCancelCommand: () => void;
  disabled: boolean;
}) {
  const isCmdActive = commandPhase !== 'idle';
  const cmdDisabled = disabled || commandPhase === 'thinking';
  const cmdTone =
    commandPhase === 'listening'
      ? 'border-[var(--danger-soft)] bg-[var(--danger-soft)] text-[var(--danger)] hover:bg-[var(--danger-soft)]'
      : commandPhase === 'thinking'
      ? 'border-[var(--border)] bg-[var(--surface-2)] text-[var(--text-faint)]'
      : 'border-[var(--violet-soft)] bg-[var(--violet-soft)] text-[var(--violet)] hover:bg-[var(--surface)]';
  const cmdLabel =
    commandPhase === 'listening'
      ? 'Listening — tap to stop'
      : commandPhase === 'thinking'
      ? 'Routing…'
      : 'Say a command';

  return (
    <div className="flex flex-col gap-2">
      <div className="flex flex-wrap items-center justify-center gap-2">
        <ChipButton
          label="↺ Repeat"
          onClick={onRepeat}
          disabled={cmdDisabled}
          tone="neutral"
        />
        <ChipButton
          label="⏭ Skip"
          onClick={onSkip}
          disabled={cmdDisabled}
          tone="neutral"
        />
        <ChipButton
          label="⏹ End"
          onClick={onRest}
          disabled={commandPhase === 'thinking'}
          tone="danger"
        />
        <span className="mx-1 hidden text-[var(--text-faint)] sm:inline">·</span>
        <button
          type="button"
          onClick={isCmdActive ? onCancelCommand : onStartCommand}
          disabled={cmdDisabled}
          title="Voice command via Cactus Whisper + FunctionGemma 270M"
          className={
            'rounded-full border px-3 py-1.5 text-sm transition disabled:opacity-60 disabled:pointer-events-none ' +
            cmdTone
          }
        >
          {commandPhase === 'listening' && (
            <span className="mr-1.5 inline-block h-1.5 w-1.5 animate-pulse rounded-full bg-[var(--danger)]" />
          )}
          {cmdLabel}
        </button>
      </div>
      {intentResult && <IntentResultChip result={intentResult} />}
    </div>
  );
}

function ChipButton({
  label,
  onClick,
  disabled,
  tone,
}: {
  label: string;
  onClick: () => void;
  disabled: boolean;
  tone: 'neutral' | 'danger';
}) {
  // "End" is destructive enough that we tone it differently so the
  // user (often elderly, often shaky) doesn't tap it by mistake.
  const cls =
    tone === 'danger'
      ? 'border-[var(--danger-soft)] bg-[var(--danger-soft)] text-[var(--danger)] hover:opacity-90'
      : 'border-[var(--border)] bg-[var(--surface-2)] text-[var(--text)] hover:bg-[var(--surface)]';
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      className={
        'rounded-full border px-3 py-1.5 text-sm transition disabled:opacity-60 disabled:pointer-events-none ' +
        cls
      }
    >
      {label}
    </button>
  );
}

function IntentResultChip({ result }: { result: IntentResultMsg }) {
  const isNone = result.action === 'none';
  const sourceLabel =
    result.source === 'functiongemma'
      ? 'FunctionGemma 270M'
      : 'Regex fallback';
  // Friendly labels for the routed action — matches the button copy.
  const actionLabel: Record<IntentResultMsg['action'], string> = {
    skip: 'Skip drill',
    rest: 'End session',
    repeat_prompt: 'Repeat prompt',
    none: 'No match',
  };
  const tone = isNone
    ? 'border-[var(--border)] bg-[var(--surface-2)] text-[var(--text-faint)]'
    : 'border-[var(--accent-ring)] bg-[var(--accent-soft)] text-[var(--accent)]';
  const usedWhisper = result.transcribe_source === 'whisper';
  return (
    <div className="mt-3 flex flex-col gap-1.5 text-[11px]">
      {/* Transcript line — shows what STT (or the user's keyboard)
          actually produced, separately from the routed action. */}
      {result.transcript && (
        <div className="font-mono text-[var(--text-muted)]">
          <span className="text-[var(--text-faint)]">heard:</span> "
          {result.transcript}"
        </div>
      )}
      <div className="flex flex-wrap items-center gap-2">
        <span
          className={
            'rounded-full border px-2 py-0.5 uppercase tracking-wider ' + tone
          }
        >
          → {actionLabel[result.action]}
        </span>
        <span className="font-mono text-[var(--text-faint)]">
          {Math.round(result.confidence * 100)}% confidence
        </span>
        {usedWhisper && result.transcribe_latency_ms !== null && (
          <span className="font-mono text-[var(--text-faint)]">
            Whisper {result.transcribe_latency_ms} ms
          </span>
        )}
        <span className="font-mono text-[var(--text-faint)]">
          {sourceLabel} {result.latency_ms} ms
        </span>
        {!result.intent_model_loaded && result.source === 'heuristic' && (
          <span className="text-[10px] italic text-[var(--warning)]">
            (FunctionGemma still warming up)
          </span>
        )}
      </div>
    </div>
  );
}

function SummaryCard({
  summary,
  onRestart,
}: {
  summary: SummaryMsg['summary'];
  onRestart: () => void;
}) {
  return (
    <div className="card p-8 text-center">
      <div className="text-xs uppercase tracking-wider text-[var(--text-faint)]">
        Session summary
      </div>
      <div className="mt-3 grid grid-cols-3 gap-4 text-center">
        <Stat
          label="Drills completed"
          value={`${summary.advanced} / ${summary.total}`}
        />
        <Stat label="Retries" value={summary.retries} />
        <Stat
          label="Avg loudness"
          value={summary.avg_dbfs == null ? '—' : `${summary.avg_dbfs} dBFS`}
        />
      </div>
      {summary.json_failures > 0 && (
        <div className="mt-3 text-xs text-[var(--warning)]">
          {summary.json_failures} turn{summary.json_failures === 1 ? '' : 's'}{' '}
          had model parse errors.
        </div>
      )}
      {summary.rest_called && (
        <div className="mt-3 text-xs text-[var(--info)]">
          Ended early (rest requested).
        </div>
      )}
      <div className="mt-5 break-all text-[10px] text-[var(--text-faint)]">
        log: {summary.session_log}
      </div>
      <button
        onClick={onRestart}
        className="mt-6 rounded-full bg-[var(--accent)] px-6 py-2 text-sm font-medium text-[var(--accent-fg)] transition hover:bg-[var(--accent-strong)]"
      >
        Run another session
      </button>
    </div>
  );
}

function Stat({ label, value }: { label: string; value: number | string }) {
  return (
    <div className="card-soft p-4">
      <div className="text-2xl font-semibold text-[var(--text)]">{value}</div>
      <div className="mt-1 text-xs text-[var(--text-faint)]">{label}</div>
    </div>
  );
}

function ErrorCard({ message }: { message: string }) {
  return (
    <div className="rounded-2xl border border-[var(--danger-soft)] bg-[var(--danger-soft)] p-6">
      <div className="text-xs uppercase tracking-wider text-[var(--danger)]">
        Cannot continue
      </div>
      <div className="mt-2 text-sm text-[var(--text)]">{message}</div>
      <div className="mt-3 text-xs text-[var(--text-muted)]">
        Make sure the backend started cleanly. Run <code>./run-web</code> from
        the repo root.
      </div>
    </div>
  );
}

// ---- helpers --------------------------------------------------------------

function buildDrillTTS(drill: DrillMsg): string {
  // For instruction-only phases (warmup, glide), `note` is empty and the
  // `prompt` IS the full instruction — just speak it. For phases with
  // explicit content (counting, main_task), speak the cue then the
  // expected utterance.
  const note = (drill.note ?? '').trim();
  const prompt = (drill.prompt ?? '').trim();
  if (!note) return prompt;
  return `${note.replace(/[.!?,;: ]+$/, '')}. Now: ${prompt}.`;
}

function joinForSpeech(ack: string, feedback: string): string {
  const trim = (s: string) => s.replace(/[\s.!?,;:]+$/, '');
  const parts = [ack, feedback].map(trim).filter(Boolean);
  if (!parts.length) return '';
  return parts.join('. ') + '.';
}
