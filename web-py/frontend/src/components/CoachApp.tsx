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
  testSpeech,
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

export default function CoachApp() {
  const [phase, setPhase] = useState<Phase>('connecting');
  const [drill, setDrill] = useState<DrillMsg | null>(null);
  const [metrics, setMetrics] = useState<MetricsMsg | null>(null);
  const [coach, setCoach] = useState<CoachMsg | null>(null);
  const [summary, setSummary] = useState<SummaryMsg['summary'] | null>(null);
  const [errMsg, setErrMsg] = useState<string | null>(null);
  const [transientError, setTransientError] = useState<string | null>(null);
  const [wsState, setWsState] = useState<string>('CONNECTING');
  const [autoMode, setAutoMode] = useState(true);
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
        setWsState('CLOSED');
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
      .then(() => setWsState('OPEN'))
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
        setMetrics(msg);
        return;
      case 'thinking':
        setPhase('thinking');
        return;
      case 'coach': {
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
        setSummary(msg.summary);
        setPhase('done');
        cancelSpeech();
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
  const restSession = () => wsRef.current?.send({ type: 'command', action: 'rest' });

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

  // ---- Render ------------------------------------------------------------

  const analysis: AudioAnalysis | null =
    phase === 'recording' ? recRef.current?.analysis() ?? null : null;
  const liveLevel = analysis?.rms != null ? Math.min(1, analysis.rms * 4) : 0;
  const liveScore = computeLiveScore(analysis, drill?.target_dbfs ?? -25);

  return (
    <main className="mx-auto flex min-h-screen w-full max-w-2xl flex-col px-6 py-10">
      <Header wsState={wsState} />

      <section className="flex flex-1 flex-col gap-4">
        {!isTTSAvailable() && (
          <div className="rounded-xl border border-amber-500/40 bg-amber-500/10 px-4 py-2 text-xs text-amber-200">
            This browser doesn't expose <code>speechSynthesis</code>. The coach
            will be silent — try Chrome, Edge, or Safari.
          </div>
        )}
        {phase === 'connecting' && (
          <>
            <StatusCard label="Connecting to local coach…" spinner />
            <TestVoiceRow />
          </>
        )}
        {phase === 'loading' && (
          <>
            <StatusCard
              label="Loading Gemma 4 on your machine…"
              hint="One-time, takes ~6 seconds. The model never leaves your device."
              spinner
            />
            <TestVoiceRow />
          </>
        )}
        {phase === 'ready' && (
          <>
            <StartCard onStart={startSession} />
            <TestVoiceRow />
          </>
        )}

        {(phase === 'drill' ||
          phase === 'recording' ||
          phase === 'thinking' ||
          phase === 'feedback') &&
          drill && (
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

function Header({ wsState }: { wsState: string }) {
  const dotColor =
    wsState === 'OPEN'
      ? 'bg-emerald-400'
      : wsState === 'CONNECTING'
      ? 'bg-amber-400'
      : 'bg-rose-500';
  return (
    <header className="mb-8 flex items-start justify-between">
      <div>
        <h1 className="text-3xl font-semibold tracking-tight">Voice Coach</h1>
        <p className="mt-1 text-sm text-zinc-400">
          On-device speech practice. Powered by Gemma 4 + Cactus.
        </p>
      </div>
      <div
        className="mt-2 flex items-center gap-2 text-[10px] uppercase tracking-wider text-zinc-500"
        title={`WebSocket: ${wsState}`}
      >
        <span className={`inline-block h-2 w-2 rounded-full ${dotColor}`} />
        {wsState}
      </div>
    </header>
  );
}

function Footer() {
  return (
    <footer className="mt-8 border-t border-zinc-800 pt-4 text-xs text-zinc-500">
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
    <div className="rounded-2xl border border-zinc-800 bg-zinc-900/60 p-8 text-center">
      <div className="text-lg">{label}</div>
      {hint && <div className="mt-2 text-sm text-zinc-500">{hint}</div>}
      {spinner && (
        <div className="mt-6 inline-block h-6 w-6 animate-spin rounded-full border-2 border-zinc-700 border-t-zinc-200" />
      )}
    </div>
  );
}

function StartCard({ onStart }: { onStart: () => void }) {
  return (
    <div className="rounded-2xl border border-zinc-800 bg-zinc-900/60 p-10 text-center">
      <h2 className="text-2xl font-semibold">Ready when you are.</h2>
      <p className="mx-auto mt-3 max-w-md text-zinc-400">
        Ten short drills: 3 vowel warm-ups, 5 phrases, 2 conversation prompts.
        Speak when prompted; the coach will respond with feedback.
      </p>
      <button
        onClick={onStart}
        className="mt-8 rounded-full bg-emerald-500 px-8 py-3 font-medium text-zinc-950 transition hover:bg-emerald-400 active:scale-[0.98]"
      >
        Start session
      </button>
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
      <StageIndicator drill={drill} />
      <PromptCard drill={drill} onRepeat={onRepeat} />
      <MicButton
        phase={phase}
        level={level}
        autoMode={autoMode}
        speaking={analysis?.speaking ?? false}
        onStart={onStartRec}
        onStop={onStopRec}
      />
      {/* One thin action bar inline with the mic — no separate panel
          at the bottom. Voice commands invoke the same actions as
          the pills; chip below shows the routed verdict + latencies. */}
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
      <LiveAnalyzer
        analysis={analysis}
        target_dbfs={drill.target_dbfs}
        phase={phase}
        liveScore={liveScore}
      />
      {showTranscript && (
        <LiveTranscript transcript={transcript} phase={phase} />
      )}
      <AutoModeRow autoMode={autoMode} onToggle={onToggleAuto} phase={phase} />
      {metrics && <MetricsLine metrics={metrics} target={drill.target_dbfs} />}
      {coach && <CoachCard coach={coach} />}
      {transientError && <TransientError message={transientError} />}
    </div>
  );
}

function StageIndicator({ drill }: { drill: DrillMsg }) {
  return (
    <div className="flex flex-col gap-2 text-xs">
      {drill.exercise_name && (
        <div className="flex items-center justify-between text-[11px] uppercase tracking-wider text-zinc-500">
          <span>
            {drill.category_name && (
              <>
                <span className="text-zinc-600">{drill.category_name}</span>
                <span className="mx-1 text-zinc-700">›</span>
              </>
            )}
            <span className="text-zinc-300">{drill.exercise_name}</span>
          </span>
          <span>
            step {drill.position + 1} / {drill.total}
          </span>
        </div>
      )}
      <div className="flex gap-1">
        {STAGES.map((s) => {
          const active = drill.stage === s;
          return (
            <span
              key={s}
              className={
                'rounded-full border px-3 py-1 transition ' +
                (active
                  ? 'border-emerald-400/70 bg-emerald-400/10 text-emerald-300'
                  : 'border-zinc-800 bg-zinc-900/40 text-zinc-500')
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
    <div className="rounded-2xl border border-zinc-800 bg-zinc-900/60 p-6">
      <div className="text-xs uppercase tracking-wider text-zinc-500">
        {isInstructionOnly ? 'Phase cue' : 'Say this'}
      </div>
      {isInstructionOnly ? (
        <div className="mt-2 text-2xl font-medium leading-snug">
          {drill.prompt}
        </div>
      ) : (
        <>
          <div className="mt-2 text-3xl font-medium leading-tight">
            “{drill.prompt}”
          </div>
          <div className="mt-3 text-sm text-zinc-400">{drill.note}</div>
        </>
      )}
      {drill.focus && (
        <div className="mt-3 text-xs text-emerald-300/80">
          focus: {drill.focus}
        </div>
      )}
      {(reps > 0 || dur > 0) && (
        <div className="mt-1 font-mono text-[11px] text-zinc-500">
          {reps > 1 && `target ${reps} reps`}
          {reps > 1 && dur > 0 && ' · '}
          {dur > 0 && `~${dur}s each`}
        </div>
      )}
      <button
        onClick={onRepeat}
        className="mt-4 text-xs text-zinc-400 underline-offset-4 hover:text-zinc-200 hover:underline"
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

  const ringScale = isRecording ? 1 + level * 0.6 : 1;
  const ringTone = isRecording
    ? speaking
      ? 'bg-emerald-400/25'
      : 'bg-rose-500/20'
    : 'bg-transparent';

  return (
    <div className="my-4 flex flex-col items-center gap-2">
      <div className="relative">
        {isRecording && (
          <div
            className={`absolute inset-0 rounded-full transition-transform duration-100 ${ringTone}`}
            style={{ transform: `scale(${ringScale})` }}
          />
        )}
        <button
          onClick={isRecording ? onStop : onStart}
          disabled={disabled}
          className={
            'relative flex h-24 w-24 items-center justify-center rounded-full text-3xl font-semibold transition active:scale-95 ' +
            (isRecording
              ? speaking
                ? 'bg-emerald-500 text-zinc-950 shadow-lg shadow-emerald-500/40'
                : 'bg-rose-500 text-white shadow-lg shadow-rose-500/40'
              : isThinking
              ? 'bg-zinc-700 text-zinc-400'
              : 'bg-emerald-500 text-zinc-950 shadow-lg shadow-emerald-500/30 hover:bg-emerald-400')
          }
          aria-label={isRecording ? 'Stop recording' : 'Start recording'}
        >
          {isThinking ? (
            <span className="inline-block h-6 w-6 animate-spin rounded-full border-2 border-zinc-500 border-t-zinc-200" />
          ) : isRecording ? (
            '■'
          ) : (
            '🎤'
          )}
        </button>
      </div>
      <div className="text-xs text-zinc-500">
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
    <div className="rounded-xl border border-zinc-800 bg-zinc-900/40 px-4 py-2 font-mono text-xs text-zinc-400">
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
    <div className="rounded-2xl border border-zinc-800 bg-zinc-900/50 p-4">
      <div className="mb-3 flex items-center justify-between text-[10px] uppercase tracking-wider text-zinc-500">
        <span>Live signal</span>
        <span className="flex items-center gap-1.5">
          <span
            className={
              'inline-block h-2 w-2 rounded-full transition ' +
              (speaking
                ? 'bg-emerald-400 shadow-[0_0_6px_2px_rgba(52,211,153,0.6)]'
                : recording
                ? 'bg-zinc-600'
                : 'bg-zinc-700')
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
  const ringColor =
    !active
      ? 'stroke-zinc-700'
      : value >= 70
      ? 'stroke-emerald-400'
      : value >= 40
      ? 'stroke-amber-400'
      : 'stroke-rose-400';
  return (
    <div className="flex w-28 flex-col items-center">
      <div className="text-[10px] uppercase tracking-wider text-zinc-500">
        Score
      </div>
      <div className="relative mt-2">
        <svg width={SIZE} height={SIZE} className="-rotate-90">
          <circle
            cx={SIZE / 2}
            cy={SIZE / 2}
            r={R}
            strokeWidth={STROKE}
            className="fill-none stroke-zinc-800"
          />
          <circle
            cx={SIZE / 2}
            cy={SIZE / 2}
            r={R}
            strokeWidth={STROKE}
            strokeLinecap="round"
            strokeDasharray={`${dash.toFixed(2)} ${C.toFixed(2)}`}
            className={`fill-none transition-[stroke-dasharray,stroke] duration-150 ${ringColor}`}
          />
        </svg>
        <div className="pointer-events-none absolute inset-0 flex flex-col items-center justify-center">
          <span
            className={
              'text-2xl font-semibold tabular-nums ' +
              (active ? 'text-zinc-100' : 'text-zinc-600')
            }
          >
            {active ? value : '—'}
          </span>
          <span className="text-[9px] uppercase tracking-wider text-zinc-500">
            / 100
          </span>
        </div>
      </div>
      {active && score && (
        <div className="mt-1 grid w-full grid-cols-2 gap-1 text-center font-mono text-[9px] text-zinc-500">
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
    <div className="rounded-2xl border border-zinc-800 bg-zinc-900/50 p-4">
      <div className="mb-2 flex items-center justify-between text-[10px] uppercase tracking-wider text-zinc-500">
        <span>Live transcript</span>
        <span className="flex items-center gap-1.5">
          <span
            className={
              'inline-block h-1.5 w-1.5 rounded-full ' +
              (recording
                ? 'animate-pulse bg-emerald-400'
                : phase === 'thinking'
                ? 'bg-amber-400'
                : 'bg-zinc-600')
            }
          />
          {recording ? 'live' : phase === 'thinking' ? 'finalising' : 'captured'}
        </span>
      </div>
      <div className="min-h-[3.5rem] text-base leading-snug">
        {empty ? (
          <span className="text-sm italic text-zinc-600">
            {recording
              ? 'Start speaking — your words will appear here.'
              : 'Nothing transcribed.'}
          </span>
        ) : (
          <>
            <span className="text-zinc-100">{transcript.final}</span>
            {transcript.interim && (
              <>
                {transcript.final && ' '}
                <span className="italic text-zinc-400">
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
  // Visual ramp: dim → emerald → amber when over-driven. The ramp is purely
  // the bar's gradient; the absolute value still comes from `loudFrac`.
  const barTone = !active
    ? 'from-zinc-700 to-zinc-700'
    : loudFrac > 0.85
    ? 'from-amber-500 to-rose-500'
    : onTarget
    ? 'from-emerald-500 to-emerald-300'
    : 'from-emerald-600/80 to-emerald-400/80';
  return (
    <div>
      <div className="flex items-center justify-between text-[10px] uppercase tracking-wider text-zinc-500">
        <span>Loudness</span>
        <span className="font-mono normal-case tracking-normal text-zinc-400">
          {active && isFinite(dbfs) ? `${dbfs.toFixed(1)} dBFS` : '—'}
          <span className="ml-2 text-zinc-600">
            target {target_dbfs.toFixed(0)}
          </span>
        </span>
      </div>
      <div className="relative mt-1.5 h-3 w-full overflow-hidden rounded-full bg-zinc-800">
        <div
          className={`h-full rounded-full bg-gradient-to-r transition-[width] duration-75 ${barTone}`}
          style={{ width: `${(loudFrac * 100).toFixed(1)}%` }}
        />
        {/* Target marker: vertical line at targetFrac. */}
        <div
          className="pointer-events-none absolute top-[-2px] h-[calc(100%+4px)] w-0.5 bg-zinc-300/80"
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
      <div className="text-[10px] uppercase tracking-wider text-zinc-500">
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
                  className={
                    'w-full rounded-sm transition-[height] duration-75 ' +
                    (active
                      ? 'bg-gradient-to-t from-emerald-500 to-emerald-300'
                      : 'bg-zinc-800')
                  }
                  style={{ height: `${(height * 100).toFixed(1)}%` }}
                />
              </div>
              <div className="text-[9px] text-zinc-600">{BAND_LABELS[i]}</div>
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
      <div className="text-[10px] uppercase tracking-wider text-zinc-500">
        Pitch
      </div>
      <div className="relative mt-2 flex h-20 w-full items-end justify-center rounded-md border border-zinc-800 bg-zinc-950/40">
        {/* Vertical pitch column (bottom = 80 Hz, top = 400 Hz). */}
        <div className="relative h-full w-2 rounded-full bg-zinc-800/70">
          {value != null && (
            <div
              className="absolute left-1/2 h-3 w-3 -translate-x-1/2 -translate-y-1/2 rounded-full bg-sky-400 shadow-[0_0_6px_2px_rgba(56,189,248,0.5)] transition-[bottom] duration-100"
              style={{ bottom: `${(frac * 100).toFixed(1)}%` }}
            />
          )}
        </div>
      </div>
      <div className="mt-1 text-center font-mono text-[11px] text-zinc-400">
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
    <div className="flex items-center justify-between rounded-xl border border-zinc-800 bg-zinc-900/40 px-4 py-2 text-xs text-zinc-400">
      <span>
        Auto mic{' '}
        <span className="text-zinc-600">
          (arms after the prompt, stops when you go quiet)
        </span>
      </span>
      <button
        onClick={onToggle}
        disabled={disabled}
        className={
          'relative inline-flex h-6 w-11 shrink-0 items-center rounded-full transition disabled:opacity-40 ' +
          (autoMode ? 'bg-emerald-500' : 'bg-zinc-700')
        }
        aria-pressed={autoMode}
        aria-label="Toggle auto mic"
      >
        <span
          className={
            'inline-block h-4 w-4 transform rounded-full bg-white transition ' +
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
    advance: 'bg-emerald-500/15 text-emerald-300 border-emerald-500/30',
    retry: 'bg-amber-500/15 text-amber-300 border-amber-500/30',
    rest: 'bg-sky-500/15 text-sky-300 border-sky-500/30',
  };
  return (
    <div className="rounded-2xl border border-zinc-800 bg-zinc-900/70 p-5">
      <div className="flex items-center justify-between text-xs uppercase tracking-wider text-zinc-500">
        <span>Coach</span>
        <span className="font-mono text-[10px] text-zinc-600">
          {coach.latency_s.toFixed(1)}s
        </span>
      </div>

      {coach.heard && (
        <div className="mt-3 text-sm">
          <span className="text-zinc-500">heard:</span>{' '}
          <span className="font-medium">“{coach.heard}”</span>{' '}
          {!matched && (
            <span className="ml-1 rounded-full bg-rose-500/15 px-2 py-0.5 text-[10px] uppercase tracking-wider text-rose-300">
              mismatch
            </span>
          )}
        </div>
      )}

      {coach.ack && (
        <div className="mt-2 text-base text-zinc-100">{coach.ack}</div>
      )}
      {coach.feedback && (
        <div className="mt-1 text-base text-zinc-300">{coach.feedback}</div>
      )}

      <div className="mt-4 flex flex-wrap items-center gap-2 text-[11px]">
        <span
          className={
            'rounded-full border px-2 py-0.5 uppercase tracking-wider ' +
            (actionStyles[coach.next_action] ?? 'border-zinc-700 text-zinc-400')
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
                  ? 'border-emerald-500/30 bg-emerald-500/10 text-emerald-300'
                  : 'border-zinc-700 bg-zinc-800/50 text-zinc-500')
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
    <div className="rounded-xl border border-rose-500/40 bg-rose-500/10 px-4 py-2 text-sm text-rose-200">
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
      ? 'border-rose-500/40 bg-rose-500/15 text-rose-200 hover:bg-rose-500/25'
      : commandPhase === 'thinking'
      ? 'border-zinc-700 bg-zinc-800/60 text-zinc-400'
      : 'border-sky-500/40 bg-sky-500/10 text-sky-200 hover:bg-sky-500/20';
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
        <span className="mx-1 hidden text-zinc-700 sm:inline">·</span>
        <button
          type="button"
          onClick={isCmdActive ? onCancelCommand : onStartCommand}
          disabled={cmdDisabled}
          title="Voice command via Cactus Whisper + FunctionGemma 270M"
          className={
            'rounded-full border px-3 py-1.5 text-sm transition disabled:opacity-40 ' +
            cmdTone
          }
        >
          {commandPhase === 'listening' && (
            <span className="mr-1.5 inline-block h-1.5 w-1.5 animate-pulse rounded-full bg-rose-300" />
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
      ? 'border-rose-500/30 bg-rose-500/10 text-rose-200 hover:bg-rose-500/20'
      : 'border-zinc-700 bg-zinc-900/50 text-zinc-200 hover:bg-zinc-800/70';
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      className={
        'rounded-full border px-3 py-1.5 text-sm transition disabled:opacity-40 ' +
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
    ? 'border-zinc-700 bg-zinc-800/60 text-zinc-400'
    : 'border-emerald-500/40 bg-emerald-500/15 text-emerald-200';
  const usedWhisper = result.transcribe_source === 'whisper';
  return (
    <div className="mt-3 flex flex-col gap-1.5 text-[11px]">
      {/* Transcript line — shows what STT (or the user's keyboard)
          actually produced, separately from the routed action. */}
      {result.transcript && (
        <div className="font-mono text-zinc-400">
          <span className="text-zinc-600">heard:</span> "{result.transcript}"
        </div>
      )}
      <div className="flex flex-wrap items-center gap-2">
        <span
          className={'rounded-full border px-2 py-0.5 uppercase tracking-wider ' + tone}
        >
          → {actionLabel[result.action]}
        </span>
        <span className="font-mono text-zinc-500">
          {Math.round(result.confidence * 100)}% confidence
        </span>
        {usedWhisper && result.transcribe_latency_ms !== null && (
          <span className="font-mono text-zinc-500">
            Whisper {result.transcribe_latency_ms} ms
          </span>
        )}
        <span className="font-mono text-zinc-500">
          {sourceLabel} {result.latency_ms} ms
        </span>
        {!result.intent_model_loaded && result.source === 'heuristic' && (
          <span className="text-[10px] italic text-amber-400/80">
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
    <div className="rounded-2xl border border-zinc-800 bg-zinc-900/60 p-8 text-center">
      <div className="text-xs uppercase tracking-wider text-zinc-500">
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
        <div className="mt-3 text-xs text-amber-400">
          {summary.json_failures} turn{summary.json_failures === 1 ? '' : 's'}{' '}
          had model parse errors.
        </div>
      )}
      {summary.rest_called && (
        <div className="mt-3 text-xs text-sky-400">
          Ended early (rest requested).
        </div>
      )}
      <div className="mt-5 break-all text-[10px] text-zinc-600">
        log: {summary.session_log}
      </div>
      <button
        onClick={onRestart}
        className="mt-6 rounded-full bg-emerald-500 px-6 py-2 text-sm font-medium text-zinc-950 transition hover:bg-emerald-400"
      >
        Run another session
      </button>
    </div>
  );
}

function Stat({ label, value }: { label: string; value: number | string }) {
  return (
    <div className="rounded-xl border border-zinc-800 bg-zinc-900/40 p-4">
      <div className="text-2xl font-semibold">{value}</div>
      <div className="mt-1 text-xs text-zinc-500">{label}</div>
    </div>
  );
}

function ErrorCard({ message }: { message: string }) {
  return (
    <div className="rounded-2xl border border-rose-500/40 bg-rose-500/10 p-6">
      <div className="text-xs uppercase tracking-wider text-rose-300">
        Cannot continue
      </div>
      <div className="mt-2 text-sm text-rose-100">{message}</div>
      <div className="mt-3 text-xs text-rose-200/80">
        Make sure the backend started cleanly. Run <code>./run-web</code> from
        the repo root.
      </div>
    </div>
  );
}

function TestVoiceRow() {
  const [tested, setTested] = useState(false);
  const [ok, setOk] = useState<boolean | null>(null);
  const onTest = () => {
    const fired = testSpeech();
    setTested(true);
    setOk(fired);
  };
  return (
    <div className="flex items-center justify-between rounded-xl border border-zinc-800 bg-zinc-900/40 px-4 py-2 text-xs text-zinc-400">
      <span>
        {!tested && 'Quick check: does your browser play voice?'}
        {tested && ok && (
          <>
            ✓ Test queued. If you didn't hear "Voice test, can you hear me?",
            check System Output volume and the page's audio permission.
          </>
        )}
        {tested && ok === false && (
          <span className="text-rose-300">
            Could not start a TTS utterance. Try Chrome/Edge/Safari.
          </span>
        )}
      </span>
      <button
        onClick={onTest}
        className="ml-3 shrink-0 rounded-full border border-zinc-700 px-3 py-1 text-zinc-300 transition hover:bg-zinc-800/60"
      >
        🔊 Test voice
      </button>
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
