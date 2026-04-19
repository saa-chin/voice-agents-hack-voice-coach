import { useEffect, useRef, useState } from 'react';
import {
  cancelSpeech,
  createRecorder,
  int16ToBase64,
  isTTSAvailable,
  primeTTS,
  speak,
  testSpeech,
  type RecorderHandle,
} from '../lib/audio';
import {
  connect,
  type CoachMsg,
  type Connection,
  type DrillMsg,
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

const STAGES = ['warmup', 'phrase', 'conversation'] as const;
type Stage = (typeof STAGES)[number];

const STAGE_LABEL: Record<Stage, string> = {
  warmup: 'Warm-up',
  phrase: 'Phrases',
  conversation: 'Conversation',
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
  const [, forceTick] = useState(0);

  const wsRef = useRef<Connection | null>(null);
  const recRef = useRef<RecorderHandle | null>(null);
  const levelTimer = useRef<number | null>(null);
  // Track current phase via ref so async WS handlers don't read stale state.
  const phaseRef = useRef<Phase>('connecting');
  // StrictMode in dev runs effects twice. Without a guard we open two WS
  // connections per mount, which the server logs show happening.
  const wsInitialized = useRef(false);

  useEffect(() => {
    phaseRef.current = phase;
  }, [phase]);

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
      recRef.current?.cancel();
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
        setTransientError(null);
        setPhase('drill');
        speak(`Please say: ${msg.prompt}`);
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
        const spoken = joinForSpeech(msg.ack, msg.feedback);
        if (spoken) speak(spoken);
        return;
      }
      case 'advance':
      case 'retry':
      case 'rest':
        // Visual cue handled by next 'drill' or 'session_done' message.
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
    setTransientError(null);
    try {
      if (!recRef.current) {
        recRef.current = await createRecorder();
      }
      await recRef.current.start();
      setPhase('recording');
      // Force a re-render every 80ms so the live level meter animates.
      levelTimer.current = window.setInterval(
        () => forceTick((t) => t + 1),
        80,
      ) as unknown as number;
    } catch (exc: any) {
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
    try {
      const { pcm } = await recRef.current.stop();
      recRef.current = null;
      wsRef.current?.send({
        type: 'audio',
        pcm_b64: int16ToBase64(pcm),
        sample_rate: 16000,
      });
    } catch (exc: any) {
      setTransientError(`Capture error: ${exc?.message ?? exc}`);
      setPhase('drill');
    }
  };

  const repeatPrompt = () => {
    if (drill) speak(`Please say: ${drill.prompt}`);
  };
  const skipDrill = () => wsRef.current?.send({ type: 'command', action: 'skip' });
  const restSession = () => wsRef.current?.send({ type: 'command', action: 'rest' });

  // ---- Render ------------------------------------------------------------

  const liveLevel = phase === 'recording' ? recRef.current?.level() ?? 0 : 0;

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
              transientError={transientError}
              onStartRec={startRecording}
              onStopRec={stopRecording}
              onRepeat={repeatPrompt}
              onSkip={skipDrill}
              onRest={restSession}
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
  transientError,
  onStartRec,
  onStopRec,
  onRepeat,
  onSkip,
  onRest,
}: {
  drill: DrillMsg;
  metrics: MetricsMsg | null;
  coach: CoachMsg | null;
  phase: Phase;
  level: number;
  transientError: string | null;
  onStartRec: () => void;
  onStopRec: () => void;
  onRepeat: () => void;
  onSkip: () => void;
  onRest: () => void;
}) {
  return (
    <div className="flex flex-col gap-4">
      <StageIndicator drill={drill} />
      <PromptCard drill={drill} onRepeat={onRepeat} />
      <MicButton
        phase={phase}
        level={level}
        onStart={onStartRec}
        onStop={onStopRec}
      />
      {metrics && <MetricsLine metrics={metrics} target={drill.target_dbfs} />}
      {coach && <CoachCard coach={coach} />}
      {transientError && <TransientError message={transientError} />}
      <ActionsRow
        onSkip={onSkip}
        onRest={onRest}
        disableSkip={phase === 'recording' || phase === 'thinking'}
      />
    </div>
  );
}

function StageIndicator({ drill }: { drill: DrillMsg }) {
  return (
    <div className="flex items-center justify-between text-xs">
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
      <span className="text-zinc-500">
        {drill.position + 1} of {drill.total}
      </span>
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
  return (
    <div className="rounded-2xl border border-zinc-800 bg-zinc-900/60 p-6">
      <div className="text-xs uppercase tracking-wider text-zinc-500">
        Say this
      </div>
      <div className="mt-2 text-3xl font-medium leading-tight">
        “{drill.prompt}”
      </div>
      {drill.note && (
        <div className="mt-3 text-sm text-zinc-400">{drill.note}</div>
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
  onStart,
  onStop,
}: {
  phase: Phase;
  level: number;
  onStart: () => void;
  onStop: () => void;
}) {
  const isRecording = phase === 'recording';
  const isThinking = phase === 'thinking';
  const disabled = isThinking;

  const ringScale = isRecording ? 1 + level * 0.6 : 1;

  return (
    <div className="my-4 flex flex-col items-center gap-2">
      <div className="relative">
        {isRecording && (
          <div
            className="absolute inset-0 rounded-full bg-rose-500/20 transition-transform duration-100"
            style={{ transform: `scale(${ringScale})` }}
          />
        )}
        <button
          onClick={isRecording ? onStop : onStart}
          disabled={disabled}
          className={
            'relative flex h-24 w-24 items-center justify-center rounded-full text-3xl font-semibold transition active:scale-95 ' +
            (isRecording
              ? 'bg-rose-500 text-white shadow-lg shadow-rose-500/40'
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
          ? 'Tap to stop'
          : isThinking
          ? 'Coach is listening…'
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

function ActionsRow({
  onSkip,
  onRest,
  disableSkip,
}: {
  onSkip: () => void;
  onRest: () => void;
  disableSkip: boolean;
}) {
  return (
    <div className="mt-2 flex justify-center gap-3 text-sm">
      <button
        onClick={onSkip}
        disabled={disableSkip}
        className="rounded-full border border-zinc-800 bg-zinc-900/40 px-4 py-1.5 text-zinc-400 transition hover:bg-zinc-800/60 hover:text-zinc-200 disabled:opacity-40"
      >
        Skip drill
      </button>
      <button
        onClick={onRest}
        className="rounded-full border border-zinc-800 bg-zinc-900/40 px-4 py-1.5 text-zinc-400 transition hover:bg-zinc-800/60 hover:text-zinc-200"
      >
        End session
      </button>
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

function joinForSpeech(ack: string, feedback: string): string {
  const trim = (s: string) => s.replace(/[\s.!?,;:]+$/, '');
  const parts = [ack, feedback].map(trim).filter(Boolean);
  if (!parts.length) return '';
  return parts.join('. ') + '.';
}
