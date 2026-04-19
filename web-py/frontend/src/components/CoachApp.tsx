import { useEffect, useRef, useState } from 'react';
import {
  cancelServerSpeech,
  cancelSpeech,
  createRecognizer,
  createRecorder,
  int16ToBase64,
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
  type ThinkingStep,
} from '../lib/ws';

const BACKEND_PORT =
  (import.meta as any).env?.PUBLIC_BACKEND_PORT ?? '8765';

function resolveWsUrl(): string {
  const explicit = (import.meta as any).env?.PUBLIC_BACKEND_WS_URL;
  if (explicit) return explicit;
  if (typeof window !== 'undefined' && window.location?.hostname) {
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

// Orb visual state, distinct from `Phase` so the mapping stays in
// one place and the canvas doesn't need to know about WS lifecycle.
type OrbMode = 'idle' | 'listening' | 'thinking' | 'speaking';

function phaseToOrb(phase: Phase): OrbMode {
  if (phase === 'recording') return 'listening';
  if (phase === 'thinking') return 'thinking';
  if (phase === 'drill' || phase === 'feedback') return 'speaking';
  return 'idle';
}

export default function CoachApp() {
  const [phase, setPhase] = useState<Phase>('connecting');
  const [drill, setDrill] = useState<DrillMsg | null>(null);
  const [_metrics, setMetrics] = useState<MetricsMsg | null>(null);
  const [coach, setCoach] = useState<CoachMsg | null>(null);
  const [summary, setSummary] = useState<SummaryMsg['summary'] | null>(null);
  const [errMsg, setErrMsg] = useState<string | null>(null);
  const [transientError, setTransientError] = useState<string | null>(null);
  const [liveTranscript, setLiveTranscript] = useState<{
    final: string;
    interim: string;
  }>({ final: '', interim: '' });
  // Intent router state is still tracked so voice commands keep working,
  // but not displayed prominently — a tiny hint line surfaces the
  // verdict so the feature remains discoverable.
  const [_intentResult, setIntentResult] = useState<IntentResultMsg | null>(
    null,
  );
  const [_thinkingLabel, setThinkingLabel] = useState<string | null>(null);
  const [, forceTick] = useState(0);

  const wsRef = useRef<Connection | null>(null);
  const recRef = useRef<RecorderHandle | null>(null);
  const recognizerRef = useRef<RecognizerHandle | null>(null);
  const serverTtsPlayingRef = useRef(false);
  const serverTtsAvailableRef = useRef(false);
  const levelTimer = useRef<number | null>(null);
  const phaseRef = useRef<Phase>('connecting');
  const endingRef = useRef(false);
  const startRecordingRef = useRef<() => Promise<void>>();
  const stopRecordingRef = useRef<() => Promise<void>>();
  const wsInitialized = useRef(false);

  useEffect(() => {
    phaseRef.current = phase;
  }, [phase]);

  // Force dark background so the orb reads like Siri on any system.
  useEffect(() => {
    if (typeof document !== 'undefined') {
      document.documentElement.classList.add('dark');
    }
  }, []);

  // ---- WS lifecycle ------------------------------------------------------

  useEffect(() => {
    if (wsInitialized.current) return;
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
    ws.ready.catch((e) => {
      console.error('[coach] ws.ready rejected', e);
      setErrMsg(
        `Could not reach coach server at ${BACKEND_WS_URL}. Is the backend running?`,
      );
      setPhase('error');
    });

    return () => {
      ws.close();
      cancelSpeech();
      cancelServerSpeech();
      recRef.current?.cancel();
      recognizerRef.current?.cancel();
      recognizerRef.current = null;
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
        setThinkingLabel(null);
        setPhase('drill');
        speakDrillPrompt(msg, () => {
          // Auto-arm the mic the moment the prompt finishes reading.
          if (phaseRef.current !== 'drill') return;
          startRecordingRef.current?.();
        });
        return;
      case 'metrics':
        if (endingRef.current) return;
        setMetrics(msg);
        return;
      case 'thinking': {
        if (endingRef.current) return;
        setPhase('thinking');
        const step: ThinkingStep = msg.step ?? 'analyzing_audio';
        const label = msg.label ?? defaultThinkingLabel(step);
        setThinkingLabel(label);
        return;
      }
      case 'coach': {
        if (endingRef.current) return;
        setCoach(msg);
        setPhase('feedback');
        if (!serverTtsAvailableRef.current) {
          const spoken = joinForSpeech(msg.ack, msg.feedback);
          if (spoken) speak(spoken);
        }
        return;
      }
      case 'audio_reply': {
        if (endingRef.current) return;
        serverTtsPlayingRef.current = true;
        playWavBase64(msg.wav_b64)
          .catch((e) => {
            console.warn('[coach] server TTS playback failed', e);
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
        return;
      case 'intent_result':
        setIntentResult(msg);
        return;
      case 'session_done':
        setSummary(msg.summary);
        setPhase('done');
        cancelSpeech();
        cancelServerSpeech();
        endingRef.current = false;
        return;
      case 'error':
        setTransientError(msg.message);
        return;
    }
  }

  // ---- User actions ------------------------------------------------------

  const startSession = () => {
    setSummary(null);
    setErrMsg(null);
    setTransientError(null);
    endingRef.current = false;
    primeTTS();
    if (!wsRef.current) {
      setTransientError('No WebSocket connection. Reload the page.');
      return;
    }
    wsRef.current.send({ type: 'start_session' });
  };

  const startRecording = async () => {
    cancelSpeech();
    cancelServerSpeech();
    setTransientError(null);
    setLiveTranscript({ final: '', interim: '' });
    setThinkingLabel(null);
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
        vad: {
          enabled: true,
          onSpeechEnd: () => {
            stopRecordingRef.current?.();
          },
        },
      });
      await recRef.current.start();
      setPhase('recording');
      levelTimer.current = window.setInterval(
        () => forceTick((t) => t + 1),
        50,
      ) as unknown as number;

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
    if (recognizerRef.current) {
      try { recognizerRef.current.stop(); } catch { /* ignore */ }
      recognizerRef.current = null;
    }
    const rec = recRef.current;
    recRef.current = null;
    try {
      const { pcm, durationMs } = await rec.stop();
      if (pcm.length === 0) {
        setTransientError(
          `No audio captured (${durationMs.toFixed(0)}ms). Try again.`,
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

  const speakDrillPrompt = (msg: DrillMsg, onDone?: () => void) => {
    const fire = () => {
      try { onDone?.(); } catch (e) { console.warn('[coach] onDone threw', e); }
    };
    if (msg.prompt_wav_b64) {
      cancelSpeech();
      playWavBase64(msg.prompt_wav_b64)
        .catch((e) => {
          console.warn('[coach] drill prompt wav failed, using browser TTS', e);
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
        if (phaseRef.current === 'drill') {
          startRecordingRef.current?.();
        }
      });
    }
  };
  const skipDrill = () => wsRef.current?.send({ type: 'command', action: 'skip' });

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
    if (levelTimer.current !== null) {
      window.clearInterval(levelTimer.current);
      levelTimer.current = null;
    }
    setTransientError(null);
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

  startRecordingRef.current = startRecording;
  stopRecordingRef.current = stopRecording;

  // ---- Render ------------------------------------------------------------

  const analysis: AudioAnalysis | null =
    phase === 'recording' ? recRef.current?.analysis() ?? null : null;
  const liveLevel = analysis?.rms != null ? Math.min(1, analysis.rms * 4) : 0;

  // Headline = what's above the orb (small, slightly dim). For drills we
  // show the expected utterance (in quotes) or the instruction. For the
  // coach's reply we show what it said so the user can read + hear.
  const headline = (() => {
    if (phase === 'feedback' && coach) {
      const parts = [coach.ack, coach.feedback].filter(Boolean);
      return parts.length ? parts.join(' ') : null;
    }
    if ((phase === 'drill' || phase === 'recording' || phase === 'thinking') && drill) {
      if (drill.note?.trim()) return `“${drill.prompt}”`;
      return drill.prompt;
    }
    return null;
  })();

  // Caption = what's under the orb (Siri-style "Listening…", "Go ahead", etc).
  const caption = (() => {
    switch (phase) {
      case 'connecting': return 'Connecting…';
      case 'loading': return 'Waking the coach on your device…';
      case 'ready': return 'Tap to start';
      case 'drill': return 'Listen, then speak';
      case 'recording': return analysis?.speaking ? 'Listening…' : 'Go ahead, I\'m listening';
      case 'thinking': return 'Thinking…';
      case 'feedback': return 'Coach';
      case 'done': return 'Session complete';
      case 'error': return 'Can\'t reach the coach';
    }
  })();

  // Click semantics on the orb change with phase. One target, one
  // affordance — no separate mic / start / stop buttons.
  const orbAction = (() => {
    if (phase === 'ready' || phase === 'done') return startSession;
    if (phase === 'drill') return () => startRecordingRef.current?.();
    if (phase === 'recording') return () => stopRecordingRef.current?.();
    return undefined;
  })();

  const orbMode = phaseToOrb(phase);

  return (
    <main className="relative z-10 mx-auto flex min-h-screen w-full max-w-2xl flex-col items-center justify-between px-6 py-8">
      {!isTTSAvailable() && (
        <div className="rounded-xl border border-[var(--warning-soft)] bg-[var(--warning-soft)] px-4 py-2 text-xs text-[var(--warning)]">
          This browser has no <code>speechSynthesis</code>. The coach will be
          silent — try Chrome, Edge, or Safari.
        </div>
      )}

      {/* Top area: single line of context (prompt or coach reply). */}
      <div className="flex min-h-[72px] w-full max-w-xl items-center justify-center pt-6 text-center">
        {headline && phase !== 'done' && phase !== 'error' && (
          <p className="text-lg font-medium leading-snug text-[var(--text)]/90 transition-opacity duration-300">
            {headline}
          </p>
        )}
        {phase === 'error' && (
          <p className="text-sm text-[var(--danger)]">{errMsg}</p>
        )}
      </div>

      {/* Center: the orb. */}
      <div className="flex flex-1 flex-col items-center justify-center gap-6">
        <SiriOrb
          level={liveLevel}
          mode={orbMode}
          onClick={orbAction}
          clickable={orbAction != null}
        />

        <div className="flex min-h-[56px] flex-col items-center gap-1 text-center">
          <div className="text-sm font-medium tracking-wide text-[var(--text-muted)]">
            {caption}
          </div>
          <TranscriptLine
            transcript={liveTranscript}
            visible={phase === 'recording' || phase === 'thinking'}
          />
        </div>
      </div>

      {/* Bottom: minimal controls. Text links rather than chunky buttons. */}
      <div className="flex min-h-[56px] w-full items-center justify-center pb-2">
        {phase === 'done' && summary ? (
          <DoneLine summary={summary} onRestart={startSession} />
        ) : phase === 'drill' || phase === 'recording' || phase === 'thinking' || phase === 'feedback' ? (
          <SessionControls
            onRepeat={repeatPrompt}
            onSkip={skipDrill}
            onEnd={restSession}
            disabled={phase === 'thinking'}
          />
        ) : null}
      </div>

      {transientError && (
        <div className="fixed inset-x-0 bottom-4 mx-auto w-fit max-w-sm rounded-full bg-[var(--danger-soft)] px-4 py-1.5 text-xs text-[var(--danger)] backdrop-blur">
          {transientError}
        </div>
      )}
    </main>
  );
}

// ---- Siri orb -------------------------------------------------------------
//
// Canvas-drawn glowing blob. Four radial gradients orbit inside a circular
// clip; their size breathes with the latest mic RMS and their speed +
// palette shift based on `mode`. A subtle specular highlight and an outer
// glow halo make the whole thing feel materially 3D without shipping a
// WebGL shader. Intentionally not audio-FFT-driven: we already compute a
// smoothed RMS upstream and the orb should remain legible at low signal.

type OrbColor = { r: number; g: number; b: number; a: number };

const ORB_PALETTE: Record<OrbMode, OrbColor[]> = {
  idle: [
    { r: 99, g: 102, b: 241, a: 0.55 },
    { r: 56, g: 189, b: 248, a: 0.40 },
    { r: 167, g: 139, b: 250, a: 0.45 },
  ],
  listening: [
    { r: 16, g: 185, b: 129, a: 0.70 },
    { r: 52, g: 211, b: 153, a: 0.65 },
    { r: 59, g: 130, b: 246, a: 0.55 },
    { r: 168, g: 85, b: 247, a: 0.50 },
  ],
  thinking: [
    { r: 245, g: 158, b: 11, a: 0.65 },
    { r: 168, g: 85, b: 247, a: 0.60 },
    { r: 59, g: 130, b: 246, a: 0.55 },
    { r: 244, g: 63, b: 94, a: 0.45 },
  ],
  speaking: [
    { r: 52, g: 211, b: 153, a: 0.75 },
    { r: 56, g: 189, b: 248, a: 0.65 },
    { r: 168, g: 85, b: 247, a: 0.55 },
  ],
};

function rgba(c: OrbColor, aMul = 1) {
  return `rgba(${c.r}, ${c.g}, ${c.b}, ${(c.a * aMul).toFixed(3)})`;
}

function SiriOrb({
  level,
  mode,
  onClick,
  clickable,
}: {
  level: number;
  mode: OrbMode;
  onClick?: () => void;
  clickable: boolean;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const rafRef = useRef<number | null>(null);
  const startRef = useRef(performance.now());
  const levelRef = useRef(level);
  const smoothLevelRef = useRef(0);
  const modeRef = useRef<OrbMode>(mode);

  useEffect(() => {
    levelRef.current = level;
  }, [level]);
  useEffect(() => {
    modeRef.current = mode;
  }, [mode]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const size = 320;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    canvas.width = size * dpr;
    canvas.height = size * dpr;
    canvas.style.width = `${size}px`;
    canvas.style.height = `${size}px`;
    ctx.scale(dpr, dpr);

    const draw = () => {
      const t = (performance.now() - startRef.current) / 1000;
      const m = modeRef.current;

      // Low-pass the audio level so the orb glides instead of jittering
      // at 20 Hz frames; speaking floor keeps the orb alive while the
      // coach talks (we don't have its output amplitude, so we fake a
      // gentle sinusoid).
      let target = levelRef.current;
      if (m === 'speaking') target = Math.max(target, 0.18 + Math.sin(t * 3.2) * 0.06);
      if (m === 'thinking') target = Math.max(target, 0.12);
      if (m === 'idle') target = 0.02;
      smoothLevelRef.current += (target - smoothLevelRef.current) * 0.14;
      const lv = smoothLevelRef.current;

      const speedMul =
        m === 'thinking' ? 2.4 : m === 'listening' ? 1.6 : m === 'speaking' ? 1.3 : 0.6;
      const colors = ORB_PALETTE[m];

      ctx.clearRect(0, 0, size, size);

      const cx = size / 2;
      const cy = size / 2;
      const baseR = size / 2;

      ctx.save();
      ctx.beginPath();
      ctx.arc(cx, cy, baseR, 0, Math.PI * 2);
      ctx.clip();

      // Deep backdrop so the colored lobes read as luminous, not pastel.
      const bg = ctx.createRadialGradient(cx, cy, 0, cx, cy, baseR);
      bg.addColorStop(0, '#0a0d14');
      bg.addColorStop(1, '#04060a');
      ctx.fillStyle = bg;
      ctx.fillRect(0, 0, size, size);

      // Colored lobes on an additive blend.
      ctx.globalCompositeOperation = 'lighter';
      const breathe = 1 + lv * 0.12 + Math.sin(t * 1.8) * 0.015;
      for (let i = 0; i < colors.length; i++) {
        const phase =
          t * (0.28 + i * 0.11) * speedMul + (i * Math.PI * 2) / colors.length;
        const orbitR = baseR * (0.22 + Math.sin(t * 0.55 + i * 1.3) * 0.07);
        const x = cx + Math.cos(phase) * orbitR;
        const y = cy + Math.sin(phase * 1.08 + i) * orbitR;
        const r =
          baseR * (0.55 + lv * 0.35) *
          (0.9 + Math.sin(t * 0.9 + i * 2) * 0.06) *
          breathe;
        const g = ctx.createRadialGradient(x, y, 0, x, y, r);
        g.addColorStop(0, rgba(colors[i], 1));
        g.addColorStop(0.55, rgba(colors[i], 0.35));
        g.addColorStop(1, 'rgba(0,0,0,0)');
        ctx.fillStyle = g;
        ctx.fillRect(0, 0, size, size);
      }

      // Specular highlight — upper-left bias like a pearl.
      const hl = ctx.createRadialGradient(
        cx - baseR * 0.32,
        cy - baseR * 0.38,
        0,
        cx - baseR * 0.32,
        cy - baseR * 0.38,
        baseR * 0.55,
      );
      hl.addColorStop(0, 'rgba(255, 255, 255, 0.22)');
      hl.addColorStop(1, 'rgba(255, 255, 255, 0)');
      ctx.fillStyle = hl;
      ctx.fillRect(0, 0, size, size);

      ctx.restore();

      // Outer bloom so the orb feels like it's emitting, not painted on.
      ctx.globalCompositeOperation = 'source-over';
      const ring = ctx.createRadialGradient(
        cx,
        cy,
        baseR * 0.94,
        cx,
        cy,
        baseR * 1.18,
      );
      const bloomColor =
        m === 'thinking'
          ? 'rgba(168, 85, 247, 0.30)'
          : m === 'listening'
          ? 'rgba(52, 211, 153, 0.30)'
          : m === 'speaking'
          ? 'rgba(56, 189, 248, 0.28)'
          : 'rgba(99, 102, 241, 0.22)';
      ring.addColorStop(0, bloomColor);
      ring.addColorStop(1, 'rgba(0,0,0,0)');
      ctx.fillStyle = ring;
      ctx.fillRect(0, 0, size, size);

      rafRef.current = requestAnimationFrame(draw);
    };
    rafRef.current = requestAnimationFrame(draw);
    return () => {
      if (rafRef.current != null) cancelAnimationFrame(rafRef.current);
    };
  }, []);

  return (
    <button
      type="button"
      onClick={onClick}
      disabled={!clickable}
      aria-label="Voice coach"
      className={
        'siri-orb relative block rounded-full outline-none transition-transform duration-200 ' +
        (clickable
          ? 'cursor-pointer hover:scale-[1.02] active:scale-[0.98] focus-visible:ring-2 focus-visible:ring-[var(--accent-ring)]'
          : 'cursor-default')
      }
    >
      <canvas ref={canvasRef} className="block rounded-full" />
    </button>
  );
}

// ---- Small supporting components -----------------------------------------

function TranscriptLine({
  transcript,
  visible,
}: {
  transcript: { final: string; interim: string };
  visible: boolean;
}) {
  const text = [transcript.final, transcript.interim].filter(Boolean).join(' ');
  if (!visible || !text) return <div className="h-4" aria-hidden="true" />;
  return (
    <div className="max-w-md truncate text-xs italic text-[var(--text-faint)]">
      “{text}”
    </div>
  );
}

function SessionControls({
  onRepeat,
  onSkip,
  onEnd,
  disabled,
}: {
  onRepeat: () => void;
  onSkip: () => void;
  onEnd: () => void;
  disabled: boolean;
}) {
  const base =
    'rounded-full px-3 py-1 text-xs uppercase tracking-[0.18em] text-[var(--text-muted)] transition hover:text-[var(--text)] disabled:opacity-40 disabled:pointer-events-none';
  return (
    <div className="flex items-center gap-1">
      <button type="button" onClick={onRepeat} disabled={disabled} className={base}>
        Repeat
      </button>
      <span className="text-[var(--text-faint)]">·</span>
      <button type="button" onClick={onSkip} disabled={disabled} className={base}>
        Skip
      </button>
      <span className="text-[var(--text-faint)]">·</span>
      <button
        type="button"
        onClick={onEnd}
        className={base + ' hover:text-[var(--danger)]'}
      >
        End
      </button>
    </div>
  );
}

function DoneLine({
  summary,
  onRestart,
}: {
  summary: SummaryMsg['summary'];
  onRestart: () => void;
}) {
  return (
    <div className="flex flex-col items-center gap-3 text-center">
      <div className="text-xs uppercase tracking-[0.18em] text-[var(--text-faint)]">
        {summary.advanced} / {summary.total} drills
        {summary.avg_dbfs != null && (
          <span> · avg {summary.avg_dbfs} dBFS</span>
        )}
      </div>
      <button
        type="button"
        onClick={onRestart}
        className="rounded-full border border-[var(--border-strong)] bg-[var(--surface-2)] px-5 py-1.5 text-sm text-[var(--text)] transition hover:bg-[var(--surface)]"
      >
        Start again
      </button>
    </div>
  );
}

// ---- helpers --------------------------------------------------------------

function buildDrillTTS(drill: DrillMsg): string {
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

function defaultThinkingLabel(step: ThinkingStep): string {
  switch (step) {
    case 'analyzing_audio': return 'Analyzing your audio…';
    case 'generating_response': return 'Generating coaching response…';
    case 'parsing_response': return 'Parsing model output…';
    case 'synthesizing_voice': return 'Synthesizing voice…';
  }
}
