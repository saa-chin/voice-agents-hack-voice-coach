/**
 * Typed WebSocket wrapper for the /ws/coach endpoint.
 * Mirrors the wire format documented in backend/app/main.py.
 *
 * Behaviours that matter for a hackathon-grade UX:
 *  - Sends issued before the socket reaches OPEN are *queued* and flushed on
 *    open. Without this, a fast click during connect drops the message.
 *  - Every send/recv is logged to the browser console so a single Cmd+Opt+I
 *    is enough to debug "nothing happens" symptoms end-to-end.
 *  - Send failures (CLOSED socket, JSON encode error) are reported to the
 *    `onSendError` callback so the UI can surface them instead of silently
 *    losing the message.
 */

export interface DrillMsg {
  type: 'drill';
  stage: string;
  index: number;
  prompt: string;
  note: string;
  target_dbfs: number;
  total: number;
  position: number;
  // Richer context from the clinical program. Backend always sends these
  // (defaults to empty strings / 0 for legacy drills); frontend can rely on
  // them being present on every `drill` message.
  category_id?: string;
  category_name?: string;
  exercise_id?: string;
  exercise_name?: string;
  focus?: string;
  target_repetitions?: number;
  target_duration_sec?: number;
  /** Base64-encoded WAV of the drill prompt rendered server-side via
   * macOS `say`. When present, the client should play this instead of
   * speaking the prompt via browser speechSynthesis — keeps the entire
   * session on ONE consistent voice (otherwise drill prompts and coach
   * replies use different TTS engines and sound like two voices). */
  prompt_wav_b64?: string;
}

export interface MetricsMsg {
  type: 'metrics';
  dbfs: number | null;
  duration_s: number;
}

export interface CoachMsg {
  type: 'coach';
  heard: string;
  matched_prompt: boolean;
  ack: string;
  feedback: string;
  next_action: 'retry' | 'advance' | 'rest';
  metrics_observed: Record<string, boolean>;
  latency_s: number;
}

export interface IntentResultMsg {
  type: 'intent_result';
  /** Routed action — `none` means the router refused to act. */
  action: 'skip' | 'rest' | 'repeat_prompt' | 'none';
  /** 0..1 self-reported confidence. */
  confidence: number;
  /** Echo of the (lowercased, whitespace-collapsed) input. */
  utterance: string;
  /** Which classifier decided: `functiongemma` or `heuristic`. */
  source: 'functiongemma' | 'heuristic';
  /** Wall-clock classification time in ms. Useful as a "sub-100ms"
   * proof point when the router is the FunctionGemma model. */
  latency_ms: number;
  /** Whether FunctionGemma 270M was loaded at the time of the call.
   * False means the heuristic ran by necessity. */
  intent_model_loaded: boolean;
  /** What Whisper heard (audio path) or the typed text (typed path).
   * Surfaced so the demo chip can show the transcript next to the
   * routed action. */
  transcript: string;
  /** Whisper-tiny transcription latency in ms. null for the typed
   * path (no STT was performed). */
  transcribe_latency_ms: number | null;
  /** "whisper" when the server transcribed audio on-device, "client"
   * when the user typed (or when an older client transcribed locally
   * before sending text). */
  transcribe_source: 'whisper' | 'client';
}

export interface AudioReplyMsg {
  type: 'audio_reply';
  /** Base64-encoded WAV of the coach's spoken reply, rendered by the
   * server via macOS `say`. Played as-is — keeps every byte of TTS
   * audio on this machine. */
  wav_b64: string;
  /** Identifies the renderer (currently always "macos_say") so the
   * frontend can show provenance and degrade gracefully if a future
   * server uses a different on-device TTS. */
  source: string;
}

export interface SummaryMsg {
  type: 'session_done';
  summary: {
    advanced: number;
    total: number;
    retries: number;
    avg_dbfs: number | null;
    json_failures: number;
    rest_called: boolean;
    session_log: string;
  };
}

export interface ReadyMsg {
  type: 'ready';
  /** True when the backend can render the coach's reply to a WAV via
   * macOS `say` and ship it as `audio_reply`. The client uses this to
   * pick exactly one TTS source per session — server WAV when true,
   * browser speechSynthesis when false — so the two never race and
   * play simultaneously. */
  tts_available?: boolean;
  /** Snapshot of the optional models at handshake time; UI uses this
   * for status badges. */
  intent_loaded?: boolean;
  whisper_loaded?: boolean;
}

/** Granular progress beats inside the "thinking" phase. The backend
 * emits one frame per beat (in order) so the UI can render a live
 * stepper with per-step elapsed time instead of a static "Coach is
 * thinking…" placeholder for the full ~10s. The legacy single
 * `{ type: "thinking" }` frame (no step) remains valid: handlers
 * should treat it as the first beat. */
export type ThinkingStep =
  | 'analyzing_audio'
  | 'generating_response'
  | 'parsing_response'
  | 'synthesizing_voice';

export interface ThinkingMsg {
  type: 'thinking';
  /** Backend sends a step on every frame; older servers may omit it. */
  step?: ThinkingStep;
  /** Short, user-facing label for this step. Backend always includes
   * one; the frontend has a fallback table for older servers. */
  label?: string;
}

export type ServerMessage =
  | { type: 'loading' }
  | ReadyMsg
  | DrillMsg
  | MetricsMsg
  | ThinkingMsg
  | CoachMsg
  | AudioReplyMsg
  | { type: 'advance' }
  | { type: 'retry' }
  | { type: 'rest' }
  | IntentResultMsg
  | SummaryMsg
  | { type: 'error'; code: number; message: string };

export type ClientMessage =
  | { type: 'start_session' }
  | { type: 'audio'; pcm_b64: string; sample_rate: number }
  | { type: 'command'; action: 'skip' | 'rest' | 'repeat_prompt' }
  | { type: 'intent'; utterance: string }
  | { type: 'intent_audio'; pcm_b64: string; sample_rate: number };

export interface Connection {
  send: (msg: ClientMessage) => void;
  close: () => void;
  ready: Promise<void>;
  /** Current readyState as a friendly string. */
  state: () => 'CONNECTING' | 'OPEN' | 'CLOSING' | 'CLOSED';
}

export interface ConnectOptions {
  onCloseOrError?: (info: { code: number; reason: string; clean: boolean }) => void;
  onSendError?: (err: Error, msg: ClientMessage) => void;
}

const STATES = ['CONNECTING', 'OPEN', 'CLOSING', 'CLOSED'] as const;

function logSend(msg: ClientMessage) {
  // Audio messages are big — don't dump the base64 blob into console.
  if (msg.type === 'audio' || msg.type === 'intent_audio') {
    const bytes = Math.round((msg.pcm_b64.length * 3) / 4);
    console.log('[coach ws] →', msg.type, { sample_rate: msg.sample_rate, base64_len: msg.pcm_b64.length, est_pcm_bytes: bytes });
  } else {
    console.log('[coach ws] →', msg);
  }
}

function logRecv(msg: ServerMessage) {
  // Same treatment for inbound TTS audio — just summarise the size.
  if (msg.type === 'audio_reply') {
    const bytes = Math.round((msg.wav_b64.length * 3) / 4);
    console.log('[coach ws] ← audio_reply', { source: msg.source, base64_len: msg.wav_b64.length, est_wav_bytes: bytes });
  } else if (msg.type === 'drill' && msg.prompt_wav_b64) {
    // Drill payloads can carry a multi-KB prompt WAV — log a summary
    // instead of dumping the base64 blob into devtools.
    const { prompt_wav_b64, ...rest } = msg;
    const bytes = Math.round((prompt_wav_b64.length * 3) / 4);
    console.log('[coach ws] ←', { ...rest, prompt_wav: { base64_len: prompt_wav_b64.length, est_wav_bytes: bytes } });
  } else {
    console.log('[coach ws] ←', msg);
  }
}

export function connect(
  url: string,
  onMessage: (msg: ServerMessage) => void,
  opts: ConnectOptions = {},
): Connection {
  console.log('[coach ws] connecting to', url);
  const ws = new WebSocket(url);
  const queue: ClientMessage[] = [];
  let opened = false;

  const ready = new Promise<void>((resolve, reject) => {
    const onOpen = () => {
      opened = true;
      console.log('[coach ws] open');
      // Flush anything queued during CONNECTING.
      if (queue.length) {
        console.log('[coach ws] flushing', queue.length, 'queued message(s)');
        const drained = queue.splice(0, queue.length);
        for (const m of drained) doSend(m);
      }
      resolve();
    };
    const onError = (e: Event) => {
      console.error('[coach ws] error before open', e);
      reject(e);
    };
    ws.addEventListener('open', onOpen, { once: true });
    ws.addEventListener('error', onError, { once: true });
  });

  ws.addEventListener('message', (ev) => {
    let parsed: ServerMessage;
    try {
      parsed = JSON.parse(ev.data) as ServerMessage;
    } catch (e) {
      console.error('[coach ws] bad incoming JSON', ev.data, e);
      return;
    }
    logRecv(parsed);
    try {
      onMessage(parsed);
    } catch (e) {
      console.error('[coach ws] handler threw on', parsed, e);
    }
  });

  ws.addEventListener('close', (ev) => {
    console.log('[coach ws] close', { code: ev.code, reason: ev.reason, clean: ev.wasClean });
    opts.onCloseOrError?.({ code: ev.code, reason: ev.reason, clean: ev.wasClean });
  });

  function doSend(msg: ClientMessage): void {
    try {
      const payload = JSON.stringify(msg);
      ws.send(payload);
      logSend(msg);
    } catch (e) {
      const err = e instanceof Error ? e : new Error(String(e));
      console.error('[coach ws] send failed', { msg, err, state: STATES[ws.readyState] });
      opts.onSendError?.(err, msg);
    }
  }

  return {
    send(msg) {
      if (!opened || ws.readyState !== WebSocket.OPEN) {
        console.log('[coach ws] queueing (state=' + STATES[ws.readyState] + ')', msg);
        queue.push(msg);
        return;
      }
      doSend(msg);
    },
    close() {
      try {
        ws.close();
      } catch {
        /* ignore */
      }
    },
    ready,
    state: () => STATES[ws.readyState],
  };
}
