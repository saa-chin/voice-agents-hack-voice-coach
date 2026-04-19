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

export type ServerMessage =
  | { type: 'loading' }
  | { type: 'ready' }
  | DrillMsg
  | MetricsMsg
  | { type: 'thinking' }
  | CoachMsg
  | { type: 'advance' }
  | { type: 'retry' }
  | { type: 'rest' }
  | SummaryMsg
  | { type: 'error'; code: number; message: string };

export type ClientMessage =
  | { type: 'start_session' }
  | { type: 'audio'; pcm_b64: string; sample_rate: number }
  | { type: 'command'; action: 'skip' | 'rest' | 'repeat_prompt' };

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
  if (msg.type === 'audio') {
    const bytes = Math.round((msg.pcm_b64.length * 3) / 4);
    console.log('[coach ws] → audio', { sample_rate: msg.sample_rate, base64_len: msg.pcm_b64.length, est_pcm_bytes: bytes });
  } else {
    console.log('[coach ws] →', msg);
  }
}

function logRecv(msg: ServerMessage) {
  console.log('[coach ws] ←', msg);
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
