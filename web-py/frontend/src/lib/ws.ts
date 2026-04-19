/**
 * Typed WebSocket wrapper for the /ws/coach endpoint.
 * Mirrors the wire format documented in backend/app/main.py.
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
}

export function connect(
  url: string,
  onMessage: (msg: ServerMessage) => void,
  onCloseOrError?: (info: { code: number; reason: string }) => void,
): Connection {
  const ws = new WebSocket(url);

  const ready = new Promise<void>((resolve, reject) => {
    const cleanup = () => {
      ws.removeEventListener('open', onOpen);
      ws.removeEventListener('error', onError);
    };
    const onOpen = () => { cleanup(); resolve(); };
    const onError = (e: Event) => { cleanup(); reject(e); };
    ws.addEventListener('open', onOpen, { once: true });
    ws.addEventListener('error', onError, { once: true });
  });

  ws.addEventListener('message', (ev) => {
    try {
      onMessage(JSON.parse(ev.data) as ServerMessage);
    } catch (e) {
      console.error('coach: bad ws message', ev.data, e);
    }
  });

  ws.addEventListener('close', (ev) => {
    onCloseOrError?.({ code: ev.code, reason: ev.reason });
  });

  return {
    send(msg) {
      ws.send(JSON.stringify(msg));
    },
    close() {
      try { ws.close(); } catch { /* ignore */ }
    },
    ready,
  };
}
