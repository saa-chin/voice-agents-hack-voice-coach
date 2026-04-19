import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { connect } from './ws';
import type { ServerMessage, ClientMessage } from './ws';

/**
 * A controllable WebSocket fake that lets tests step through CONNECTING →
 * OPEN → CLOSED states deterministically and inject incoming messages.
 */
class FakeWebSocket {
  static readyStates = { CONNECTING: 0, OPEN: 1, CLOSING: 2, CLOSED: 3 };
  readyState = FakeWebSocket.readyStates.CONNECTING;
  url: string;
  onmessage?: (ev: MessageEvent) => void;
  onclose?: (ev: CloseEvent) => void;
  sent: string[] = [];
  sendShouldThrow = false;
  closed = false;
  private listeners: Record<string, Array<(ev: any) => void>> = {};

  constructor(url: string) {
    this.url = url;
  }

  addEventListener(type: string, fn: (ev: any) => void, _opts?: any) {
    (this.listeners[type] ??= []).push(fn);
  }
  removeEventListener(type: string, fn: (ev: any) => void) {
    this.listeners[type] = (this.listeners[type] ?? []).filter((f) => f !== fn);
  }

  fire(type: string, ev: any) {
    for (const fn of this.listeners[type] ?? []) fn(ev);
  }

  triggerOpen() {
    this.readyState = FakeWebSocket.readyStates.OPEN;
    this.fire('open', new Event('open'));
  }
  triggerError() {
    this.fire('error', new Event('error'));
  }
  triggerMessage(payload: unknown) {
    const ev = { data: typeof payload === 'string' ? payload : JSON.stringify(payload) };
    this.fire('message', ev);
  }
  triggerClose(code = 1000, reason = '', wasClean = true) {
    this.readyState = FakeWebSocket.readyStates.CLOSED;
    this.fire('close', { code, reason, wasClean });
  }

  send(data: string) {
    if (this.sendShouldThrow) {
      throw new Error('send blew up');
    }
    this.sent.push(data);
  }
  close() {
    this.closed = true;
    this.readyState = FakeWebSocket.readyStates.CLOSED;
  }
}

declare global {
  // eslint-disable-next-line no-var
  var __lastWS: FakeWebSocket | undefined;
}

beforeEach(() => {
  // Constants are read off the global to match the real WebSocket shape.
  (globalThis as any).WebSocket = function (url: string) {
    const ws = new FakeWebSocket(url);
    globalThis.__lastWS = ws;
    return ws as unknown as WebSocket;
  } as unknown as typeof WebSocket;
  Object.assign((globalThis as any).WebSocket, FakeWebSocket.readyStates);
  vi.spyOn(console, 'log').mockImplementation(() => {});
  vi.spyOn(console, 'warn').mockImplementation(() => {});
  vi.spyOn(console, 'error').mockImplementation(() => {});
});

afterEach(() => {
  delete (globalThis as any).__lastWS;
  vi.restoreAllMocks();
});

function lastWS(): FakeWebSocket {
  const ws = globalThis.__lastWS;
  if (!ws) throw new Error('no WS created');
  return ws;
}

describe('connect()', () => {
  it('opens to the requested URL', () => {
    connect('ws://example/test', () => {});
    expect(lastWS().url).toBe('ws://example/test');
  });

  it('reports state as CONNECTING before open', () => {
    const c = connect('ws://example/test', () => {});
    expect(c.state()).toBe('CONNECTING');
  });

  it('reports state as OPEN after open event', async () => {
    const c = connect('ws://example/test', () => {});
    lastWS().triggerOpen();
    await c.ready;
    expect(c.state()).toBe('OPEN');
  });

  it('reports state as CLOSED after close', () => {
    const c = connect('ws://example/test', () => {});
    lastWS().triggerOpen();
    lastWS().triggerClose();
    expect(c.state()).toBe('CLOSED');
  });

  it('ready promise rejects on error before open', async () => {
    const c = connect('ws://example/test', () => {});
    lastWS().triggerError();
    await expect(c.ready).rejects.toBeDefined();
  });
});

describe('send buffering', () => {
  it('queues messages sent while CONNECTING and flushes on open', async () => {
    const c = connect('ws://example/test', () => {});
    const m1: ClientMessage = { type: 'start_session' };
    const m2: ClientMessage = { type: 'command', action: 'rest' };
    c.send(m1);
    c.send(m2);
    expect(lastWS().sent).toEqual([]); // nothing sent yet
    lastWS().triggerOpen();
    await c.ready;
    expect(lastWS().sent.map((s) => JSON.parse(s))).toEqual([m1, m2]);
  });

  it('sends immediately once OPEN', async () => {
    const c = connect('ws://example/test', () => {});
    lastWS().triggerOpen();
    await c.ready;
    c.send({ type: 'start_session' });
    expect(lastWS().sent).toEqual([JSON.stringify({ type: 'start_session' })]);
  });

  it('calls onSendError when send() throws', async () => {
    const errors: Array<[Error, ClientMessage]> = [];
    const c = connect('ws://example/test', () => {}, {
      onSendError: (err, msg) => errors.push([err, msg]),
    });
    lastWS().triggerOpen();
    await c.ready;
    lastWS().sendShouldThrow = true;
    c.send({ type: 'start_session' });
    expect(errors).toHaveLength(1);
    expect(errors[0][0].message).toBe('send blew up');
    expect(errors[0][1]).toEqual({ type: 'start_session' });
  });
});

describe('message dispatch', () => {
  it('parses JSON and forwards to handler', () => {
    const received: ServerMessage[] = [];
    connect('ws://example/test', (m) => received.push(m));
    lastWS().triggerOpen();
    lastWS().triggerMessage({ type: 'ready' });
    lastWS().triggerMessage({ type: 'thinking' });
    expect(received).toEqual([{ type: 'ready' }, { type: 'thinking' }]);
  });

  it('logs and skips malformed JSON without crashing', () => {
    const received: ServerMessage[] = [];
    connect('ws://example/test', (m) => received.push(m));
    lastWS().triggerOpen();
    lastWS().triggerMessage('this is not json {');
    expect(received).toEqual([]);
    expect(console.error).toHaveBeenCalled();
  });

  it('does not let handler exceptions take down the WS', () => {
    const handler = vi.fn(() => {
      throw new Error('handler bug');
    });
    connect('ws://example/test', handler);
    lastWS().triggerOpen();
    lastWS().triggerMessage({ type: 'ready' });
    expect(handler).toHaveBeenCalled();
    expect(console.error).toHaveBeenCalled();
  });
});

describe('close handler', () => {
  it('fires onCloseOrError with code/reason/clean', () => {
    const closes: any[] = [];
    connect('ws://example/test', () => {}, {
      onCloseOrError: (info) => closes.push(info),
    });
    lastWS().triggerOpen();
    lastWS().triggerClose(1006, 'server gone', false);
    expect(closes).toEqual([{ code: 1006, reason: 'server gone', clean: false }]);
  });
});

describe('close()', () => {
  it('marks the WS closed without throwing on stack swallowed', () => {
    const c = connect('ws://example/test', () => {});
    lastWS().triggerOpen();
    c.close();
    expect(lastWS().closed).toBe(true);
  });
});

describe('audio message size logging', () => {
  it('logs base64 length and estimated PCM bytes for audio messages', async () => {
    const c = connect('ws://example/test', () => {});
    lastWS().triggerOpen();
    await c.ready;
    c.send({ type: 'audio', pcm_b64: 'AAAA'.repeat(100), sample_rate: 16000 });
    // Audio sends now log with the type as a separate arg so the
    // intent_audio path can share the same compact size summary
    // without duplicating the format string.
    const audioLog = (console.log as any).mock.calls.find(
      (args: unknown[]) =>
        args[0] === '[coach ws] →' && args[1] === 'audio',
    );
    expect(audioLog).toBeDefined();
    const meta = audioLog?.[2];
    expect(meta).toMatchObject({
      sample_rate: 16000,
      base64_len: 400,
      est_pcm_bytes: 300,
    });
  });

  it('uses the same compact summary for intent_audio messages', async () => {
    const c = connect('ws://example/test', () => {});
    lastWS().triggerOpen();
    await c.ready;
    c.send({
      type: 'intent_audio',
      pcm_b64: 'AAAA'.repeat(50),
      sample_rate: 16000,
    });
    const log = (console.log as any).mock.calls.find(
      (args: unknown[]) =>
        args[0] === '[coach ws] →' && args[1] === 'intent_audio',
    );
    expect(log).toBeDefined();
    expect(log?.[2]).toMatchObject({
      sample_rate: 16000,
      base64_len: 200,
      est_pcm_bytes: 150,
    });
  });
});
