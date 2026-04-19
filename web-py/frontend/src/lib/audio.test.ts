/**
 * Unit tests for audio.ts.
 *
 * `createRecorder()` is excluded — it requires `MediaRecorder` and an actual
 * audio stream from `getUserMedia`, which happy-dom doesn't implement. The
 * pure helpers (int16ToBase64, the internal resampler exposed via TTS,
 * primeTTS, speak, testSpeech, voice selection, status reporting) are
 * fully covered.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

// Voice list returned by the synthesis stub.
let mockVoices: SpeechSynthesisVoice[] = [];

// Spies stay stable across tests; we reset call history per-test.
const speakSpy = vi.fn();
const cancelSpy = vi.fn();
const addEventSpy = vi.fn();

class FakeUtterance {
  text: string;
  voice: SpeechSynthesisVoice | null = null;
  lang = '';
  rate = 1;
  pitch = 1;
  volume = 1;
  onstart: ((ev: any) => void) | null = null;
  onend: ((ev: any) => void) | null = null;
  onerror: ((ev: any) => void) | null = null;
  constructor(text: string) {
    this.text = text;
  }
}

beforeEach(async () => {
  mockVoices = [
    { name: 'Junk', lang: 'fr-FR' } as SpeechSynthesisVoice,
    { name: 'Samantha', lang: 'en-US' } as SpeechSynthesisVoice,
    { name: 'Daniel', lang: 'en-GB' } as SpeechSynthesisVoice,
  ];
  speakSpy.mockReset();
  cancelSpy.mockReset();
  addEventSpy.mockReset();
  vi.spyOn(console, 'log').mockImplementation(() => {});
  vi.spyOn(console, 'warn').mockImplementation(() => {});
  vi.spyOn(console, 'error').mockImplementation(() => {});

  (globalThis as any).SpeechSynthesisUtterance = FakeUtterance;
  (globalThis as any).window = globalThis as any;
  (globalThis as any).speechSynthesis = {
    getVoices: () => mockVoices,
    speak: (u: FakeUtterance) => speakSpy(u),
    cancel: () => cancelSpy(),
    addEventListener: (...args: any[]) => addEventSpy(...args),
  };
  // Reset module so the singleton voice cache + primed flag is fresh per test.
  vi.resetModules();
});

afterEach(() => {
  vi.restoreAllMocks();
  delete (globalThis as any).speechSynthesis;
  delete (globalThis as any).SpeechSynthesisUtterance;
});

async function loadAudio() {
  return await import('./audio');
}

// ---- int16ToBase64 -----------------------------------------------------

describe('int16ToBase64', () => {
  it('round-trips a small Int16Array', async () => {
    const { int16ToBase64 } = await loadAudio();
    const arr = new Int16Array([0, 1, -1, 32767, -32768]);
    const b64 = int16ToBase64(arr);
    // Decode back via atob and compare bytes.
    const bin = atob(b64);
    const bytes = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
    const view = new Int16Array(bytes.buffer);
    expect(Array.from(view)).toEqual(Array.from(arr));
  });

  it('handles an empty array', async () => {
    const { int16ToBase64 } = await loadAudio();
    expect(int16ToBase64(new Int16Array(0))).toBe('');
  });

  it('handles arrays larger than the chunk size', async () => {
    const { int16ToBase64 } = await loadAudio();
    // Larger than the 0x8000 chunk break in the implementation.
    const big = new Int16Array(50_000);
    for (let i = 0; i < big.length; i++) big[i] = (i * 13) & 0x7fff;
    const b64 = big.length > 0 ? loadAudio() && int16ToBase64(big) : '';
    expect(typeof b64).toBe('string');
    expect((b64 as string).length).toBeGreaterThan(0);
    // Round-trip a few sample positions.
    const bin = atob(b64 as string);
    const bytes = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
    const view = new Int16Array(bytes.buffer);
    expect(view[0]).toBe(big[0]);
    expect(view[12345]).toBe(big[12345]);
    expect(view[big.length - 1]).toBe(big[big.length - 1]);
  });
});

// ---- floatToInt16 ------------------------------------------------------

describe('floatToInt16', () => {
  it('clips out-of-range values', async () => {
    const { floatToInt16 } = await loadAudio();
    const out = floatToInt16(new Float32Array([1.5, -1.5, 2.0, -2.0]));
    expect(out[0]).toBe(32767);
    expect(out[1]).toBe(-32768);
    expect(out[2]).toBe(32767);
    expect(out[3]).toBe(-32768);
  });

  it('scales mid-range correctly', async () => {
    const { floatToInt16 } = await loadAudio();
    const out = floatToInt16(new Float32Array([0, 0.5, -0.5, 1.0, -1.0]));
    expect(out[0]).toBe(0);
    expect(out[1]).toBe(Math.round(0.5 * 0x7fff));
    expect(out[2]).toBe(Math.round(-0.5 * 0x8000));
    expect(out[3]).toBe(32767);
    expect(out[4]).toBe(-32768);
  });

  it('preserves length', async () => {
    const { floatToInt16 } = await loadAudio();
    const out = floatToInt16(new Float32Array(1024));
    expect(out.length).toBe(1024);
  });
});

// ---- audioBufferToInt16 -----------------------------------------------

class FakeAudioBuffer implements AudioBuffer {
  numberOfChannels: number;
  length: number;
  sampleRate: number;
  duration = 0;
  private data: Float32Array[];

  constructor(channels: Float32Array[], sampleRate: number) {
    this.data = channels;
    this.numberOfChannels = channels.length;
    this.length = channels[0]?.length ?? 0;
    this.sampleRate = sampleRate;
  }

  getChannelData(ch: number): Float32Array {
    return this.data[ch];
  }
  copyFromChannel() {}
  copyToChannel() {}
}

describe('audioBufferToInt16', () => {
  it('passes through when sample rate already matches target', async () => {
    const { audioBufferToInt16 } = await loadAudio();
    const ch = new Float32Array([0, 0.5, -0.5, 1.0]);
    const buf = new FakeAudioBuffer([ch], 16000);
    const out = audioBufferToInt16(buf as any, 16000);
    expect(out.length).toBe(4);
    expect(out[0]).toBe(0);
    expect(out[3]).toBe(32767);
  });

  it('downsamples 48 kHz → 16 kHz to ~1/3 the length', async () => {
    const { audioBufferToInt16 } = await loadAudio();
    const ch = new Float32Array(48000); // 1 second
    const buf = new FakeAudioBuffer([ch], 48000);
    const out = audioBufferToInt16(buf as any, 16000);
    // 48000 * (16000/48000) = 16000 samples
    expect(out.length).toBe(16000);
  });

  it('upsamples 8 kHz → 16 kHz to ~2x the length', async () => {
    const { audioBufferToInt16 } = await loadAudio();
    const ch = new Float32Array(8000);
    const buf = new FakeAudioBuffer([ch], 8000);
    const out = audioBufferToInt16(buf as any, 16000);
    expect(out.length).toBe(16000);
  });

  it('mixes multi-channel down to mono', async () => {
    const { audioBufferToInt16 } = await loadAudio();
    // L = 1.0 throughout, R = -1.0 throughout → mono should be 0.0
    const left = new Float32Array(1000).fill(1.0);
    const right = new Float32Array(1000).fill(-1.0);
    const buf = new FakeAudioBuffer([left, right], 16000);
    const out = audioBufferToInt16(buf as any, 16000);
    for (const s of out) expect(s).toBe(0);
  });

  it('preserves a constant signal through the resampler', async () => {
    const { audioBufferToInt16 } = await loadAudio();
    const ch = new Float32Array(48000).fill(0.5);
    const buf = new FakeAudioBuffer([ch], 48000);
    const out = audioBufferToInt16(buf as any, 16000);
    // Constant signal should remain constant after linear interpolation,
    // modulo 1 lsb of rounding.
    const expected = Math.round(0.5 * 0x7fff);
    for (const s of out) {
      expect(Math.abs(s - expected)).toBeLessThan(2);
    }
  });
});

// ---- TTS: voice selection ---------------------------------------------

describe('voice selection (via speak)', () => {
  it('picks Samantha by name when available', async () => {
    const a = await loadAudio();
    a.primeTTS();
    expect(speakSpy).toHaveBeenCalled();
    const u = speakSpy.mock.calls[0][0] as FakeUtterance;
    expect(u.voice?.name).toBe('Samantha');
    expect(u.lang).toBe('en-US');
  });

  it('falls back to first English voice if no preferred name matches', async () => {
    mockVoices = [
      { name: 'Random', lang: 'en-NZ' } as SpeechSynthesisVoice,
      { name: 'Other', lang: 'fr-FR' } as SpeechSynthesisVoice,
    ];
    const a = await loadAudio();
    a.primeTTS();
    const u = speakSpy.mock.calls[0][0] as FakeUtterance;
    expect(u.voice?.name).toBe('Random');
  });

  it('falls back to first voice of any locale when no English present', async () => {
    mockVoices = [{ name: 'Frenchy', lang: 'fr-FR' } as SpeechSynthesisVoice];
    const a = await loadAudio();
    a.primeTTS();
    const u = speakSpy.mock.calls[0][0] as FakeUtterance;
    expect(u.voice?.name).toBe('Frenchy');
  });

  it('handles empty voice list (engine still gets called with default)', async () => {
    mockVoices = [];
    const a = await loadAudio();
    a.primeTTS();
    expect(speakSpy).toHaveBeenCalled();
    const u = speakSpy.mock.calls[0][0] as FakeUtterance;
    expect(u.voice).toBeNull();
    // lang fallback should still be set to en-US.
    expect(u.lang).toBe('en-US');
  });
});

// ---- isTTSAvailable ---------------------------------------------------

describe('isTTSAvailable', () => {
  it('returns true when speechSynthesis is exposed', async () => {
    const a = await loadAudio();
    expect(a.isTTSAvailable()).toBe(true);
  });

  it('returns false when speechSynthesis is missing', async () => {
    delete (globalThis as any).speechSynthesis;
    const a = await loadAudio();
    expect(a.isTTSAvailable()).toBe(false);
  });
});

// ---- primeTTS ---------------------------------------------------------

describe('primeTTS', () => {
  it('queues exactly one utterance on first call and is idempotent', async () => {
    const a = await loadAudio();
    a.primeTTS();
    a.primeTTS();
    a.primeTTS();
    expect(speakSpy).toHaveBeenCalledTimes(1);
  });

  it('marks status as primed', async () => {
    const a = await loadAudio();
    expect(a.getTTSStatus().primed).toBe(false);
    a.primeTTS();
    expect(a.getTTSStatus().primed).toBe(true);
  });

  it('no-ops when speechSynthesis is unavailable', async () => {
    delete (globalThis as any).speechSynthesis;
    const a = await loadAudio();
    a.primeTTS();
    expect(speakSpy).not.toHaveBeenCalled();
    expect(a.getTTSStatus().primed).toBe(false);
  });

  it('catches engine speak() errors and records the failure', async () => {
    speakSpy.mockImplementationOnce(() => {
      throw new Error('synth dead');
    });
    const a = await loadAudio();
    a.primeTTS();
    // Prime should not throw, and primed stays false on failure.
    expect(a.getTTSStatus().primed).toBe(false);
  });
});

// ---- speak -----------------------------------------------------------

describe('speak', () => {
  it('queues the requested text after priming', async () => {
    const a = await loadAudio();
    a.primeTTS();
    speakSpy.mockReset();
    a.speak('hello world');
    expect(speakSpy).toHaveBeenCalledTimes(1);
    expect((speakSpy.mock.calls[0][0] as FakeUtterance).text).toBe('hello world');
  });

  it('does not queue when text is empty/whitespace', async () => {
    const a = await loadAudio();
    a.primeTTS();
    speakSpy.mockReset();
    a.speak('');
    a.speak('   \t\n');
    expect(speakSpy).not.toHaveBeenCalled();
  });

  it('warns but still attempts when called before primeTTS', async () => {
    const a = await loadAudio();
    a.speak('unprimed call');
    expect(console.warn).toHaveBeenCalled();
    expect(speakSpy).toHaveBeenCalled();
  });

  it('records error status when speechSynthesis is missing', async () => {
    delete (globalThis as any).speechSynthesis;
    const a = await loadAudio();
    a.speak('hi');
    const s = a.getTTSStatus();
    expect(s.ok).toBe(false);
    expect(s.lastError).toContain('speechSynthesis');
  });

  it('captures synchronous throw from engine.speak()', async () => {
    const a = await loadAudio();
    a.primeTTS();
    speakSpy.mockReset();
    speakSpy.mockImplementationOnce(() => {
      throw new Error('boom');
    });
    a.speak('will fail');
    const s = a.getTTSStatus();
    expect(s.ok).toBe(false);
    expect(s.lastError).toBe('boom');
  });

  it('utterance fires onstart/onend/onerror callbacks for status tracking', async () => {
    const a = await loadAudio();
    a.primeTTS();
    speakSpy.mockReset();
    a.speak('test status');
    const u = speakSpy.mock.calls[0][0] as FakeUtterance;
    // Simulate the engine starting → ending.
    u.onstart?.({});
    expect(a.getTTSStatus().ok).toBe(true);
    u.onend?.({});
    // Simulate an error event.
    u.onerror?.({ error: 'not-allowed' });
    const s = a.getTTSStatus();
    expect(s.ok).toBe(false);
    expect(s.lastError).toBe('not-allowed');
  });

  it('onerror with no .error field reports "unknown"', async () => {
    const a = await loadAudio();
    a.primeTTS();
    speakSpy.mockReset();
    a.speak('x');
    const u = speakSpy.mock.calls[0][0] as FakeUtterance;
    u.onerror?.({});
    expect(a.getTTSStatus().lastError).toBe('unknown');
  });
});

// ---- testSpeech -------------------------------------------------------

describe('testSpeech', () => {
  it('returns true and queues an utterance', async () => {
    const a = await loadAudio();
    expect(a.testSpeech()).toBe(true);
    expect(speakSpy).toHaveBeenCalled();
  });

  it('marks the engine primed (counts as the prime gesture)', async () => {
    const a = await loadAudio();
    a.testSpeech();
    expect(a.getTTSStatus().primed).toBe(true);
  });

  it('returns false when speechSynthesis is unavailable', async () => {
    delete (globalThis as any).speechSynthesis;
    const a = await loadAudio();
    expect(a.testSpeech()).toBe(false);
  });

  it('returns false (and logs) on engine throw', async () => {
    speakSpy.mockImplementationOnce(() => {
      throw new Error('engine dead');
    });
    const a = await loadAudio();
    expect(a.testSpeech()).toBe(false);
    expect(console.error).toHaveBeenCalled();
  });
});

// ---- cancelSpeech -----------------------------------------------------

describe('cancelSpeech', () => {
  it('calls speechSynthesis.cancel() once', async () => {
    const a = await loadAudio();
    a.cancelSpeech();
    expect(cancelSpy).toHaveBeenCalledTimes(1);
  });

  it('no-ops when speechSynthesis is unavailable', async () => {
    delete (globalThis as any).speechSynthesis;
    const a = await loadAudio();
    a.cancelSpeech();
    expect(cancelSpy).not.toHaveBeenCalled();
  });

  it('swallows engine.cancel() exceptions', async () => {
    cancelSpy.mockImplementationOnce(() => {
      throw new Error('cancel dead');
    });
    const a = await loadAudio();
    expect(() => a.cancelSpeech()).not.toThrow();
  });
});

// ---- voiceschanged listener ------------------------------------------

describe('voiceschanged listener', () => {
  it('subscribes at module load', async () => {
    await loadAudio();
    const calls = addEventSpy.mock.calls.filter((c) => c[0] === 'voiceschanged');
    expect(calls.length).toBe(1);
  });

  it('re-selects voice when voices change', async () => {
    const a = await loadAudio();
    // Fire prime to seed the cached voice as Samantha.
    a.primeTTS();
    speakSpy.mockReset();
    // Add a new preferred voice and trigger the listener manually.
    mockVoices = [{ name: 'Karen', lang: 'en-AU' } as SpeechSynthesisVoice];
    const handler = addEventSpy.mock.calls.find((c) => c[0] === 'voiceschanged')?.[1];
    expect(handler).toBeDefined();
    handler();
    a.speak('after change');
    const u = speakSpy.mock.calls[0][0] as FakeUtterance;
    expect(u.voice?.name).toBe('Karen');
  });
});
