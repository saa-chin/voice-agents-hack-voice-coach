/**
 * Mic capture and TTS for the Voice Coach web UI.
 *
 * Capture path:
 *   getUserMedia → MediaRecorder (webm/opus) → Blob
 *   → AudioContext.decodeAudioData → mono Float32 → resample to 16 kHz
 *   → Int16 PCM little-endian → base64
 *
 * The server expects exactly the same format the CLI uses (16 kHz mono int16
 * little-endian) so that the same `coach.build_prompt` / `_cactus_complete_audio`
 * path works.
 */

const TARGET_SAMPLE_RATE = 16000;

export interface RecorderHandle {
  start: () => Promise<void>;
  stop: () => Promise<{ pcm: Int16Array; durationMs: number }>;
  cancel: () => void;
  /** Latest 0..1 normalised loudness, updated while recording. */
  level: () => number;
}

/* v8 ignore start */
// createRecorder is a thin adapter over getUserMedia + MediaRecorder +
// AudioContext.decodeAudioData. None of those are available in jsdom/happy-dom,
// and verifying them only matters in a real browser. The pure DSP helpers it
// internally uses (audioBufferToInt16, floatToInt16) are exported and unit-tested.
export async function createRecorder(): Promise<RecorderHandle> {
  const stream = await navigator.mediaDevices.getUserMedia({
    audio: {
      channelCount: 1,
      echoCancellation: true,
      noiseSuppression: true,
      autoGainControl: true,
    },
  });

  const mimeType = pickMimeType();
  const recorder = new MediaRecorder(stream, mimeType ? { mimeType } : undefined);

  // Lightweight level meter via WebAudio analyser, independent of MediaRecorder.
  const audioCtx = new AudioContext();
  const source = audioCtx.createMediaStreamSource(stream);
  const analyser = audioCtx.createAnalyser();
  analyser.fftSize = 1024;
  source.connect(analyser);
  const buf = new Uint8Array(analyser.fftSize);

  let levelValue = 0;
  let levelTimer: number | null = null;

  const tickLevel = () => {
    analyser.getByteTimeDomainData(buf);
    let sum = 0;
    for (let i = 0; i < buf.length; i++) {
      const v = (buf[i] - 128) / 128;
      sum += v * v;
    }
    const rms = Math.sqrt(sum / buf.length);
    levelValue = Math.min(1, rms * 4); // scale: speaking ~0.05-0.2 RMS → 0.2-0.8 bar
  };

  const chunks: Blob[] = [];
  recorder.addEventListener('dataavailable', (e) => {
    if (e.data && e.data.size > 0) chunks.push(e.data);
  });

  let startedAt = 0;

  const stopStream = () => {
    if (levelTimer !== null) {
      window.clearInterval(levelTimer);
      levelTimer = null;
    }
    stream.getTracks().forEach((t) => t.stop());
    audioCtx.close().catch(() => {});
  };

  return {
    async start() {
      chunks.length = 0;
      // emit chunks every 250ms; keeps memory pressure off MediaRecorder.
      recorder.start(250);
      startedAt = performance.now();
      levelTimer = window.setInterval(tickLevel, 50) as unknown as number;
    },
    async stop() {
      const stopped = new Promise<void>((resolve) => {
        recorder.addEventListener('stop', () => resolve(), { once: true });
      });
      recorder.stop();
      await stopped;
      const durationMs = performance.now() - startedAt;
      stopStream();

      const blob = new Blob(chunks, { type: chunks[0]?.type ?? mimeType ?? 'audio/webm' });
      const arrayBuffer = await blob.arrayBuffer();

      // Decode via a fresh AudioContext at the device's default rate.
      const decodeCtx = new AudioContext();
      let audioBuffer: AudioBuffer;
      try {
        audioBuffer = await decodeCtx.decodeAudioData(arrayBuffer);
      } finally {
        decodeCtx.close().catch(() => {});
      }

      const pcm = audioBufferToInt16(audioBuffer, TARGET_SAMPLE_RATE);
      return { pcm, durationMs };
    },
    cancel() {
      try {
        if (recorder.state !== 'inactive') recorder.stop();
      } catch {
        // already stopped
      }
      stopStream();
    },
    level() {
      return levelValue;
    },
  };
}

/* v8 ignore stop */

/* v8 ignore start */
// Browser-detection branch — both arms covered only in real browsers.
function pickMimeType(): string | undefined {
  if (typeof MediaRecorder === 'undefined') return undefined;
  const candidates = [
    'audio/webm;codecs=opus',
    'audio/webm',
    'audio/ogg;codecs=opus',
    'audio/mp4',
  ];
  for (const c of candidates) {
    if (MediaRecorder.isTypeSupported(c)) return c;
  }
  return undefined;
}
/* v8 ignore stop */

/**
 * Convert a Web Audio AudioBuffer (multi-channel Float32 in [-1, 1]) to a
 * mono 16 kHz Int16Array. Linear-interpolation resample — adequate for speech.
 * Exported for direct unit testing; in production it's only called by
 * createRecorder() after decodeAudioData.
 */
export function audioBufferToInt16(buf: AudioBuffer, targetSr: number): Int16Array {
  const inSr = buf.sampleRate;
  const inLen = buf.length;
  const channels = buf.numberOfChannels;

  // Mix down to mono (Float32, [-1, 1]).
  const mono = new Float32Array(inLen);
  for (let ch = 0; ch < channels; ch++) {
    const data = buf.getChannelData(ch);
    for (let i = 0; i < inLen; i++) mono[i] += data[i];
  }
  if (channels > 1) {
    const inv = 1 / channels;
    for (let i = 0; i < inLen; i++) mono[i] *= inv;
  }

  if (inSr === targetSr) {
    return floatToInt16(mono);
  }

  // Linear-interpolation resample. Good enough for speech; cheap.
  const ratio = targetSr / inSr;
  const outLen = Math.max(1, Math.floor(inLen * ratio));
  const out = new Float32Array(outLen);
  const step = inSr / targetSr;
  for (let i = 0; i < outLen; i++) {
    const t = i * step;
    const lo = Math.floor(t);
    const hi = Math.min(inLen - 1, lo + 1);
    const frac = t - lo;
    out[i] = mono[lo] * (1 - frac) + mono[hi] * frac;
  }
  return floatToInt16(out);
}

/** Clip + scale Float32 [-1, 1] samples to Int16Array. Exported for tests. */
export function floatToInt16(float32: Float32Array): Int16Array {
  const out = new Int16Array(float32.length);
  for (let i = 0; i < float32.length; i++) {
    const x = Math.max(-1, Math.min(1, float32[i]));
    out[i] = x < 0 ? Math.round(x * 0x8000) : Math.round(x * 0x7fff);
  }
  return out;
}

export function int16ToBase64(pcm: Int16Array): string {
  const bytes = new Uint8Array(pcm.buffer, pcm.byteOffset, pcm.byteLength);
  // Build the binary string in chunks to avoid argument-length limits.
  let binary = '';
  const chunk = 0x8000;
  for (let i = 0; i < bytes.length; i += chunk) {
    binary += String.fromCharCode(...bytes.subarray(i, i + chunk));
  }
  return btoa(binary);
}

// ---- TTS ------------------------------------------------------------------
//
// Cross-browser quirks we work around here:
//
//   1. `speechSynthesis.getVoices()` returns [] until voices have asynchronously
//      loaded. We listen for `voiceschanged` once at module load.
//   2. The first `speak()` of a tab MUST be initiated inside a user-gesture
//      handler. Async WebSocket handlers don't count. `primeTTS()` performs
//      that first speak inside a click handler so subsequent async speaks work.
//   3. Calling `speechSynthesis.cancel()` immediately before `speak()` races on
//      Chrome and can kill the upcoming utterance. Don't cancel unless we
//      explicitly want to interrupt (e.g. user starts recording).

let preferredVoice: SpeechSynthesisVoice | null = null;
let primed = false;
let lastSpeakOk = true;
let lastSpeakError: string | null = null;

if (typeof window !== 'undefined' && 'speechSynthesis' in window) {
  // Trigger voice list loading + listen for late arrivals (Chrome).
  window.speechSynthesis.getVoices();
  window.speechSynthesis.addEventListener?.('voiceschanged', () => {
    preferredVoice = null;
    const v = ensureVoice();
    console.log('[coach tts] voiceschanged → re-selected', v?.name);
  });
}

function ensureVoice(): SpeechSynthesisVoice | null {
  if (typeof window === 'undefined' || !('speechSynthesis' in window)) return null;
  if (preferredVoice) return preferredVoice;
  const voices = window.speechSynthesis.getVoices();
  if (!voices.length) return null;
  const preferredNames = ['Samantha', 'Karen', 'Daniel', 'Moira', 'Google US English', 'Alex'];
  preferredVoice =
    voices.find((v) => preferredNames.includes(v.name) && v.lang.startsWith('en')) ??
    voices.find((v) => v.lang.startsWith('en')) ??
    voices[0];
  console.log(
    '[coach tts] selected voice:',
    preferredVoice?.name,
    preferredVoice?.lang,
    `(${voices.length} voices available)`,
  );
  return preferredVoice;
}

function makeUtterance(text: string): SpeechSynthesisUtterance {
  const utter = new SpeechSynthesisUtterance(text);
  const v = ensureVoice();
  if (v) utter.voice = v;
  // Setting `lang` works even when no specific voice was selected, and helps
  // the engine pick a sensible default. Crucial on Chrome where the engine
  // may otherwise pick a non-English voice and emit silence on English text.
  utter.lang = v?.lang ?? 'en-US';
  utter.rate = 0.95;
  utter.pitch = 1.0;
  utter.volume = 1.0;
  utter.onstart = () => {
    lastSpeakOk = true;
    lastSpeakError = null;
    console.log('[coach tts] ▶ speaking:', text);
  };
  utter.onend = () => {
    console.log('[coach tts] ■ ended:', text);
  };
  utter.onerror = (ev) => {
    const reason = (ev as SpeechSynthesisErrorEvent).error || 'unknown';
    lastSpeakOk = false;
    lastSpeakError = reason;
    console.warn('[coach tts] ✗ error:', reason, 'on:', text);
  };
  return utter;
}

/**
 * Prime the speech engine inside a user-gesture handler. Speaks an audible
 * "Let's begin" so the user gets immediate confirmation TTS works AND the
 * engine registers the user gesture for subsequent async calls.
 *
 * Safe to call multiple times. After the first successful prime, subsequent
 * calls are a no-op.
 */
export function primeTTS(): void {
  if (typeof window === 'undefined' || !('speechSynthesis' in window)) {
    console.warn('[coach tts] prime: speechSynthesis unavailable');
    return;
  }
  if (primed) {
    console.log('[coach tts] already primed');
    return;
  }
  try {
    const u = makeUtterance("Let's begin.");
    window.speechSynthesis.speak(u);
    primed = true;
    console.log('[coach tts] primed (queued audible utterance)');
  } catch (e) {
    console.warn('[coach tts] prime failed', e);
  }
}

/**
 * Speak `text`. Does NOT cancel any in-flight utterance — the new one will
 * queue. Use `cancelSpeech()` first if you want hard interruption (e.g. user
 * tapped the mic button to start recording).
 */
export function speak(text: string): void {
  if (typeof window === 'undefined' || !('speechSynthesis' in window)) {
    console.warn('[coach tts] speak: speechSynthesis unavailable');
    lastSpeakOk = false;
    lastSpeakError = 'speechSynthesis API not present in this browser';
    return;
  }
  const trimmed = text.trim();
  if (!trimmed) return;
  if (!primed) {
    console.warn(
      '[coach tts] speak() before primeTTS() — browser may suppress this. ' +
        'Make sure primeTTS() ran inside a click handler.',
    );
  }
  try {
    const u = makeUtterance(trimmed);
    window.speechSynthesis.speak(u);
  } catch (e) {
    lastSpeakOk = false;
    lastSpeakError = e instanceof Error ? e.message : String(e);
    console.error('[coach tts] threw', e);
  }
}

/**
 * Convenience for the diagnostic "Test voice" button — speaks at full volume,
 * primes if needed, and returns whether the call was made.
 */
export function testSpeech(): boolean {
  if (typeof window === 'undefined' || !('speechSynthesis' in window)) return false;
  primed = true; // count this click as the prime
  try {
    const u = makeUtterance('Voice test, can you hear me?');
    window.speechSynthesis.speak(u);
    console.log('[coach tts] test utterance queued');
    return true;
  } catch (e) {
    console.error('[coach tts] test failed', e);
    return false;
  }
}

export function cancelSpeech(): void {
  if (typeof window !== 'undefined' && 'speechSynthesis' in window) {
    try {
      window.speechSynthesis.cancel();
    } catch {
      /* ignore */
    }
  }
}

export function isTTSAvailable(): boolean {
  return typeof window !== 'undefined' && 'speechSynthesis' in window;
}

export function getTTSStatus(): { ok: boolean; primed: boolean; lastError: string | null } {
  return { ok: lastSpeakOk, primed, lastError: lastSpeakError };
}
