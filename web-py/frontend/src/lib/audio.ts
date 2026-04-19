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

// MUST match cli/coach.py:SR. The Python backend rejects audio at any other
// sample rate. If you change one, change both.
const TARGET_SAMPLE_RATE = 16000;

/**
 * Per-frame audio analysis snapshot. Updated ~20 Hz while recording.
 * Numbers are computed against the AudioContext sample rate, NOT the 16 kHz
 * target — this is purely for live visualisation and VAD; the encoded PCM
 * sent to the backend is still the resampled 16 kHz mono int16.
 */
export interface AudioAnalysis {
  /** RMS amplitude of the latest frame, 0..1 (linear). */
  rms: number;
  /** dBFS of the latest frame: 20 * log10(rms). -Infinity when silent. */
  dbfs: number;
  /** Per-band normalised energy 0..1, log-spaced across the voice range. */
  bands: number[];
  /** Estimated dominant fundamental in 80..400 Hz, or null if unvoiced. */
  pitchHz: number | null;
  /** Single-frame voiced classification (RMS above start threshold). */
  voiced: boolean;
  /** VAD state with hysteresis — true between speech onset and hangover. */
  speaking: boolean;
  /** Cumulative fraction of frames classified as voiced since recording
   * started, 0..1. Useful for a stable "% voiced" score. */
  voicedRatio: number;
  /** AudioContext sample rate (typically 44.1 / 48 kHz). */
  sampleRate: number;
}

export interface VadOptions {
  /** Master toggle. When false, `speaking` always tracks `voiced`. */
  enabled?: boolean;
  /** RMS that counts as voiced. Default 0.025. */
  rmsStart?: number;
  /** RMS below which we count silence (use < rmsStart for hysteresis). */
  rmsEnd?: number;
  /** Consecutive voiced frames needed to enter "speaking" state. */
  startFrames?: number;
  /** ms of continuous silence after speech before we trigger onSpeechEnd. */
  endHangoverMs?: number;
  /** Fired on speech onset (after startFrames voiced frames). */
  onSpeechStart?: () => void;
  /** Fired after `endHangoverMs` of silence following speech. */
  onSpeechEnd?: () => void;
}

export interface CreateRecorderOptions {
  vad?: VadOptions;
}

const DEFAULT_VAD: Required<Omit<VadOptions, 'onSpeechStart' | 'onSpeechEnd'>> = {
  enabled: false,
  rmsStart: 0.025,
  rmsEnd: 0.012,
  startFrames: 3,
  endHangoverMs: 900,
};

const NUM_BANDS = 8;
// Log-spaced band edges across the speech-relevant range. Hz.
const BAND_EDGES = [80, 160, 320, 500, 800, 1300, 2100, 3300, 5000];

export interface RecorderHandle {
  start: () => Promise<void>;
  stop: () => Promise<{ pcm: Int16Array; durationMs: number }>;
  cancel: () => void;
  /** Latest 0..1 normalised loudness, updated while recording. */
  level: () => number;
  /** Latest full analysis snapshot (rms/dbfs/bands/pitch/vad). */
  analysis: () => AudioAnalysis;
}

/* v8 ignore start */
// createRecorder is a thin adapter over getUserMedia + MediaRecorder +
// AudioContext.decodeAudioData. None of those are available in jsdom/happy-dom,
// and verifying them only matters in a real browser. The pure DSP helpers it
// internally uses (audioBufferToInt16, floatToInt16) are exported and unit-tested.
export async function createRecorder(
  opts: CreateRecorderOptions = {},
): Promise<RecorderHandle> {
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

  // Lightweight live analyser via WebAudio, independent of MediaRecorder.
  // fftSize 2048 gives us ~23 Hz bin width at 48 kHz — enough resolution for
  // a usable pitch dominant-bin estimate while staying cheap to compute.
  const audioCtx = new AudioContext();
  const source = audioCtx.createMediaStreamSource(stream);
  const analyser = audioCtx.createAnalyser();
  analyser.fftSize = 2048;
  analyser.smoothingTimeConstant = 0.6;
  source.connect(analyser);

  const sampleRate = audioCtx.sampleRate;
  const fftSize = analyser.fftSize;
  const binCount = analyser.frequencyBinCount;
  const binHz = sampleRate / fftSize;
  const timeBuf = new Uint8Array(fftSize);
  const freqBuf = new Float32Array(binCount); // dB values, typically -100..0

  // Pre-compute band bin ranges (NUM_BANDS log-spaced over voice frequencies).
  const bandRanges: Array<[number, number]> = [];
  for (let i = 0; i < NUM_BANDS; i++) {
    const lo = Math.max(0, Math.floor(BAND_EDGES[i] / binHz));
    const hi = Math.min(binCount - 1, Math.ceil(BAND_EDGES[i + 1] / binHz));
    bandRanges.push([lo, Math.max(lo, hi)]);
  }

  // Pitch search range: 80–400 Hz (typical adult speech F0).
  const pitchMinBin = Math.max(1, Math.floor(80 / binHz));
  const pitchMaxBin = Math.min(binCount - 1, Math.ceil(400 / binHz));

  const vadCfg = { ...DEFAULT_VAD, ...(opts.vad ?? {}) };
  const onSpeechStart = opts.vad?.onSpeechStart;
  const onSpeechEnd = opts.vad?.onSpeechEnd;

  let levelValue = 0;
  let levelTimer: number | null = null;
  let voicedStreak = 0;
  let silenceStartedAt: number | null = null;
  let speakingState = false;
  let voicedFrames = 0;
  let totalFrames = 0;
  let lastAnalysis: AudioAnalysis = {
    rms: 0,
    dbfs: -Infinity,
    bands: new Array(NUM_BANDS).fill(0),
    pitchHz: null,
    voiced: false,
    speaking: false,
    voicedRatio: 0,
    sampleRate,
  };

  const tickLevel = () => {
    analyser.getByteTimeDomainData(timeBuf);
    let sum = 0;
    for (let i = 0; i < timeBuf.length; i++) {
      const v = (timeBuf[i] - 128) / 128;
      sum += v * v;
    }
    const rms = Math.sqrt(sum / timeBuf.length);
    levelValue = Math.min(1, rms * 4); // scale: speaking ~0.05-0.2 RMS → 0.2-0.8 bar

    // ---- spectrum + pitch -------------------------------------------------
    analyser.getFloatFrequencyData(freqBuf);
    const bands = new Array(NUM_BANDS);
    for (let b = 0; b < NUM_BANDS; b++) {
      const [lo, hi] = bandRanges[b];
      let acc = 0;
      let n = 0;
      for (let i = lo; i <= hi; i++) {
        // Convert dB → linear-ish 0..1 (clamp -90 dB floor → 0, 0 dB → 1).
        const norm = Math.max(0, (freqBuf[i] + 90) / 90);
        acc += norm;
        n += 1;
      }
      bands[b] = n ? Math.min(1, acc / n) : 0;
    }

    let pitchHz: number | null = null;
    if (rms > vadCfg.rmsEnd) {
      let maxDb = -Infinity;
      let maxIdx = -1;
      for (let i = pitchMinBin; i <= pitchMaxBin; i++) {
        if (freqBuf[i] > maxDb) {
          maxDb = freqBuf[i];
          maxIdx = i;
        }
      }
      // Require at least some signal above the noise floor before reporting.
      if (maxIdx > 0 && maxDb > -65) pitchHz = maxIdx * binHz;
    }

    // ---- VAD with hysteresis ---------------------------------------------
    const voiced = rms >= vadCfg.rmsStart;
    if (vadCfg.enabled) {
      if (voiced) {
        voicedStreak += 1;
        silenceStartedAt = null;
        if (!speakingState && voicedStreak >= vadCfg.startFrames) {
          speakingState = true;
          try { onSpeechStart?.(); } catch (e) { console.error('[coach vad] onSpeechStart threw', e); }
        }
      } else if (rms < vadCfg.rmsEnd) {
        voicedStreak = 0;
        if (speakingState) {
          if (silenceStartedAt === null) silenceStartedAt = performance.now();
          else if (performance.now() - silenceStartedAt >= vadCfg.endHangoverMs) {
            speakingState = false;
            silenceStartedAt = null;
            try { onSpeechEnd?.(); } catch (e) { console.error('[coach vad] onSpeechEnd threw', e); }
          }
        }
      }
      // Between rmsEnd and rmsStart: dead-band, hold current state.
    } else {
      speakingState = voiced;
    }

    totalFrames += 1;
    if (voiced) voicedFrames += 1;

    lastAnalysis = {
      rms,
      dbfs: rms > 0 ? 20 * Math.log10(rms) : -Infinity,
      bands,
      pitchHz,
      voiced,
      speaking: speakingState,
      voicedRatio: totalFrames > 0 ? voicedFrames / totalFrames : 0,
      sampleRate,
    };
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
      voicedStreak = 0;
      silenceStartedAt = null;
      speakingState = false;
      voicedFrames = 0;
      totalFrames = 0;
      // emit chunks every 250ms; keeps memory pressure off MediaRecorder.
      recorder.start(250);
      startedAt = performance.now();
      // 50ms ≈ 20 Hz refresh — smooth visualisers without burning CPU.
      levelTimer = window.setInterval(tickLevel, 50) as unknown as number;
    },
    async stop() {
      // If MediaRecorder isn't actively recording, the stop event never fires
      // and we'd hang forever. Bail with a clear message instead.
      if (recorder.state === 'inactive') {
        stopStream();
        throw new Error('Recorder was not active (already stopped or never started).');
      }
      const stopped = new Promise<void>((resolve) => {
        recorder.addEventListener('stop', () => resolve(), { once: true });
      });
      recorder.stop();
      await stopped;
      const durationMs = performance.now() - startedAt;
      stopStream();

      const blobType = chunks[0]?.type ?? mimeType ?? 'audio/webm';
      const blob = new Blob(chunks, { type: blobType });
      console.log('[coach audio] stop:', {
        chunks: chunks.length,
        blob_bytes: blob.size,
        blob_type: blobType,
        duration_ms: Math.round(durationMs),
      });

      // No audio captured at all — usually means the user clicked stop
      // before the recorder produced any data (sub-100ms tap).
      if (blob.size === 0 || chunks.length === 0) {
        throw new Error(
          `No audio captured (${chunks.length} chunks, ${durationMs.toFixed(0)}ms). ` +
            'Hold the mic for at least half a second.',
        );
      }

      const arrayBuffer = await blob.arrayBuffer();

      // Decode via a fresh AudioContext at the device's default rate.
      const decodeCtx = new AudioContext();
      let audioBuffer: AudioBuffer;
      try {
        audioBuffer = await decodeCtx.decodeAudioData(arrayBuffer);
      } catch (e) {
        // Replace the bare browser "Decoding failed" with something we can
        // actually act on — including codec / size context.
        const reason = e instanceof Error ? e.message : String(e);
        throw new Error(
          `Audio decode failed (mime=${blobType}, ${blob.size} bytes): ${reason}. ` +
            'This usually means the recording was too short or the codec was unsupported.',
        );
      } finally {
        decodeCtx.close().catch(() => {});
      }

      const pcm = audioBufferToInt16(audioBuffer, TARGET_SAMPLE_RATE);
      console.log('[coach audio] decoded:', {
        decoded_samples: audioBuffer.length,
        decoded_sr: audioBuffer.sampleRate,
        decoded_channels: audioBuffer.numberOfChannels,
        out_pcm_samples: pcm.length,
        out_seconds: (pcm.length / TARGET_SAMPLE_RATE).toFixed(2),
      });
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
    analysis() {
      return lastAnalysis;
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
// Track utterances we've queued but not yet seen end/error for. Used to drive
// the auto mic gating: while we're still speaking, the mic must stay muted.
let pendingUtterances = 0;

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

function makeUtterance(
  text: string,
  onEnd?: (info: { ok: boolean; reason?: string }) => void,
): SpeechSynthesisUtterance {
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
  // Reentrancy guard: each terminal event (end/error) should only fire the
  // user callback once and only decrement the pending counter once.
  let settled = false;
  const settle = (ok: boolean, reason?: string) => {
    if (settled) return;
    settled = true;
    pendingUtterances = Math.max(0, pendingUtterances - 1);
    try { onEnd?.({ ok, reason }); } catch (e) { console.error('[coach tts] onEnd threw', e); }
  };
  utter.onstart = () => {
    lastSpeakOk = true;
    lastSpeakError = null;
    console.log('[coach tts] ▶ speaking:', text);
  };
  utter.onend = () => {
    console.log('[coach tts] ■ ended:', text);
    settle(true);
  };
  utter.onerror = (ev) => {
    const reason = (ev as SpeechSynthesisErrorEvent).error || 'unknown';
    lastSpeakOk = false;
    lastSpeakError = reason;
    console.warn('[coach tts] ✗ error:', reason, 'on:', text);
    settle(false, reason);
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
 *
 * If `opts.onEnd` is provided it will be invoked exactly once when the
 * utterance finishes (`ok: true`) or errors out (`ok: false`). If the engine
 * is unavailable or `speak()` throws synchronously, `onEnd` is invoked with
 * `ok: false` so callers can still progress (e.g. arm the mic anyway).
 */
export function speak(
  text: string,
  opts?: { onEnd?: (info: { ok: boolean; reason?: string }) => void },
): void {
  if (typeof window === 'undefined' || !('speechSynthesis' in window)) {
    console.warn('[coach tts] speak: speechSynthesis unavailable');
    lastSpeakOk = false;
    lastSpeakError = 'speechSynthesis API not present in this browser';
    try { opts?.onEnd?.({ ok: false, reason: 'no-speechSynthesis' }); } catch {}
    return;
  }
  const trimmed = text.trim();
  if (!trimmed) {
    try { opts?.onEnd?.({ ok: true }); } catch {}
    return;
  }
  if (!primed) {
    console.warn(
      '[coach tts] speak() before primeTTS() — browser may suppress this. ' +
        'Make sure primeTTS() ran inside a click handler.',
    );
  }
  try {
    const u = makeUtterance(trimmed, opts?.onEnd);
    pendingUtterances += 1;
    window.speechSynthesis.speak(u);
  } catch (e) {
    lastSpeakOk = false;
    lastSpeakError = e instanceof Error ? e.message : String(e);
    console.error('[coach tts] threw', e);
    try { opts?.onEnd?.({ ok: false, reason: lastSpeakError }); } catch {}
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
  // After cancel(), we won't get end events for in-flight utterances, so
  // reset the counter ourselves to keep auto-mic gating accurate.
  pendingUtterances = 0;
}

export function isTTSAvailable(): boolean {
  return typeof window !== 'undefined' && 'speechSynthesis' in window;
}

// ---- Server-rendered TTS playback ----------------------------------------
//
// The backend may render the coach's reply via macOS `say` and ship a
// WAV down the WebSocket. Playing that here keeps the entire voice
// path (in AND out) on the user's machine — no Chrome cloud-TTS, no
// platform-dependent voice quality. If the WAV doesn't arrive (e.g.
// `say` not present on a Linux backend), the caller still has
// `speak()` above as a graceful fallback.

let currentServerAudio: HTMLAudioElement | null = null;

/** Stop any in-flight server-rendered TTS clip. Safe to call when nothing
 * is playing. Used to interrupt the coach when the user starts a new
 * recording or cancels the session. */
export function cancelServerSpeech(): void {
  if (currentServerAudio) {
    try {
      currentServerAudio.pause();
      currentServerAudio.src = '';
    } catch {
      /* ignore */
    }
    currentServerAudio = null;
  }
}

/**
 * Play a base64-encoded WAV. Returns a promise that resolves when
 * playback finishes (or rejects with the underlying error). Any prior
 * server-rendered clip is hard-cancelled first so two coach replies
 * never overlap.
 */
export function playWavBase64(wavB64: string): Promise<void> {
  return new Promise<void>((resolve, reject) => {
    if (!wavB64) {
      resolve();
      return;
    }
    cancelServerSpeech();
    let url: string;
    try {
      // Decode base64 → Uint8Array → Blob → object URL. We don't use
      // `data:audio/wav;base64,...` because some browsers refuse to
      // play long data URLs inside <audio>.
      const bin = atob(wavB64);
      const buf = new Uint8Array(bin.length);
      for (let i = 0; i < bin.length; i += 1) buf[i] = bin.charCodeAt(i);
      const blob = new Blob([buf], { type: 'audio/wav' });
      url = URL.createObjectURL(blob);
    } catch (e) {
      reject(e instanceof Error ? e : new Error(String(e)));
      return;
    }
    const audio = new Audio(url);
    currentServerAudio = audio;
    const cleanup = () => {
      if (currentServerAudio === audio) currentServerAudio = null;
      try {
        URL.revokeObjectURL(url);
      } catch {
        /* ignore */
      }
    };
    audio.addEventListener('ended', () => {
      cleanup();
      resolve();
    });
    audio.addEventListener('error', () => {
      cleanup();
      reject(new Error('audio playback failed'));
    });
    audio.play().catch((e) => {
      cleanup();
      reject(e instanceof Error ? e : new Error(String(e)));
    });
  });
}

export function isServerSpeaking(): boolean {
  return currentServerAudio !== null && !currentServerAudio.paused;
}

/**
 * Whether TTS is currently producing (or about to produce) audio. The mic
 * auto-arm logic uses this to avoid capturing the coach's own voice.
 */
export function isTTSSpeaking(): boolean {
  if (typeof window === 'undefined' || !('speechSynthesis' in window)) return false;
  // Combine our pending counter with the engine's own flags to handle the
  // brief window between speak() returning and onstart firing.
  const ss = window.speechSynthesis;
  return pendingUtterances > 0 || ss.speaking || ss.pending;
}

export function getTTSStatus(): { ok: boolean; primed: boolean; lastError: string | null } {
  return { ok: lastSpeakOk, primed, lastError: lastSpeakError };
}

// ---- Live transcription (Web Speech API) ---------------------------------
//
// We layer the browser's `SpeechRecognition` API on top of the same mic
// stream the recorder uses. It runs entirely client-side (no audio leaves
// the page), runs in parallel with `MediaRecorder` (Chrome and Safari both
// allow simultaneous getUserMedia + SpeechRecognition consumers), and gives
// us interim transcripts that update many times per second — perfect for
// "see what you said as you say it".
//
// Notes:
//  • Firefox does not implement SpeechRecognition. `isRecognitionAvailable()`
//    returns false there and the UI quietly hides the transcript card.
//  • Some Chrome/Safari versions throw "InvalidStateError" if you call
//    `start()` while a previous instance is still terminating. We swallow
//    that and let the next turn create a fresh instance.

export interface RecognitionUpdate {
  /** Concatenated final transcripts so far (only changes when a chunk
   * is committed by the engine). */
  final: string;
  /** Latest tentative transcript that may still change. Empty when the
   * engine isn't currently mid-utterance. */
  interim: string;
}

export interface RecognizerHandle {
  start: () => void;
  stop: () => void;
  /** Hard abort — drops any pending interim transcript. */
  cancel: () => void;
  /** Latest snapshot, also passed to onUpdate. */
  transcript: () => RecognitionUpdate;
}

export interface CreateRecognizerOptions {
  lang?: string;
  onUpdate?: (info: RecognitionUpdate) => void;
  onError?: (err: string) => void;
}

export function isRecognitionAvailable(): boolean {
  if (typeof window === 'undefined') return false;
  return (
    'SpeechRecognition' in window || 'webkitSpeechRecognition' in window
  );
}

/* v8 ignore start */
// Real-browser-only path; happy-dom doesn't ship SpeechRecognition.
export function createRecognizer(
  opts: CreateRecognizerOptions = {},
): RecognizerHandle | null {
  if (!isRecognitionAvailable()) return null;
  const Ctor: any =
    (window as any).SpeechRecognition ?? (window as any).webkitSpeechRecognition;
  let r: any;
  try {
    r = new Ctor();
  } catch (e) {
    console.warn('[coach asr] construct failed', e);
    return null;
  }
  r.lang = opts.lang ?? 'en-US';
  r.continuous = true;
  r.interimResults = true;
  // Some Chrome builds default this to a non-zero value and surface only
  // "best" results; we always want the top hypothesis for live display.
  r.maxAlternatives = 1;

  let finalText = '';
  let interimText = '';
  let started = false;

  const emit = () => {
    try {
      opts.onUpdate?.({ final: finalText, interim: interimText });
    } catch (e) {
      console.error('[coach asr] onUpdate threw', e);
    }
  };

  r.onresult = (ev: any) => {
    interimText = '';
    for (let i = ev.resultIndex; i < ev.results.length; i++) {
      const res = ev.results[i];
      const text = (res[0]?.transcript ?? '').trim();
      if (!text) continue;
      if (res.isFinal) {
        finalText = (finalText ? finalText + ' ' : '') + text;
      } else {
        interimText += (interimText ? ' ' : '') + text;
      }
    }
    emit();
  };
  r.onerror = (ev: any) => {
    const reason = ev?.error || 'unknown';
    // "no-speech" and "aborted" are routine and not worth surfacing.
    if (reason !== 'no-speech' && reason !== 'aborted') {
      console.warn('[coach asr] error', reason);
      try { opts.onError?.(reason); } catch {}
    }
  };
  r.onend = () => {
    started = false;
    console.log('[coach asr] ■ ended');
  };

  return {
    start() {
      if (started) return;
      finalText = '';
      interimText = '';
      try {
        r.start();
        started = true;
        console.log('[coach asr] ▶ started');
      } catch (e) {
        // "InvalidStateError" if a previous run hasn't fully released.
        console.warn('[coach asr] start failed', e);
      }
    },
    stop() {
      if (!started) return;
      try { r.stop(); } catch { /* swallow */ }
    },
    cancel() {
      if (!started) return;
      try { r.abort(); } catch { /* swallow */ }
      started = false;
    },
    transcript() {
      return { final: finalText, interim: interimText };
    },
  };
}
/* v8 ignore stop */
