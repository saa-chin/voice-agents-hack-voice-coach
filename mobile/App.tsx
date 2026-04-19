import React, { useEffect, useRef, useState } from 'react';
import {
  Alert,
  PermissionsAndroid,
  Platform,
  ScrollView,
  StatusBar,
  StyleSheet,
  Text,
  TouchableOpacity,
  useColorScheme,
  View,
} from 'react-native';
import {
  SafeAreaProvider,
  useSafeAreaInsets,
} from 'react-native-safe-area-context';
import { useCactusSTT } from 'cactus-react-native';
import { NitroModules } from 'react-native-nitro-modules';
import AudioRecord from 'react-native-audio-record';
import { Buffer } from 'buffer';

// Single-model pipeline: Gemma 4 reasons over audio directly (no whisper
// step). https://docs.cactuscompute.com/latest/blog/gemma4/
//
// Streaming approach: rolling-window batch transcription.
// We *don't* use Cactus's streamTranscribeProcess — that's a whisper-style
// LocalAgreement loop that accumulates audio internally and re-decodes the
// whole growing buffer on every call. Gemma's audio encoder isn't built for
// that and OOMs (malloc fail) after ~10-20s on iPhone.
//
// Instead: keep recording into a JS-side ring buffer, and every
// TRANSCRIBE_INTERVAL_MS fire one stateless `transcribe()` call on the most
// recent MAX_WINDOW_S of audio. Per the blog, Gemma 4 does 30s of audio in
// ~0.3s on M-series ARM, so this stays well under realtime.
const MODEL_SLUG = 'gemma-4-e2b-it';
const QUANT: 'int4' | 'int8' = 'int4';
const INTERNAL_NAME = `${MODEL_SLUG}-${QUANT}`;
const WEIGHTS_URL = `https://huggingface.co/Cactus-Compute/gemma-4-E2B-it/resolve/v1.13/weights/${INTERNAL_NAME}.zip`;

const VOICE_PROMPT =
  'Transcribe the audio verbatim. Output only the spoken words, exactly as heard. No commentary, no analysis, no extra text.';

const RECORD_OPTIONS = {
  sampleRate: 16000,
  channels: 1,
  bitsPerSample: 16,
  audioSource: 6,
  wavFile: 'voice-coach-clip.wav',
};

const SAMPLE_RATE = 16000;
const BYTES_PER_SAMPLE = 2;
const BYTES_PER_SECOND = SAMPLE_RATE * BYTES_PER_SAMPLE; // 32000

const TRANSCRIBE_INTERVAL_MS = 1500; // re-transcribe rolling window this often
const MAX_WINDOW_S = 20; // cap context to last 20s to keep latency bounded
const MIN_WINDOW_BYTES = BYTES_PER_SECOND; // need >=1s before first pass

type CactusFS = {
  modelExists(model: string): Promise<boolean>;
  downloadModel(
    model: string,
    url: string,
    onProgress?: (progress: number) => void,
  ): Promise<void>;
};
const cactusFS = NitroModules.createHybridObject<CactusFS>('CactusFileSystem');

async function ensureMicPermission(): Promise<boolean> {
  if (Platform.OS !== 'android') return true;
  const granted = await PermissionsAndroid.request(
    PermissionsAndroid.PERMISSIONS.RECORD_AUDIO,
    {
      title: 'Microphone permission',
      message: 'AI Voice Coach needs your microphone.',
      buttonPositive: 'OK',
    },
  );
  return granted === PermissionsAndroid.RESULTS.GRANTED;
}

function App() {
  const isDarkMode = useColorScheme() === 'dark';
  return (
    <SafeAreaProvider>
      <StatusBar barStyle={isDarkMode ? 'light-content' : 'dark-content'} />
      <AppContent isDarkMode={isDarkMode} />
    </SafeAreaProvider>
  );
}

function AppContent({ isDarkMode }: { isDarkMode: boolean }) {
  const insets = useSafeAreaInsets();
  const stt = useCactusSTT({ model: MODEL_SLUG, options: { quantization: QUANT } });

  const [downloading, setDownloading] = useState(false);
  const [downloadProgress, setDownloadProgress] = useState(0);
  const [modelReady, setModelReady] = useState(false);

  const [isRecording, setIsRecording] = useState(false);
  const [transcript, setTranscript] = useState('');
  const [lastLatencyMs, setLastLatencyMs] = useState<number | null>(null);
  const [localError, setLocalError] = useState<string | null>(null);
  const [diag, setDiag] = useState<string>('');

  const recorderReady = useRef(false);
  const recordStartedAt = useRef<number>(0);
  const pcmBytes = useRef<number[]>([]);
  const transcribingRef = useRef(false);
  const isRecordingRef = useRef(false);
  const sttRef = useRef(stt);

  useEffect(() => {
    sttRef.current = stt;
  }, [stt]);

  useEffect(() => {
    if (!recorderReady.current) {
      AudioRecord.init(RECORD_OPTIONS);
      AudioRecord.on('data', (data: string) => {
        const chunk = Buffer.from(data, 'base64');
        for (let i = 0; i < chunk.length; i++) {
          pcmBytes.current.push(chunk[i]);
        }
      });
      recorderReady.current = true;
    }
  }, []);

  // Manual Gemma 4 download
  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const exists = await cactusFS.modelExists(INTERNAL_NAME);
        if (cancelled) return;
        if (exists) {
          setModelReady(true);
          return;
        }
        setDownloading(true);
        await cactusFS.downloadModel(INTERNAL_NAME, WEIGHTS_URL, (p: number) => {
          if (!cancelled) setDownloadProgress(p);
        });
        if (!cancelled) {
          setDownloading(false);
          setModelReady(true);
        }
      } catch (err: any) {
        if (!cancelled) {
          setDownloading(false);
          setLocalError(`Gemma 4 download failed: ${err?.message ?? err}`);
        }
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  // Run one transcription pass on the most recent window of audio.
  const runTranscribePass = async () => {
    if (transcribingRef.current) return;
    const buf = pcmBytes.current;
    if (buf.length < MIN_WINDOW_BYTES) return;

    const maxBytes = MAX_WINDOW_S * BYTES_PER_SECOND;
    const start = Math.max(0, buf.length - maxBytes);
    // Snapshot via slice so the buffer can keep growing during inference.
    const window = buf.slice(start);

    transcribingRef.current = true;
    const t0 = Date.now();
    try {
      const result = await sttRef.current.transcribe({
        audio: window,
        prompt: VOICE_PROMPT,
        options: { useVad: true, maxTokens: 256, temperature: 0.0 },
      });
      const text = (result.response ?? '').trim();
      if (text) setTranscript(text);
      setLastLatencyMs(Date.now() - t0);
    } catch (err: any) {
      setLocalError(err?.message ?? 'transcribe() failed.');
    } finally {
      transcribingRef.current = false;
    }
  };

  // Drive the rolling-window passes while recording.
  useEffect(() => {
    if (!isRecording) return;
    const id = setInterval(runTranscribePass, TRANSCRIBE_INTERVAL_MS);
    return () => clearInterval(id);
  }, [isRecording]);

  const handleStart = async () => {
    setLocalError(null);
    setDiag('');
    setTranscript('');
    setLastLatencyMs(null);
    if (!modelReady) {
      Alert.alert('Gemma 4 not ready', 'Wait for weights to finish downloading.');
      return;
    }
    const ok = await ensureMicPermission();
    if (!ok) {
      setLocalError('Microphone permission denied.');
      return;
    }
    try {
      pcmBytes.current = [];
      transcribingRef.current = false;
      AudioRecord.start();
      recordStartedAt.current = Date.now();
      isRecordingRef.current = true;
      setIsRecording(true);
    } catch (err: any) {
      setLocalError(err?.message ?? 'Failed to start recording.');
    }
  };

  const handleStop = async () => {
    try {
      await AudioRecord.stop();
      isRecordingRef.current = false;
      setIsRecording(false);

      const durationMs = Date.now() - recordStartedAt.current;
      const totalBytes = pcmBytes.current.length;
      setDiag(
        `bytes=${totalBytes} samples=${totalBytes / 2} dur=${durationMs}ms`,
      );

      // One final pass on the full buffer (cap at MAX_WINDOW_S) so the user
      // sees the cleanest version once we're done.
      while (transcribingRef.current) {
        await new Promise((r) => setTimeout(r, 50));
      }
      await runTranscribePass();
    } catch (err: any) {
      setLocalError(err?.message ?? 'Failed to stop recording.');
    }
  };

  const onMicPress = () => {
    if (isRecording) {
      handleStop();
    } else {
      handleStart();
    }
  };

  const palette = isDarkMode ? darkPalette : lightPalette;
  const status = describeStatus({
    downloading,
    downloadProgress,
    modelReady,
    isRecording,
    transcribing: transcribingRef.current,
    error: stt.error ?? localError,
  });

  const micDisabled = !modelReady || downloading;
  const hasText = transcript.length > 0;

  return (
    <View
      style={[
        styles.container,
        {
          paddingTop: insets.top,
          paddingBottom: insets.bottom,
          backgroundColor: palette.bg,
        },
      ]}
    >
      <View style={styles.header}>
        <Text style={[styles.title, { color: palette.text }]}>AI Voice Coach</Text>
        <Text style={[styles.subtitle, { color: palette.muted }]}>
          {MODEL_SLUG} ({QUANT}) · rolling-window streaming
        </Text>
      </View>

      <ScrollView style={styles.scroll} contentContainerStyle={styles.scrollContent}>
        <View
          style={[
            styles.card,
            {
              backgroundColor: palette.card,
              borderColor: palette.accent,
              borderLeftWidth: 3,
            },
          ]}
        >
          <Text style={[styles.cardLabel, { color: palette.muted }]}>Live Transcript</Text>
          {hasText ? (
            <Text style={[styles.cardBody, { color: palette.text }]}>{transcript}</Text>
          ) : (
            <Text style={[styles.cardBody, { color: palette.muted }]}>
              {isRecording ? 'Listening… speak now' : 'Tap mic and speak.'}
            </Text>
          )}
          {lastLatencyMs != null ? (
            <Text style={[styles.latency, { color: palette.muted }]}>
              last pass: {lastLatencyMs}ms
            </Text>
          ) : null}
        </View>

        {(stt.error || localError) ? (
          <View style={styles.errorBox}>
            <Text style={styles.errorLabel}>ERROR</Text>
            <Text selectable style={styles.errorText}>
              {stt.error ?? localError}
            </Text>
          </View>
        ) : null}

        {diag ? (
          <View style={[styles.diagBox, { borderColor: palette.border, backgroundColor: palette.card }]}>
            <Text style={[styles.diagLabel, { color: palette.muted }]}>MIC DIAGNOSTIC</Text>
            <Text selectable style={[styles.diag, { color: palette.text }]}>{diag}</Text>
          </View>
        ) : null}
      </ScrollView>

      <View style={styles.footer}>
        <Text
          style={[
            styles.statusText,
            { color: status.tone === 'error' ? '#ff6b6b' : palette.muted },
          ]}
        >
          {status.label}
        </Text>

        <TouchableOpacity
          accessibilityRole="button"
          accessibilityLabel={isRecording ? 'Stop streaming' : 'Start streaming'}
          onPress={onMicPress}
          disabled={micDisabled}
          activeOpacity={0.85}
          style={[
            styles.mic,
            {
              backgroundColor: isRecording ? '#e53935' : palette.accent,
              opacity: micDisabled ? 0.4 : 1,
            },
          ]}
        >
          <Text style={styles.micIcon}>{isRecording ? '■' : '🎤'}</Text>
        </TouchableOpacity>

        <Text style={[styles.hint, { color: palette.muted }]}>
          {isRecording ? 'Tap to stop' : 'Tap to stream live'}
        </Text>
      </View>
    </View>
  );
}

type StatusInput = {
  downloading: boolean;
  downloadProgress: number;
  modelReady: boolean;
  isRecording: boolean;
  transcribing: boolean;
  error: string | null;
};

function describeStatus(s: StatusInput): { label: string; tone: 'info' | 'error' } {
  if (s.error) return { label: s.error, tone: 'error' };
  if (s.downloading) {
    return {
      label: `Downloading Gemma 4 (≈4.4 GB)… ${Math.round(s.downloadProgress * 100)}%`,
      tone: 'info',
    };
  }
  if (!s.modelReady) return { label: 'Preparing Gemma 4…', tone: 'info' };
  if (s.isRecording) return { label: 'Streaming…', tone: 'info' };
  if (s.transcribing) return { label: 'Finalizing…', tone: 'info' };
  return { label: 'Ready', tone: 'info' };
}

const lightPalette = {
  bg: '#f7f8fa',
  text: '#111418',
  muted: '#5a6373',
  card: '#ffffff',
  border: '#e3e6ec',
  accent: '#3f7afe',
};

const darkPalette = {
  bg: '#0b0f17',
  text: '#f4f6fb',
  muted: '#9aa3b2',
  card: '#141a25',
  border: '#1f2735',
  accent: '#3f7afe',
};

const styles = StyleSheet.create({
  container: { flex: 1 },
  header: { paddingHorizontal: 24, paddingTop: 16, paddingBottom: 8 },
  title: { fontSize: 28, fontWeight: '700', letterSpacing: -0.5 },
  subtitle: { marginTop: 4, fontSize: 12, fontWeight: '500' },
  scroll: { flex: 1 },
  scrollContent: { padding: 20, gap: 12 },
  card: {
    borderRadius: 14,
    borderWidth: StyleSheet.hairlineWidth,
    padding: 16,
    minHeight: 120,
  },
  cardLabel: {
    fontSize: 11,
    fontWeight: '700',
    letterSpacing: 0.6,
    textTransform: 'uppercase',
    marginBottom: 8,
  },
  cardBody: { fontSize: 16, lineHeight: 23 },
  latency: { marginTop: 8, fontSize: 11, fontFamily: Platform.OS === 'ios' ? 'Menlo' : 'monospace' },
  diagBox: {
    marginTop: 4,
    padding: 12,
    borderRadius: 12,
    borderWidth: StyleSheet.hairlineWidth,
  },
  diagLabel: {
    fontSize: 11,
    fontWeight: '700',
    letterSpacing: 0.6,
    marginBottom: 4,
  },
  diag: { fontSize: 13, fontFamily: Platform.OS === 'ios' ? 'Menlo' : 'monospace' },
  errorBox: {
    padding: 14,
    borderRadius: 12,
    backgroundColor: '#3a0e0e',
    borderWidth: 1,
    borderColor: '#ff6b6b',
  },
  errorLabel: {
    fontSize: 11,
    fontWeight: '700',
    letterSpacing: 0.6,
    marginBottom: 6,
    color: '#ff6b6b',
  },
  errorText: {
    fontSize: 13,
    color: '#ffd6d6',
    fontFamily: Platform.OS === 'ios' ? 'Menlo' : 'monospace',
    lineHeight: 18,
  },
  footer: { alignItems: 'center', paddingBottom: 24, paddingTop: 8, gap: 12 },
  statusText: {
    fontSize: 13,
    fontWeight: '500',
    textAlign: 'center',
    paddingHorizontal: 24,
  },
  mic: {
    width: 84,
    height: 84,
    borderRadius: 42,
    alignItems: 'center',
    justifyContent: 'center',
    shadowColor: '#000',
    shadowOpacity: 0.18,
    shadowRadius: 12,
    shadowOffset: { width: 0, height: 6 },
    elevation: 6,
  },
  micIcon: { fontSize: 34, color: '#fff' },
  hint: { fontSize: 12 },
});

export default App;
