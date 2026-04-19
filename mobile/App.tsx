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

// Single-model pipeline: Gemma 4 handles audio understanding directly — per
// the Cactus blog, the audio conformer feeds the same residual stream as text,
// so streamTranscribe routes PCM chunks straight into Gemma's audio path.
// https://docs.cactuscompute.com/latest/blog/gemma4/
//
// We use the STT hook because cactus-react-native v1.13.0 doesn't yet expose
// audio input on CactusLM. The streaming API
// (streamTranscribeStart/Process/Stop) is model-agnostic and works with any
// model the engine has loaded.
const MODEL_SLUG = 'gemma-4-e2b-it';
const QUANT: 'int4' | 'int8' = 'int4';
const INTERNAL_NAME = `${MODEL_SLUG}-${QUANT}`;
const WEIGHTS_URL = `https://huggingface.co/Cactus-Compute/gemma-4-E2B-it/resolve/v1.13/weights/${INTERNAL_NAME}.zip`;

const RECORD_OPTIONS = {
  sampleRate: 16000,
  channels: 1,
  bitsPerSample: 16,
  audioSource: 6,
  wavFile: 'voice-coach-clip.wav',
};

// Stream chunking: 16kHz mono 16-bit = 32000 bytes/s. We flush every ~2s of
// audio, which matches Cactus's default minChunkSize (32000 samples).
const CHUNK_BYTES = 64000;

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
  const [localError, setLocalError] = useState<string | null>(null);
  const [diag, setDiag] = useState<string>('');

  const recorderReady = useRef(false);
  const recordStartedAt = useRef<number>(0);
  const pcmBytes = useRef<number[]>([]);
  const totalBytes = useRef<number>(0);
  const flushingRef = useRef(false);
  const streamingRef = useRef(false);
  const sttRef = useRef(stt);

  useEffect(() => {
    sttRef.current = stt;
  }, [stt]);

  // Pull a chunk off the buffer and feed it to the streaming engine. Drops
  // calls if a previous flush is still mid-inference (back-pressure).
  const tryFlush = async (force: boolean) => {
    if (!streamingRef.current) return;
    if (flushingRef.current) return;
    const have = pcmBytes.current.length;
    if (!force && have < CHUNK_BYTES) return;
    if (have === 0) return;
    flushingRef.current = true;
    const chunk = pcmBytes.current.splice(0, have);
    try {
      await sttRef.current.streamTranscribeProcess({ audio: chunk });
    } catch (err: any) {
      setLocalError(err?.message ?? 'streamTranscribeProcess failed.');
    } finally {
      flushingRef.current = false;
    }
  };

  useEffect(() => {
    if (!recorderReady.current) {
      AudioRecord.init(RECORD_OPTIONS);
      AudioRecord.on('data', (data: string) => {
        const chunk = Buffer.from(data, 'base64');
        for (let i = 0; i < chunk.length; i++) {
          pcmBytes.current.push(chunk[i]);
        }
        totalBytes.current += chunk.length;
        tryFlush(false);
      });
      recorderReady.current = true;
    }
  }, []);

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

  const handleStart = async () => {
    setLocalError(null);
    setDiag('');
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
      totalBytes.current = 0;
      flushingRef.current = false;

      await stt.streamTranscribeStart({
        confirmationThreshold: 0.85,
        minChunkSize: 32000,
        language: 'en',
      });
      streamingRef.current = true;

      AudioRecord.start();
      recordStartedAt.current = Date.now();
      setIsRecording(true);
    } catch (err: any) {
      streamingRef.current = false;
      setLocalError(err?.message ?? 'Failed to start streaming.');
    }
  };

  const handleStop = async () => {
    try {
      await AudioRecord.stop();
      const durationMs = Date.now() - recordStartedAt.current;
      setIsRecording(false);

      // Wait for any in-flight chunk, then flush whatever's left.
      while (flushingRef.current) {
        await new Promise((r) => setTimeout(r, 50));
      }
      await tryFlush(true);
      while (flushingRef.current) {
        await new Promise((r) => setTimeout(r, 50));
      }

      streamingRef.current = false;
      await stt.streamTranscribeStop();

      setDiag(
        `bytes=${totalBytes.current} samples=${totalBytes.current / 2} dur=${durationMs}ms`,
      );
    } catch (err: any) {
      streamingRef.current = false;
      setLocalError(err?.message ?? 'Failed to stop streaming.');
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
    isStreamTranscribing: stt.isStreamTranscribing,
    error: stt.error ?? localError,
  });

  const micDisabled = !modelReady || downloading;

  const confirmed = stt.streamTranscribeConfirmed;
  const pending = stt.streamTranscribePending;
  const hasText = (confirmed + pending).trim().length > 0;

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
          {MODEL_SLUG} ({QUANT}) · streaming on-device
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
            <Text style={[styles.cardBody, { color: palette.text }]}>
              <Text>{confirmed}</Text>
              <Text style={{ color: palette.muted }}>{pending}</Text>
            </Text>
          ) : (
            <Text style={[styles.cardBody, { color: palette.muted }]}>
              {isRecording ? 'Listening… speak now' : 'Tap mic and speak.'}
            </Text>
          )}
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
  isStreamTranscribing: boolean;
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
  if (s.isStreamTranscribing) return { label: 'Finalizing…', tone: 'info' };
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
