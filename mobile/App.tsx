import React, { useEffect, useRef, useState } from 'react';
import {
  ActivityIndicator,
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

// Single-model pipeline: Gemma 4 handles both audio understanding and the
// response in one forward pass — per the Cactus blog, Gemma 4 reasons directly
// over the audio modality (no separate ASR step).
// https://docs.cactuscompute.com/latest/blog/gemma4/
//
// We load it via the STT hook because cactus-react-native v1.13.0 doesn't yet
// expose audio input on the LM hook (CactusLMMessage has no audio field), but
// the underlying engine's transcribe() accepts any model and routes audio to
// the model's native audio path.
const MODEL_SLUG = 'gemma-4-e2b-it';
const QUANT: 'int4' | 'int8' = 'int4';
const INTERNAL_NAME = `${MODEL_SLUG}-${QUANT}`;
const WEIGHTS_URL = `https://huggingface.co/Cactus-Compute/gemma-4-E2B-it/resolve/v1.13/weights/${INTERNAL_NAME}.zip`;

// Gemma 4 uses chat templating natively; we pass a system-style prompt that
// tells it to listen and reply. The whisper-style prompt tokens don't apply.
const VOICE_PROMPT =
  'You are a friendly on-device voice coach. Listen to the user and reply concisely (1-3 sentences). Be encouraging and direct.';

const RECORD_OPTIONS = {
  sampleRate: 16000,
  channels: 1,
  bitsPerSample: 16,
  audioSource: 6,
  wavFile: 'voice-coach-clip.wav',
};

// Cactus's CactusFileSystem isn't re-exported from the package root, so grab
// the underlying Nitro hybrid object directly. Used to manually download
// Gemma 4 (which the auto-registry filters out — only ships int4, no int8).
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

  // Manage Gemma 4 download ourselves (registry skips it for lacking int8)
  const [downloading, setDownloading] = useState(false);
  const [downloadProgress, setDownloadProgress] = useState(0);
  const [modelReady, setModelReady] = useState(false);

  const [isRecording, setIsRecording] = useState(false);
  const [isThinking, setIsThinking] = useState(false);
  const [reply, setReply] = useState('');
  const [localError, setLocalError] = useState<string | null>(null);
  const [diag, setDiag] = useState<string>('');
  const recorderReady = useRef(false);
  const recordStartedAt = useRef<number>(0);
  const pcmBytes = useRef<number[]>([]);

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

  const handleStartRecording = async () => {
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
      setReply('');
      pcmBytes.current = [];
      AudioRecord.start();
      recordStartedAt.current = Date.now();
      setIsRecording(true);
    } catch (err: any) {
      setLocalError(err?.message ?? 'Failed to start recording.');
    }
  };

  const handleStopAndAsk = async () => {
    try {
      await AudioRecord.stop();
      const durationMs = Date.now() - recordStartedAt.current;
      setIsRecording(false);

      const samples = pcmBytes.current;
      const byteCount = samples.length;
      const sampleCount = byteCount / 2;
      const energy = computeEnergy(samples);
      setDiag(`bytes=${byteCount} samples=${sampleCount} dur=${durationMs}ms rms=${energy.toFixed(0)}`);

      if (durationMs < 800) {
        setLocalError('Clip too short — speak for at least a second.');
        return;
      }
      if (byteCount < 4000) {
        setLocalError('No audio captured. Mic may be blocked.');
        return;
      }
      if (energy < 100) {
        setLocalError("Audio is silent. Check mic permission and that the phone isn't muted.");
        return;
      }

      setIsThinking(true);
      const result = await stt.transcribe({
        audio: samples,
        prompt: VOICE_PROMPT,
        options: { useVad: true, maxTokens: 256, temperature: 0.7 },
      });
      setReply((result.response ?? '').trim() || '(empty response)');
    } catch (err: any) {
      setLocalError(err?.message ?? 'Pipeline failed.');
    } finally {
      setIsThinking(false);
    }
  };

  const onMicPress = () => {
    if (isThinking) return;
    if (isRecording) {
      handleStopAndAsk();
    } else {
      handleStartRecording();
    }
  };

  const palette = isDarkMode ? darkPalette : lightPalette;
  const status = describeStatus({
    downloading,
    downloadProgress,
    modelReady,
    isRecording,
    isThinking,
    error: stt.error ?? localError,
  });

  const micDisabled = !modelReady || downloading || isThinking;

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
          {MODEL_SLUG} ({QUANT}) · on-device via Cactus
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
          <Text style={[styles.cardLabel, { color: palette.muted }]}>Gemma 4</Text>
          <Text style={[styles.cardBody, { color: palette.text }]}>
            {reply ||
              (isThinking
                ? 'Thinking…'
                : isRecording
                ? 'Listening…'
                : 'Tap mic and speak.')}
          </Text>
        </View>
        {diag ? (
          <View style={[styles.diagBox, { borderColor: palette.border, backgroundColor: palette.card }]}>
            <Text style={[styles.diagLabel, { color: palette.muted }]}>MIC DIAGNOSTIC</Text>
            <Text style={[styles.diag, { color: palette.text }]}>{diag}</Text>
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
          accessibilityLabel={isRecording ? 'Stop recording' : 'Start recording'}
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
          {isThinking ? (
            <ActivityIndicator color="#fff" />
          ) : (
            <Text style={styles.micIcon}>{isRecording ? '■' : '🎤'}</Text>
          )}
        </TouchableOpacity>

        <Text style={[styles.hint, { color: palette.muted }]}>
          {isRecording ? 'Tap to stop & ask Gemma' : 'Tap to record'}
        </Text>
      </View>
    </View>
  );
}

function computeEnergy(bytes: number[]): number {
  if (bytes.length < 2) return 0;
  let sumSq = 0;
  let count = 0;
  for (let i = 0; i + 1 < bytes.length; i += 2) {
    const lo = bytes[i];
    const hi = bytes[i + 1];
    let sample = (hi << 8) | lo;
    if (sample & 0x8000) sample = sample - 0x10000;
    sumSq += sample * sample;
    count++;
  }
  return count === 0 ? 0 : Math.sqrt(sumSq / count);
}

type StatusInput = {
  downloading: boolean;
  downloadProgress: number;
  modelReady: boolean;
  isRecording: boolean;
  isThinking: boolean;
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
  if (s.isThinking) return { label: 'Gemma 4 is thinking…', tone: 'info' };
  if (s.isRecording) return { label: 'Recording…', tone: 'info' };
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
