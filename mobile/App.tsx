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
import AudioRecord from 'react-native-audio-record';

// Cactus React Native ships speech-to-text via Whisper / Moonshine.
// Gemma 4 (multimodal) is the LM target for this app — wire it in next via
// `useCactusLM({ model: 'gemma-4-e2b' })` once we route transcripts into chat.
const STT_MODEL = 'whisper-small';

const RECORD_OPTIONS = {
  sampleRate: 16000,
  channels: 1,
  bitsPerSample: 16,
  audioSource: 6, // VOICE_RECOGNITION on Android
  wavFile: 'voice-coach-clip.wav',
};

async function ensureMicPermission(): Promise<boolean> {
  if (Platform.OS !== 'android') return true;
  const granted = await PermissionsAndroid.request(
    PermissionsAndroid.PERMISSIONS.RECORD_AUDIO,
    {
      title: 'Microphone permission',
      message: 'AI Voice Coach needs your microphone to transcribe speech.',
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
  const stt = useCactusSTT({ model: STT_MODEL });

  const [isRecording, setIsRecording] = useState(false);
  const [isTranscribing, setIsTranscribing] = useState(false);
  const [transcript, setTranscript] = useState('');
  const [localError, setLocalError] = useState<string | null>(null);
  const recorderReady = useRef(false);

  useEffect(() => {
    if (!recorderReady.current) {
      AudioRecord.init(RECORD_OPTIONS);
      recorderReady.current = true;
    }
  }, []);

  useEffect(() => {
    if (!stt.isDownloaded && !stt.isDownloading) {
      stt.download().catch(() => {
        // surfaced via stt.error
      });
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [stt.isDownloaded]);

  const handleStartRecording = async () => {
    setLocalError(null);
    if (!stt.isDownloaded) {
      Alert.alert('Model not ready', 'Wait for the model to finish downloading.');
      return;
    }
    const ok = await ensureMicPermission();
    if (!ok) {
      setLocalError('Microphone permission denied.');
      return;
    }
    try {
      setTranscript('');
      AudioRecord.start();
      setIsRecording(true);
    } catch (err: any) {
      setLocalError(err?.message ?? 'Failed to start recording.');
    }
  };

  const handleStopAndTranscribe = async () => {
    try {
      const audioPath = await AudioRecord.stop();
      setIsRecording(false);
      setIsTranscribing(true);
      const filePath = audioPath.startsWith('file://')
        ? audioPath.replace('file://', '')
        : audioPath;
      const result = await stt.transcribe({
        audio: filePath,
        options: { useVad: true },
      });
      setTranscript(result.response?.trim() ?? '');
    } catch (err: any) {
      setLocalError(err?.message ?? 'Transcription failed.');
    } finally {
      setIsTranscribing(false);
    }
  };

  const onMicPress = () => {
    if (isTranscribing) return;
    if (isRecording) {
      handleStopAndTranscribe();
    } else {
      handleStartRecording();
    }
  };

  const palette = isDarkMode ? darkPalette : lightPalette;
  const status = describeStatus({
    isDownloading: stt.isDownloading,
    downloadProgress: stt.downloadProgress,
    isInitializing: stt.isInitializing,
    isRecording,
    isTranscribing,
    error: stt.error ?? localError,
    isDownloaded: stt.isDownloaded,
  });

  const micDisabled =
    stt.isDownloading || stt.isInitializing || isTranscribing || !stt.isDownloaded;

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
          On-device transcription via Cactus · {STT_MODEL}
        </Text>
      </View>

      <ScrollView
        style={styles.transcriptScroll}
        contentContainerStyle={styles.transcriptContent}
      >
        <View
          style={[
            styles.transcriptCard,
            { backgroundColor: palette.card, borderColor: palette.border },
          ]}
        >
          <Text style={[styles.transcriptLabel, { color: palette.muted }]}>
            Transcript
          </Text>
          <Text style={[styles.transcriptText, { color: palette.text }]}>
            {transcript || (isRecording ? 'Listening…' : 'Tap the mic and speak.')}
          </Text>
        </View>
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
          {isTranscribing ? (
            <ActivityIndicator color="#fff" />
          ) : (
            <Text style={styles.micIcon}>{isRecording ? '■' : '🎤'}</Text>
          )}
        </TouchableOpacity>

        <Text style={[styles.hint, { color: palette.muted }]}>
          {isRecording ? 'Tap to stop & transcribe' : 'Tap to record'}
        </Text>
      </View>
    </View>
  );
}

type StatusInput = {
  isDownloading: boolean;
  downloadProgress: number;
  isInitializing: boolean;
  isRecording: boolean;
  isTranscribing: boolean;
  error: string | null;
  isDownloaded: boolean;
};

function describeStatus(s: StatusInput): { label: string; tone: 'info' | 'error' } {
  if (s.error) return { label: s.error, tone: 'error' };
  if (s.isDownloading) {
    return {
      label: `Downloading model… ${Math.round(s.downloadProgress * 100)}%`,
      tone: 'info',
    };
  }
  if (s.isInitializing) return { label: 'Initializing model…', tone: 'info' };
  if (s.isTranscribing) return { label: 'Transcribing…', tone: 'info' };
  if (s.isRecording) return { label: 'Recording…', tone: 'info' };
  if (!s.isDownloaded) return { label: 'Preparing…', tone: 'info' };
  return { label: 'Ready', tone: 'info' };
}

const lightPalette = {
  bg: '#f7f8fa',
  text: '#111418',
  muted: '#5a6373',
  card: '#ffffff',
  border: '#e3e6ec',
  accent: '#111418',
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
  container: {
    flex: 1,
  },
  header: {
    paddingHorizontal: 24,
    paddingTop: 16,
    paddingBottom: 8,
  },
  title: {
    fontSize: 28,
    fontWeight: '700',
    letterSpacing: -0.5,
  },
  subtitle: {
    marginTop: 4,
    fontSize: 13,
    fontWeight: '500',
  },
  transcriptScroll: {
    flex: 1,
  },
  transcriptContent: {
    padding: 24,
  },
  transcriptCard: {
    borderRadius: 16,
    borderWidth: StyleSheet.hairlineWidth,
    padding: 18,
    minHeight: 200,
  },
  transcriptLabel: {
    fontSize: 12,
    fontWeight: '600',
    letterSpacing: 0.6,
    textTransform: 'uppercase',
    marginBottom: 10,
  },
  transcriptText: {
    fontSize: 17,
    lineHeight: 25,
  },
  footer: {
    alignItems: 'center',
    paddingBottom: 24,
    paddingTop: 8,
    gap: 12,
  },
  statusText: {
    fontSize: 13,
    fontWeight: '500',
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
  micIcon: {
    fontSize: 34,
    color: '#fff',
  },
  hint: {
    fontSize: 12,
  },
});

export default App;
