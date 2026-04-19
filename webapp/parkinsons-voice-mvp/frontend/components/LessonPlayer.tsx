'use client';

import Link from 'next/link';
import { useEffect, useMemo, useRef, useState } from 'react';
import { useRouter } from 'next/navigation';
import {
  flattenedLessons,
  getOrderedPhaseKeys,
  type FlattenedLesson,
  type LessonPhase,
  type LessonPhaseKey,
} from '../data/lessonProgram';

type AnalysisResponse = {
  referencePhrase: string;
  overallScore: number;
  metrics: Record<string, number>;
  scores: Record<string, number>;
  feedback: string[];
  disclaimer: string;
};

type LessonPlayerProps = {
  lesson: FlattenedLesson;
  lessonIndex: number;
  phaseKey: LessonPhaseKey;
};

const apiBase = process.env.NEXT_PUBLIC_API_BASE_URL ?? 'http://localhost:8000';
const preferredMimeTypes = ['audio/webm;codecs=opus', 'audio/webm', 'audio/mp4', 'audio/ogg;codecs=opus'];
const silenceThreshold = 0.035;
const silenceDurationMs = 1400;

function formatLabel(value: string) {
  return value
    .replace(/_/g, ' ')
    .replace(/([A-Z])/g, ' $1')
    .replace(/^./, (char) => char.toUpperCase());
}

function getSupportedRecordingMimeType() {
  if (typeof MediaRecorder === 'undefined') return undefined;
  return preferredMimeTypes.find((mimeType) => MediaRecorder.isTypeSupported(mimeType));
}

function audioBufferToWavBlob(audioBuffer: AudioBuffer) {
  const channelCount = audioBuffer.numberOfChannels;
  const sampleRate = audioBuffer.sampleRate;
  const frameCount = audioBuffer.length;
  const pcmData = new Int16Array(frameCount);

  for (let frameIndex = 0; frameIndex < frameCount; frameIndex += 1) {
    let mixedSample = 0;

    for (let channelIndex = 0; channelIndex < channelCount; channelIndex += 1) {
      mixedSample += audioBuffer.getChannelData(channelIndex)[frameIndex];
    }

    const monoSample = Math.max(-1, Math.min(1, mixedSample / channelCount));
    pcmData[frameIndex] = monoSample < 0 ? monoSample * 0x8000 : monoSample * 0x7fff;
  }

  const buffer = new ArrayBuffer(44 + pcmData.byteLength);
  const view = new DataView(buffer);

  const writeString = (offset: number, value: string) => {
    for (let index = 0; index < value.length; index += 1) {
      view.setUint8(offset + index, value.charCodeAt(index));
    }
  };

  writeString(0, 'RIFF');
  view.setUint32(4, 36 + pcmData.byteLength, true);
  writeString(8, 'WAVE');
  writeString(12, 'fmt ');
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeString(36, 'data');
  view.setUint32(40, pcmData.byteLength, true);

  pcmData.forEach((sample, index) => {
    view.setInt16(44 + index * 2, sample, true);
  });

  return new Blob([buffer], { type: 'audio/wav' });
}

async function convertBlobToWav(blob: Blob) {
  const audioContext = new AudioContext();

  try {
    const arrayBuffer = await blob.arrayBuffer();
    const audioBuffer = await audioContext.decodeAudioData(arrayBuffer.slice(0));
    return audioBufferToWavBlob(audioBuffer);
  } finally {
    await audioContext.close();
  }
}

function getLessonPhaseHref(lessonId: string, phase: LessonPhaseKey) {
  return `/lessons/${lessonId}/${phase}`;
}

function getPromptLines(phase: LessonPhase) {
  if (Array.isArray(phase.content)) {
    return phase.content;
  }

  if (typeof phase.content === 'string') {
    return [phase.content];
  }

  return [phase.instructions];
}

function getPrimaryButtonLabel(isRecording: boolean, loading: boolean, hasResult: boolean) {
  if (loading) return 'Analyzing';
  if (isRecording) return 'Stop';
  if (hasResult) return 'Speak Again';
  return 'Speak';
}

function getStatusLabel(isRecording: boolean, loading: boolean, hasResult: boolean, phaseName: string) {
  if (loading) return 'Analyzing your recording…';
  if (isRecording) return 'Listening… I will analyze when you finish speaking.';
  if (hasResult) return 'Done. You can repeat or go next.';
  return `Tap the button to start ${phaseName.toLowerCase()}.`;
}

export default function LessonPlayer({ lesson, lessonIndex, phaseKey }: LessonPlayerProps) {
  const router = useRouter();
  const recorderRef = useRef<MediaRecorder | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const audioUrlRef = useRef<string | null>(null);
  const monitorFrameRef = useRef<number | null>(null);
  const monitorAudioContextRef = useRef<AudioContext | null>(null);
  const speechDetectedRef = useRef(false);
  const silenceStartedAtRef = useRef<number | null>(null);

  const phaseKeys = getOrderedPhaseKeys(lesson);
  const phaseIndex = phaseKeys.indexOf(phaseKey);
  const phase = lesson.exercise.phases[phaseKey];
  const previousLesson = lessonIndex > 0 ? flattenedLessons[lessonIndex - 1] : null;
  const nextLesson = lessonIndex < flattenedLessons.length - 1 ? flattenedLessons[lessonIndex + 1] : null;
  const previousPhaseKey = phaseIndex > 0 ? phaseKeys[phaseIndex - 1] : null;
  const nextPhaseKey = phaseIndex < phaseKeys.length - 1 ? phaseKeys[phaseIndex + 1] : null;
  const previousLessonPhaseKeys = previousLesson ? getOrderedPhaseKeys(previousLesson) : [];
  const nextLessonPhaseKeys = nextLesson ? getOrderedPhaseKeys(nextLesson) : [];
  const previousHref = previousPhaseKey
    ? getLessonPhaseHref(lesson.exercise.id, previousPhaseKey)
    : previousLesson && previousLessonPhaseKeys.length > 0
      ? getLessonPhaseHref(previousLesson.exercise.id, previousLessonPhaseKeys[previousLessonPhaseKeys.length - 1])
      : null;
  const nextHref = nextPhaseKey
    ? getLessonPhaseHref(lesson.exercise.id, nextPhaseKey)
    : nextLesson && nextLessonPhaseKeys.length > 0
      ? getLessonPhaseHref(nextLesson.exercise.id, nextLessonPhaseKeys[0])
      : getLessonPhaseHref(flattenedLessons[0].exercise.id, getOrderedPhaseKeys(flattenedLessons[0])[0]);

  const [isRecording, setIsRecording] = useState(false);
  const [audioUrl, setAudioUrl] = useState<string | null>(null);
  const [result, setResult] = useState<AnalysisResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const promptLines = useMemo(() => (phase ? getPromptLines(phase) : []), [phase]);
  const sortedScores = useMemo(() => (result ? Object.entries(result.scores) : []), [result]);

  useEffect(() => {
    return () => {
      if (monitorFrameRef.current) {
        cancelAnimationFrame(monitorFrameRef.current);
      }

      if (recorderRef.current && recorderRef.current.state !== 'inactive') {
        recorderRef.current.onstop = null;
        recorderRef.current.stop();
      }

      streamRef.current?.getTracks().forEach((track) => track.stop());

      if (monitorAudioContextRef.current && monitorAudioContextRef.current.state !== 'closed') {
        void monitorAudioContextRef.current.close();
      }

      if (audioUrlRef.current) {
        URL.revokeObjectURL(audioUrlRef.current);
      }
    };
  }, []);

  if (!phase) {
    return null;
  }

  function clearAudioUrl() {
    if (audioUrlRef.current) {
      URL.revokeObjectURL(audioUrlRef.current);
      audioUrlRef.current = null;
    }

    setAudioUrl(null);
  }

  function stopAudioMonitoring() {
    if (monitorFrameRef.current) {
      cancelAnimationFrame(monitorFrameRef.current);
      monitorFrameRef.current = null;
    }

    speechDetectedRef.current = false;
    silenceStartedAtRef.current = null;

    if (monitorAudioContextRef.current) {
      const audioContext = monitorAudioContextRef.current;
      monitorAudioContextRef.current = null;

      if (audioContext.state !== 'closed') {
        void audioContext.close();
      }
    }
  }

  function cleanupStream() {
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
  }

  function resetSession() {
    setError(null);
    setResult(null);
    clearAudioUrl();
  }

  async function analyzeBlob(blob: Blob) {
    try {
      setLoading(true);
      setError(null);

      const wavBlob = await convertBlobToWav(blob);
      const formData = new FormData();
      formData.append('file', wavBlob, 'sample.wav');

      const response = await fetch(`${apiBase}/analyze`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const body = await response.json().catch(() => ({}));
        throw new Error(body.detail || 'Analysis failed.');
      }

      const data = (await response.json()) as AnalysisResponse;
      setResult(data);
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Unknown error during analysis.';
      setError(message);
    } finally {
      setLoading(false);
    }
  }

  async function startRecording() {
    resetSession();

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;
      const mimeType = getSupportedRecordingMimeType();
      const recorder = mimeType ? new MediaRecorder(stream, { mimeType }) : new MediaRecorder(stream);
      chunksRef.current = [];
      speechDetectedRef.current = false;
      silenceStartedAtRef.current = null;

      const audioContext = new AudioContext();
      monitorAudioContextRef.current = audioContext;
      const source = audioContext.createMediaStreamSource(stream);
      const analyser = audioContext.createAnalyser();
      analyser.fftSize = 2048;
      analyser.smoothingTimeConstant = 0.85;
      source.connect(analyser);
      const timeDomainData = new Uint8Array(analyser.fftSize);

      const monitorSilence = () => {
        analyser.getByteTimeDomainData(timeDomainData);

        let squareSum = 0;
        for (let index = 0; index < timeDomainData.length; index += 1) {
          const normalizedValue = (timeDomainData[index] - 128) / 128;
          squareSum += normalizedValue * normalizedValue;
        }

        const rms = Math.sqrt(squareSum / timeDomainData.length);
        const isSpeaking = rms > silenceThreshold;

        if (isSpeaking) {
          speechDetectedRef.current = true;
          silenceStartedAtRef.current = null;
        } else if (speechDetectedRef.current) {
          const now = performance.now();

          if (silenceStartedAtRef.current === null) {
            silenceStartedAtRef.current = now;
          } else if (now - silenceStartedAtRef.current >= silenceDurationMs) {
            stopRecording();
            return;
          }
        }

        monitorFrameRef.current = requestAnimationFrame(monitorSilence);
      };

      recorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          chunksRef.current.push(event.data);
        }
      };

      recorder.onstop = async () => {
        stopAudioMonitoring();

        const blob = new Blob(chunksRef.current, { type: recorder.mimeType || mimeType || 'audio/webm' });
        const nextAudioUrl = URL.createObjectURL(blob);

        clearAudioUrl();
        audioUrlRef.current = nextAudioUrl;
        setAudioUrl(nextAudioUrl);
        cleanupStream();

        await analyzeBlob(blob);
      };

      recorderRef.current = recorder;
      recorder.start();
      monitorFrameRef.current = requestAnimationFrame(monitorSilence);
      setIsRecording(true);
    } catch (err) {
      stopAudioMonitoring();
      cleanupStream();
      setError('Microphone access failed. Please allow mic permissions and try again.');
      console.error(err);
    }
  }

  function stopRecording() {
    stopAudioMonitoring();

    if (recorderRef.current && recorderRef.current.state !== 'inactive') {
      recorderRef.current.stop();
    }

    setIsRecording(false);
  }

  function handlePrimaryAction() {
    if (loading) {
      return;
    }

    if (isRecording) {
      stopRecording();
      return;
    }

    void startRecording();
  }

  return (
    <>
      <main
        style={{
          minHeight: '100vh',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          padding: 24,
        }}
      >
        <div style={{ width: '100%', maxWidth: 720, display: 'grid', gap: 24, textAlign: 'center', justifyItems: 'center' }}>
          <div style={{ display: 'grid', gap: 10, justifyItems: 'center' }}>
            <div style={{ color: 'var(--muted)', fontSize: 16, letterSpacing: 0.4 }}>
              {formatLabel(phaseKey)}
            </div>

            <div
              style={{
                display: 'grid',
                gap: 10,
                maxWidth: 620,
                padding: '8px 8px 0',
              }}
            >
              {promptLines.map((line) => (
                <div key={line} style={{ fontSize: 34, lineHeight: 1.35, fontWeight: 600 }}>
                  {line}
                </div>
              ))}
            </div>
          </div>

          <div className={`voice-orb-shell ${isRecording ? 'recording' : ''} ${loading ? 'analyzing' : ''}`}>
            <button
              onClick={handlePrimaryAction}
              disabled={loading}
              className={`voice-orb ${isRecording ? 'recording' : ''} ${loading ? 'analyzing' : ''} ${result ? 'complete' : ''}`}
              aria-label={getPrimaryButtonLabel(isRecording, loading, Boolean(result))}
            >
              {getPrimaryButtonLabel(isRecording, loading, Boolean(result))}
            </button>
          </div>

          <div style={{ minHeight: 28, color: error ? '#ffb2b2' : 'var(--muted)', fontSize: 20, lineHeight: 1.5 }}>
            {error || getStatusLabel(isRecording, loading, Boolean(result), formatLabel(phaseKey))}
          </div>

          <div style={{ display: 'flex', justifyContent: 'center', gap: 14, flexWrap: 'wrap' }}>
            <button
              onClick={() => {
                if (previousHref) {
                  router.push(previousHref);
                }
              }}
              type="button"
              disabled={!previousHref || isRecording || loading}
              style={{
                minWidth: 180,
                padding: '14px 22px',
                borderRadius: 999,
                border: '1px solid var(--card-border)',
                background: !previousHref || isRecording || loading ? '#202846' : '#121a31',
                color: 'var(--text)',
                fontSize: 20,
                fontWeight: 600,
                opacity: !previousHref || isRecording || loading ? 0.6 : 1,
              }}
            >
              Previous
            </button>

            <button
              onClick={() => router.push(nextHref)}
              type="button"
              disabled={isRecording || loading}
              style={{
                minWidth: 180,
                padding: '14px 22px',
                borderRadius: 999,
                border: '1px solid var(--card-border)',
                background: isRecording || loading ? '#202846' : '#121a31',
                color: 'var(--text)',
                fontSize: 20,
                fontWeight: 600,
                opacity: isRecording || loading ? 0.6 : 1,
              }}
            >
              Next
            </button>
          </div>

          <details
            style={{
              width: '100%',
              maxWidth: 620,
              padding: 18,
              borderRadius: 22,
              background: 'rgba(20, 27, 52, 0.72)',
              border: '1px solid var(--card-border)',
              textAlign: 'left',
            }}
          >
            <summary style={{ cursor: 'pointer', color: 'var(--muted)', fontSize: 18 }}>More details</summary>

            <div style={{ display: 'grid', gap: 18, marginTop: 18 }}>
              <div style={{ color: 'var(--text)', fontSize: 18, lineHeight: 1.7 }}>
                <div>
                  <strong>Exercise:</strong> {lesson.exercise.name}
                </div>
                <div>
                  <strong>Step:</strong> {formatLabel(phaseKey)}
                </div>
                <div>
                  <strong>Instruction:</strong> {phase.instructions}
                </div>
                {phase.focus ? (
                  <div>
                    <strong>Focus:</strong> {phase.focus}
                  </div>
                ) : null}
                {typeof phase.duration_sec === 'number' ? (
                  <div>
                    <strong>Duration:</strong> {phase.duration_sec} seconds
                  </div>
                ) : null}
                {typeof phase.repetitions === 'number' ? (
                  <div>
                    <strong>Repetitions:</strong> {phase.repetitions}
                  </div>
                ) : null}
              </div>

              {audioUrl ? <audio controls src={audioUrl} style={{ width: '100%' }} /> : null}

              {result ? (
                <div style={{ display: 'grid', gap: 12 }}>
                  <div style={{ fontSize: 20 }}>
                    <strong>Score:</strong> {result.overallScore}/100
                  </div>

                  {result.feedback.length > 0 ? (
                    <ul style={{ margin: 0, paddingLeft: 22, lineHeight: 1.8, color: 'var(--text)' }}>
                      {result.feedback.map((item) => (
                        <li key={item}>{item}</li>
                      ))}
                    </ul>
                  ) : null}

                  {sortedScores.length > 0 ? (
                    <div style={{ display: 'grid', gap: 8 }}>
                      {sortedScores.map(([key, value]) => (
                        <div key={key} style={{ color: 'var(--muted)' }}>
                          {formatLabel(key)}: <span style={{ color: 'var(--text)' }}>{value}</span>
                        </div>
                      ))}
                    </div>
                  ) : null}

                  <div style={{ color: 'var(--muted)', fontSize: 15, lineHeight: 1.6 }}>{result.disclaimer}</div>
                </div>
              ) : null}

              <div style={{ display: 'flex', flexWrap: 'wrap', gap: 12 }}>
                <button
                  onClick={resetSession}
                  type="button"
                  style={{
                    padding: '12px 16px',
                    borderRadius: 14,
                    border: '1px solid var(--card-border)',
                    background: '#121a31',
                    color: 'var(--text)',
                    fontSize: 16,
                  }}
                >
                  Clear recording
                </button>

                {previousHref ? (
                  <Link
                    href={previousHref}
                    style={{
                      padding: '12px 16px',
                      borderRadius: 14,
                      border: '1px solid var(--card-border)',
                      background: '#121a31',
                      color: 'var(--text)',
                      fontSize: 16,
                      textDecoration: 'none',
                    }}
                  >
                    Previous step
                  </Link>
                ) : null}
              </div>
            </div>
          </details>
        </div>
      </main>

      <style jsx>{`
        .voice-orb-shell {
          display: flex;
          align-items: center;
          justify-content: center;
          width: 320px;
          height: 320px;
          border-radius: 999px;
          background: radial-gradient(circle, rgba(138, 180, 255, 0.24) 0%, rgba(138, 180, 255, 0.04) 56%, transparent 72%);
        }

        .voice-orb-shell.recording,
        .voice-orb-shell.analyzing {
          animation: pulse 1.8s ease-in-out infinite;
        }

        .voice-orb {
          width: 220px;
          height: 220px;
          border: none;
          border-radius: 999px;
          color: #09111f;
          font-size: 34px;
          font-weight: 700;
          background: linear-gradient(180deg, #a7c6ff 0%, #8ab4ff 55%, #72d0ff 100%);
          box-shadow: 0 20px 60px rgba(138, 180, 255, 0.35);
          transition: transform 160ms ease, box-shadow 160ms ease, opacity 160ms ease;
        }

        .voice-orb:hover {
          transform: scale(1.02);
        }

        .voice-orb:active {
          transform: scale(0.98);
        }

        .voice-orb.recording {
          background: linear-gradient(180deg, #ffb0b0 0%, #ff8a8a 55%, #ff6d92 100%);
          box-shadow: 0 20px 60px rgba(255, 120, 140, 0.35);
        }

        .voice-orb.analyzing {
          background: linear-gradient(180deg, #b4ffd5 0%, #9cf6b6 55%, #8fe7ff 100%);
          box-shadow: 0 20px 60px rgba(156, 246, 182, 0.35);
        }

        .voice-orb.complete {
          background: linear-gradient(180deg, #cde2ff 0%, #9cc8ff 55%, #8ab4ff 100%);
        }

        @keyframes pulse {
          0% {
            transform: scale(0.98);
            opacity: 0.9;
          }

          50% {
            transform: scale(1.02);
            opacity: 1;
          }

          100% {
            transform: scale(0.98);
            opacity: 0.9;
          }
        }

        @media (max-width: 640px) {
          .voice-orb-shell {
            width: 260px;
            height: 260px;
          }

          .voice-orb {
            width: 190px;
            height: 190px;
            font-size: 28px;
          }
        }
      `}</style>
    </>
  );
}
