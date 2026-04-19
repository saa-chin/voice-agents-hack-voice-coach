'use client';

import { useMemo, useRef, useState } from 'react';

type AnalysisResponse = {
  referencePhrase: string;
  overallScore: number;
  metrics: Record<string, number>;
  scores: Record<string, number>;
  feedback: string[];
  disclaimer: string;
};

const apiBase = process.env.NEXT_PUBLIC_API_BASE_URL ?? 'http://localhost:8000';
const referencePhrase = 'The quick brown fox jumps over the lazy dog';
const preferredMimeTypes = ['audio/webm;codecs=opus', 'audio/webm', 'audio/mp4', 'audio/ogg;codecs=opus'];

function formatLabel(value: string) {
  return value
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

export default function VoiceRecorder() {
  const recorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);

  const [isRecording, setIsRecording] = useState(false);
  const [audioUrl, setAudioUrl] = useState<string | null>(null);
  const [audioBlob, setAudioBlob] = useState<Blob | null>(null);
  const [result, setResult] = useState<AnalysisResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const sortedScores = useMemo(() => {
    if (!result) return [] as Array<[string, number]>;
    return Object.entries(result.scores);
  }, [result]);

  async function startRecording() {
    setError(null);
    setResult(null);

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mimeType = getSupportedRecordingMimeType();
      const recorder = mimeType ? new MediaRecorder(stream, { mimeType }) : new MediaRecorder(stream);
      chunksRef.current = [];

      recorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          chunksRef.current.push(event.data);
        }
      };

      recorder.onstop = () => {
        if (audioUrl) {
          URL.revokeObjectURL(audioUrl);
        }

        const blob = new Blob(chunksRef.current, { type: recorder.mimeType || mimeType || 'audio/webm' });
        setAudioBlob(blob);
        setAudioUrl(URL.createObjectURL(blob));
        stream.getTracks().forEach((track) => track.stop());
      };

      recorderRef.current = recorder;
      recorder.start();
      setIsRecording(true);
    } catch (err) {
      setError('Microphone access failed. Please allow mic permissions and try again.');
      console.error(err);
    }
  }

  function stopRecording() {
    recorderRef.current?.stop();
    setIsRecording(false);
  }

  async function analyzeRecording() {
    if (!audioBlob) {
      setError('Please record audio first.');
      return;
    }

    try {
      setLoading(true);
      setError(null);
      const wavBlob = await convertBlobToWav(audioBlob);
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

  return (
    <div style={{ maxWidth: 960, margin: '0 auto', padding: 24 }}>
      <div style={{ display: 'grid', gap: 16 }}>
        <section
          style={{
            padding: 24,
            borderRadius: 24,
            background: 'rgba(20, 27, 52, 0.95)',
            border: '1px solid var(--card-border)',
            boxShadow: '0 10px 30px rgba(0,0,0,0.25)',
          }}
        >
          <p style={{ color: 'var(--accent)', marginTop: 0, fontWeight: 700 }}>Parkinson&apos;s Voice Practice MVP</p>
          <h1 style={{ marginTop: 0, fontSize: 36 }}>Record, analyze, and coach one short speech sample.</h1>
          <p style={{ color: 'var(--muted)', fontSize: 18, lineHeight: 1.6 }}>
            Read this phrase aloud: <strong>{referencePhrase}</strong>
          </p>

          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 12, marginTop: 20 }}>
            {!isRecording ? (
              <button
                onClick={startRecording}
                style={{
                  padding: '12px 18px',
                  borderRadius: 14,
                  border: 'none',
                  background: 'var(--accent)',
                  color: '#09111f',
                  fontWeight: 700,
                }}
              >
                Start recording
              </button>
            ) : (
              <button
                onClick={stopRecording}
                style={{
                  padding: '12px 18px',
                  borderRadius: 14,
                  border: '1px solid #ff8f8f',
                  background: '#311a1f',
                  color: '#ffd4d4',
                  fontWeight: 700,
                }}
              >
                Stop recording
              </button>
            )}

            <button
              onClick={analyzeRecording}
              disabled={!audioBlob || loading}
              style={{
                padding: '12px 18px',
                borderRadius: 14,
                border: '1px solid var(--card-border)',
                background: !audioBlob || loading ? '#202846' : '#1d2647',
                color: 'var(--text)',
                fontWeight: 700,
                opacity: !audioBlob || loading ? 0.6 : 1,
              }}
            >
              {loading ? 'Analyzing…' : 'Analyze recording'}
            </button>
          </div>

          {audioUrl ? (
            <div style={{ marginTop: 18 }}>
              <audio controls src={audioUrl} style={{ width: '100%' }} />
            </div>
          ) : null}

          {error ? (
            <p style={{ marginTop: 16, color: '#ffb2b2' }}>{error}</p>
          ) : (
            <p style={{ marginTop: 16, color: 'var(--muted)' }}>
              Tip: keep the mic about 6–8 inches away and record in a quiet room.
            </p>
          )}
        </section>

        {result ? (
          <>
            <section
              style={{
                padding: 24,
                borderRadius: 24,
                background: 'rgba(20, 27, 52, 0.95)',
                border: '1px solid var(--card-border)',
              }}
            >
              <h2 style={{ marginTop: 0 }}>Overall score: {result.overallScore}/100</h2>
              <p style={{ color: 'var(--muted)' }}>{result.disclaimer}</p>

              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', gap: 12 }}>
                {sortedScores.map(([key, value]) => (
                  <div
                    key={key}
                    style={{
                      padding: 16,
                      borderRadius: 18,
                      border: '1px solid var(--card-border)',
                      background: '#121a31',
                    }}
                  >
                    <div style={{ color: 'var(--muted)', marginBottom: 8 }}>{formatLabel(key)}</div>
                    <div style={{ fontSize: 28, fontWeight: 700 }}>{value}</div>
                  </div>
                ))}
              </div>
            </section>

            <section
              style={{
                padding: 24,
                borderRadius: 24,
                background: 'rgba(20, 27, 52, 0.95)',
                border: '1px solid var(--card-border)',
              }}
            >
              <h2 style={{ marginTop: 0 }}>Feedback</h2>
              <ul style={{ margin: 0, paddingLeft: 20, lineHeight: 1.8 }}>
                {result.feedback.map((item) => (
                  <li key={item}>{item}</li>
                ))}
              </ul>
            </section>

            <section
              style={{
                padding: 24,
                borderRadius: 24,
                background: 'rgba(20, 27, 52, 0.95)',
                border: '1px solid var(--card-border)',
              }}
            >
              <h2 style={{ marginTop: 0 }}>Raw metrics</h2>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', gap: 12 }}>
                {Object.entries(result.metrics).map(([key, value]) => (
                  <div
                    key={key}
                    style={{
                      padding: 16,
                      borderRadius: 18,
                      border: '1px solid var(--card-border)',
                      background: '#121a31',
                    }}
                  >
                    <div style={{ color: 'var(--muted)', marginBottom: 8 }}>{formatLabel(key)}</div>
                    <div style={{ fontSize: 24, fontWeight: 700 }}>{value}</div>
                  </div>
                ))}
              </div>
            </section>
          </>
        ) : null}
      </div>
    </div>
  );
}
