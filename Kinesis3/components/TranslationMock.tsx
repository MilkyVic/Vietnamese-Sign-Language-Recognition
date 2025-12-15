import React, { useEffect, useRef, useState } from 'react';
import { Camera, Mic, Settings, Volume2, Play, StopCircle, Hand } from 'lucide-react';

type Mode = 'signToVoice' | 'voiceToText';
type Status = 'idle' | 'recording' | 'processing' | 'success' | 'error';

const API_BASE =
  import.meta.env.VITE_API_BASE ||
  (typeof window !== 'undefined' ? window.location.origin : 'http://127.0.0.1:8001');

export const TranslationMock: React.FC = () => {
  const [mode, setMode] = useState<Mode>('signToVoice');
  const [status, setStatus] = useState<Status>('idle');
  const [result, setResult] = useState<string>('');
  const [error, setError] = useState<string | null>(null);
  const [audioUrl, setAudioUrl] = useState<string | null>(null);
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const audioRef = useRef<HTMLAudioElement | null>(null);

  useEffect(() => {
    if (audioUrl && audioRef.current) {
      audioRef.current.play().catch(() => {});
    }
  }, [audioUrl]);

  const resetAll = () => {
    setResult('');
    setError(null);
    setAudioUrl(null);
    setStatus('idle');
  };

  const stopStream = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    }
  };

  const startCamera = async () => {
    try {
      stopStream();
      const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'user' }, audio: false });
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play().catch(() => {});
      }
    } catch (err: any) {
      setError(err.message || 'Cannot access camera.');
      setStatus('error');
    }
  };

  const startMic = async () => {
    try {
      stopStream();
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;
    } catch (err: any) {
      setError(err.message || 'Cannot access microphone.');
      setStatus('error');
    }
  };

  const startRecording = async () => {
    // If already recording, stop instead of starting a new one
    if (status === 'recording') {
      stopRecording();
      return;
    }

    setResult('');
    setError(null);
    setAudioUrl(null);
    setStatus('recording');
    chunksRef.current = [];

    try {
      if (mode === 'signToVoice') {
        await startCamera();
      } else {
        await startMic();
      }
      if (!streamRef.current) throw new Error('No media stream');
      const recorder = new MediaRecorder(streamRef.current, {
        mimeType: mode === 'signToVoice' ? 'video/webm' : 'audio/webm',
      });
      mediaRecorderRef.current = recorder;
      recorder.ondataavailable = (e) => {
        if (e.data.size > 0) chunksRef.current.push(e.data);
      };
      recorder.onstop = async () => {
        const blob = new Blob(chunksRef.current, {
          type: mode === 'signToVoice' ? 'video/webm' : 'audio/webm',
        });
        const file = new File([blob], `${mode}-${Date.now()}.${mode === 'signToVoice' ? 'webm' : 'webm'}`, {
          type: blob.type,
        });
        stopStream();
        await sendToApi(file);
      };
      recorder.start();
      if (mode === 'signToVoice') {
        // force stop after 5s for sign capture
        setTimeout(() => {
          if (recorder.state !== 'inactive') recorder.stop();
        }, 5000);
      }
    } catch (err: any) {
      setError(err.message || 'Record failed');
      setStatus('error');
      stopStream();
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
      mediaRecorderRef.current.stop();
      setStatus('processing');
    }
  };

  const sendToApi = async (file: File) => {
    setStatus('processing');
    try {
      if (mode === 'signToVoice') {
        const form = new FormData();
        form.append('file', file);
        const resp = await fetch(`${API_BASE}/predict/video`, { method: 'POST', body: form });
        if (!resp.ok) {
          const data = await resp.json().catch(() => ({}));
          throw new Error(data.detail || 'Prediction failed');
        }
        const data = await resp.json();
        setResult(data.label || '');
        if (data.audio_url) {
          const full = data.audio_url.startsWith('http') ? data.audio_url : `${API_BASE}${data.audio_url}`;
          setAudioUrl(full);
        }
      } else {
        const form = new FormData();
        form.append('file', file);
        const resp = await fetch(`${API_BASE}/transcribe`, { method: 'POST', body: form });
        if (!resp.ok) {
          const data = await resp.json().catch(() => ({}));
          throw new Error(data.detail || 'Transcription failed');
        }
        const data = await resp.json();
        setResult(data.text || '');
      }
      setStatus('success');
    } catch (err: any) {
      setError(err.message || 'API failed');
      setStatus('error');
    }
  };

  const renderOutput = () => {
    if (status === 'processing') return <p className="text-sm text-slate-500">Processing...</p>;
    if (status === 'error' && error) return <p className="text-sm text-red-500">{error}</p>;
    if (status === 'success' && result) return <p className="text-lg font-semibold text-slate-900">"{result}"</p>;
    return <p className="text-sm text-slate-400 italic">Waiting for input...</p>;
  };

  const renderVoiceToTextUI = () => {
    const isRecording = status === 'recording';
    const isProcessing = status === 'processing';
    const statusLabel = isRecording ? 'Recording...' : isProcessing ? 'Processing...' : 'Tap start to begin';

    return (
      <div className="p-4 space-y-5">
        {/* Input */}
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider">Input</span>
            <span className="bg-emerald-100 text-emerald-700 px-3 py-1 rounded-full text-xs font-semibold border border-emerald-200">
              Microphone
            </span>
          </div>
          <button
            onClick={startRecording}
            className={`w-full text-center rounded-3xl border transition-all shadow-sm ${
              isRecording
                ? 'border-red-200 bg-red-50'
                : 'border-slate-100 bg-slate-50 hover:border-primary-100'
            }`}
          >
            <div className="flex flex-col items-center justify-center py-10 gap-2 text-slate-400">
              <Mic size={28} />
              <span className="text-sm font-semibold tracking-wide uppercase">{statusLabel}</span>
            </div>
          </button>
        </div>

        {/* Output */}
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider">Output</span>
            <span className="bg-slate-100 text-slate-600 px-3 py-1 rounded-full text-xs font-semibold border border-slate-200">
              Sign Animation
            </span>
          </div>
          <div className="rounded-3xl border border-slate-100 bg-slate-50 min-h-[220px] flex flex-col items-center justify-center gap-4 text-slate-300">
            <Hand size={44} />
            <span className="text-sm font-semibold uppercase tracking-wide">Waiting...</span>
          </div>

          <div className="pt-4">
            <div className="text-xs font-semibold text-slate-500 uppercase mb-2">English</div>
            <div className="rounded-2xl border border-slate-100 bg-white min-h-[80px] p-3 text-slate-700 text-sm leading-relaxed">
              {status === 'processing' && <span className="text-slate-400">Transcribing...</span>}
              {status === 'success' && result && <span>{result}</span>}
              {status !== 'success' && status !== 'processing' && <span className="text-slate-300">•••</span>}
            </div>
          </div>
        </div>
      </div>
    );
  };

  return (
    <div
      id="translator-interface"
      className="w-full max-w-md mx-auto bg-white rounded-3xl shadow-2xl border border-slate-100 overflow-hidden relative z-10 font-sans"
    >
      {/* Header */}
      <div className="bg-slate-50 p-4 border-b border-slate-100 flex items-center justify-between">
        <div className="flex items-center space-x-2">
          <span className="font-bold text-slate-800 text-sm tracking-wide">Kinesis Translator</span>
        </div>
        <div className="bg-slate-100 p-1 rounded-full flex relative">
          <button
            onClick={() => {
              stopStream();
              setMode('signToVoice');
              resetAll();
            }}
            className={`flex items-center gap-2 px-3 py-1.5 rounded-full text-xs font-semibold ${
              mode === 'signToVoice' ? 'bg-white shadow text-primary-700' : 'text-slate-500'
            }`}
          >
            <Camera size={14} /> Sign → Voice
          </button>
          <button
            onClick={() => {
              stopStream();
              setMode('voiceToText');
              resetAll();
            }}
            className={`flex items-center gap-2 px-3 py-1.5 rounded-full text-xs font-semibold ${
              mode === 'voiceToText' ? 'bg-white shadow text-primary-700' : 'text-slate-500'
            }`}
          >
            <Mic size={14} /> Voice → Text
          </button>
        </div>
      </div>

      <div className="p-4 space-y-4">
        {mode === 'voiceToText' ? (
          renderVoiceToTextUI()
        ) : (
          <>
            {/* Input (Sign -> Voice) */}
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider">Input · Sign (video)</span>
                <span className="bg-primary-50 text-primary-700 px-2 py-0.5 rounded text-[10px] font-medium border border-primary-100">
                  Live camera (5s)
                </span>
              </div>

              <div className="aspect-video bg-slate-900 rounded-xl overflow-hidden relative flex flex-col items-center justify-center group border border-slate-100 shadow-inner p-4 gap-3">
                <video ref={videoRef} className="w-full h-full object-cover rounded-lg" autoPlay muted playsInline />
                {!streamRef.current && status !== 'recording' && (
                  <p className="text-slate-300 text-sm">Start to capture</p>
                )}
              </div>
            </div>

            {/* Output (Sign -> Voice) */}
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider">Output · Voice</span>
                <span className="bg-slate-100 text-slate-600 px-2 py-0.5 rounded text-[10px] font-medium border border-slate-200">
                  Spoken Vietnamese
                </span>
              </div>

              <div className="bg-slate-50 rounded-xl p-4 border border-slate-100 min-h-[140px] flex flex-col gap-2">
                {renderOutput()}
                {audioUrl && (
                  <button
                    onClick={() => audioRef.current?.play()}
                    className="inline-flex items-center gap-2 text-xs text-primary-700 font-semibold mt-2"
                  >
                    <Volume2 size={14} /> Play audio
                  </button>
                )}
                <audio ref={audioRef} src={audioUrl || undefined} />
              </div>
            </div>

            {/* Controls */}
            <div className="pt-2 flex justify-center space-x-4">
              <button
                onClick={startRecording}
                className={`w-14 h-14 rounded-full flex items-center justify-center transition-transform hover:scale-105 ${
                  status === 'recording'
                    ? 'bg-red-500 text-white shadow-lg shadow-red-400/40 hover:bg-red-600'
                    : 'bg-primary-600 text-white shadow-lg shadow-primary-500/30 hover:bg-primary-700'
                }`}
                title={status === 'recording' ? 'Stop capture' : 'Start capture'}
                disabled={status === 'processing'}
              >
                {status === 'recording' ? <StopCircle size={20} /> : <Play size={20} className="ml-1" />}
              </button>
              <button
                className="w-12 h-12 rounded-full bg-slate-100 text-slate-600 hover:bg-slate-200 flex items-center justify-center transition-colors"
                title="Settings"
              >
                <Settings size={18} />
              </button>
            </div>
          </>
        )}
      </div>
    </div>
  );
};
