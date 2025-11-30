import React, { useState, useEffect, useRef, useCallback } from 'react';
import { Mic, Camera, Settings, Volume2, Hand, StopCircle, Play, RotateCcw } from 'lucide-react';

// --- Types ---
type TranslationMode = 'signToVoice' | 'voiceToText';
type ConnectionStatus = 'disconnected' | 'connecting' | 'connected' | 'completed';
type Message = {
  id: string;
  source: 'input' | 'output';
  text: string;
  timestamp: number;
};

// --- Mock WebSocket Service ---
// Simulates the backend architecture described in Section 4 of the spec.
class MockKinesisBackend {
  private socketOpen = false;
  private messageCallback: ((data: any) => void) | null = null;
  private lastRecognitionTime = 0;
  
  // Deterministic sequences indices
  private signIndex = 0;
  private voiceIndex = 0;
  
  // Data definitions
  private readonly signs = [
    "Chào bạn", 
    "Bạn có thể hướng dẫn tôi tới trạm xe buýt Đại học Đà Nẵng không?"
  ];

  private readonly phrases = [
    "Để tôi giúp bạn",
    "Rẽ trái 500m, và trạm xe buýt nằm bên phải nhe"
  ];

  connect() {
    return new Promise<void>((resolve) => {
      console.log('[Kinesis WS] Connecting to wss://api.kinesis.danang/ws...');
      setTimeout(() => {
        this.socketOpen = true;
        console.log('[Kinesis WS] Connected. Session ID: sess_' + Math.random().toString(36).substr(2, 9));
        // Reset indices on new connection
        this.signIndex = 0;
        this.voiceIndex = 0;
        resolve();
      }, 1500);
    });
  }

  onMessage(cb: (data: any) => void) {
    this.messageCallback = cb;
  }

  sendFrame(frameData: string) {
    if (!this.socketOpen) return;
    
    // Check if we have finished the sequence
    if (this.signIndex >= this.signs.length) {
       // Optional: Notify frontend that sequence is done
       return;
    }

    // Throttled simulation
    const now = Date.now();
    // Increased throttle to 5000ms to allow TTS to finish and user to read
    if (now - this.lastRecognitionTime > 5000) {
      // High chance to recognize for demo purposes
      if (Math.random() > 0.3) { 
        this.lastRecognitionTime = now;
        this.simulateSignRecognition();
      }
    }
  }

  sendAudioChunk(audioData: Float32Array) {
    if (!this.socketOpen) return;

    // Check if we have finished the sequence
    if (this.voiceIndex >= this.phrases.length) {
       return;
    }
    
    // Throttled simulation for speech
    const now = Date.now();
    // Increased throttle to 5000ms
    if (now - this.lastRecognitionTime > 5000) {
      if (Math.random() > 0.3) {
        this.lastRecognitionTime = now;
        this.simulateASR();
      }
    }
  }

  private simulateSignRecognition() {
    if (this.signIndex >= this.signs.length) return;

    const text = this.signs[this.signIndex];
    this.signIndex++;
    
    // Calculate latency based on text length (0.4s to 0.9s)
    // ~30ms per character to be safe, clamped between 400ms and 900ms
    const latency = Math.max(400, Math.min(900, text.length * 30));
    
    setTimeout(() => {
      if (this.messageCallback && this.socketOpen) {
        this.messageCallback({
          type: 'SIGN_RECOGNITION',
          text: text,
          isCompleted: this.signIndex >= this.signs.length // Flag if last message
        });
      }
    }, latency);
  }

  private simulateASR() {
    if (this.voiceIndex >= this.phrases.length) return;

    const fullText = this.phrases[this.voiceIndex];
    this.voiceIndex++;

    // Simulate processing delay before first partial result
    setTimeout(() => {
      if (this.messageCallback && this.socketOpen) {
        // 1. Send intermediate result (partial)
        const partial = fullText.split(' ').slice(0, Math.max(1, Math.floor(fullText.split(' ').length / 2))).join(' ') + '...';
        
        this.messageCallback({
          type: 'ASR_TRANSCRIPT',
          text: partial,
          isFinal: false
        });

        // 2. Send final result after remaining processing time
        // Latency: 0.4s to 0.9s based on length
        const finalLatency = Math.max(400, Math.min(900, fullText.length * 30));

        setTimeout(() => {
          if (this.messageCallback && this.socketOpen) {
            this.messageCallback({
              type: 'ASR_TRANSCRIPT',
              text: fullText,
              isFinal: true,
              isCompleted: this.voiceIndex >= this.phrases.length
            });
          }
        }, finalLatency);
      }
    }, 300); // Initial 300ms network delay
  }

  disconnect() {
    this.socketOpen = false;
    console.log('[Kinesis WS] Disconnected.');
  }
}

export const TranslationMock: React.FC = () => {
  const [mode, setMode] = useState<TranslationMode>('signToVoice');
  const [status, setStatus] = useState<ConnectionStatus>('disconnected');
  const [messages, setMessages] = useState<Message[]>([]);
  const [interimResult, setInterimResult] = useState<string>(''); // For intermediate ASR
  const [error, setError] = useState<string | null>(null);
  
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const backendRef = useRef<MockKinesisBackend>(new MockKinesisBackend());
  const streamRef = useRef<MediaStream | null>(null);
  const animationFrameRef = useRef<number | null>(null);
  
  // Refs to track state inside the animation loop without closures
  const statusRef = useRef<ConnectionStatus>('disconnected');
  const modeRef = useRef<TranslationMode>('signToVoice');

  // Sync refs with state
  useEffect(() => { statusRef.current = status; }, [status]);
  useEffect(() => { modeRef.current = mode; }, [mode]);

  // Initialize Speech Synthesis for TTS fallback
  const speak = useCallback((text: string) => {
    if ('speechSynthesis' in window) {
      // Small delay to simulate generation time and avoid instant overlap
      setTimeout(() => {
        const utterance = new SpeechSynthesisUtterance(text);
        utterance.lang = 'vi-VN'; // Vietnamese
        // Tune for "Da Nang" feel (simulated): shallower and slightly faster
        utterance.rate = 1.1; 
        utterance.pitch = 1.1; 
        
        // Try to find a Vietnamese voice if available
        const voices = window.speechSynthesis.getVoices();
        const viVoice = voices.find(v => v.lang.includes('vi'));
        if (viVoice) utterance.voice = viVoice;
        
        window.speechSynthesis.speak(utterance);
      }, 300);
    }
  }, []);

  // Ensure voices are loaded (Chrome requirement)
  useEffect(() => {
    if ('speechSynthesis' in window) {
        window.speechSynthesis.getVoices();
    }
  }, []);

  // --- Attach Video Stream ---
  // When status becomes 'connected' and video element is mounted, attach the stream.
  useEffect(() => {
    if (status === 'connected' && mode === 'signToVoice' && videoRef.current && streamRef.current) {
      videoRef.current.srcObject = streamRef.current;
      videoRef.current.play().catch(e => console.error("Video play error:", e));
    }
  }, [status, mode]);

  // --- Streaming Logic ---
  // Defined here to be accessible, but relies on refs for live data
  const streamingLoop = useCallback(() => {
    if (statusRef.current !== 'connected') {
      if (animationFrameRef.current) cancelAnimationFrame(animationFrameRef.current);
      return;
    }

    if (modeRef.current === 'signToVoice') {
      // 1. Capture frame from video
      if (videoRef.current && canvasRef.current && videoRef.current.readyState === videoRef.current.HAVE_ENOUGH_DATA) {
        const ctx = canvasRef.current.getContext('2d');
        if (ctx) {
          ctx.drawImage(videoRef.current, 0, 0, 640, 360);
          // 2. Convert to Base64 (simulating frame encoding)
          const frameData = canvasRef.current.toDataURL('image/jpeg', 0.5);
          // 3. Send to backend
          backendRef.current.sendFrame(frameData);
        }
      }
    } else {
      // Voice mode: simulate sending audio chunks
      // In a real app we'd read from AudioContext AnalyserNode
      backendRef.current.sendAudioChunk(new Float32Array(1024));
    }

    animationFrameRef.current = requestAnimationFrame(streamingLoop);
  }, []);

  // --- Session Management ---
  const startSession = async () => {
    setStatus('connecting');
    setError(null);
    setMessages([]);
    setInterimResult('');

    try {
      // 1. Get User Media
      const constraints = {
        video: mode === 'signToVoice' ? { facingMode: 'user', width: 640, height: 360 } : false,
        audio: true
      };
      
      const stream = await navigator.mediaDevices.getUserMedia(constraints);
      streamRef.current = stream;

      // Note: We do NOT attach videoRef.current.srcObject here because videoRef.current is null
      // while status is 'connecting'. The useEffect above handles it when status becomes 'connected'.

      // 2. Connect to Backend
      await backendRef.current.connect();
      setStatus('connected');

      // 3. Setup Listeners
      backendRef.current.onMessage((data) => {
        if (data.type === 'SIGN_RECOGNITION') {
          const newMessage: Message = {
            id: Date.now().toString(),
            source: 'output',
            text: data.text,
            timestamp: Date.now()
          };
          setMessages(prev => [newMessage, ...prev].slice(0, 3));
          speak(data.text); // Play audio
          
          if (data.isCompleted) {
             setStatus('completed');
             // Stop the loop but keep media active for a moment or just stop logic
             // For UX, we switch status to completed so loop stops sending frames
          }

        } else if (data.type === 'ASR_TRANSCRIPT') {
          if (data.isFinal) {
            setInterimResult(''); // Clear interim
            const newMessage: Message = {
              id: Date.now().toString(),
              source: 'output',
              text: data.text,
              timestamp: Date.now()
            };
            setMessages(prev => [newMessage, ...prev].slice(0, 3));
            
            if (data.isCompleted) {
                setStatus('completed');
            }
          } else {
            // Update interim display
            setInterimResult(data.text);
          }
        }
      });

      // 4. Start Loop (triggered by useEffect below)

    } catch (err) {
      console.error("Failed to start session:", err);
      setError("Could not access camera/microphone. Please check permissions.");
      setStatus('disconnected');
    }
  };

  const stopSession = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
    }
    if (animationFrameRef.current) {
      cancelAnimationFrame(animationFrameRef.current);
    }
    backendRef.current.disconnect();
    setStatus('disconnected');
    setInterimResult('');
    if ('speechSynthesis' in window) {
      window.speechSynthesis.cancel();
    }
  };

  // Watch status to start/stop loop
  useEffect(() => {
    if (status === 'connected') {
      streamingLoop();
    } else {
      if (animationFrameRef.current) cancelAnimationFrame(animationFrameRef.current);
    }
  }, [status, streamingLoop]);

  // Clean up on unmount
  useEffect(() => {
    return () => {
      stopSession();
    };
  }, []);

  return (
    <div id="translator-interface" className="w-full max-w-md mx-auto bg-white rounded-3xl shadow-2xl border border-slate-100 overflow-hidden relative z-10 font-sans">
      {/* Canvas for internal processing (hidden) */}
      <canvas ref={canvasRef} className="hidden" width={640} height={360} />

      {/* Card Header */}
      <div className="bg-slate-50 p-4 border-b border-slate-100 flex items-center justify-between">
        <div className="flex items-center space-x-2">
          <span className="font-bold text-slate-800 text-sm tracking-wide">Kinesis Translator</span>
        </div>
        <div className="flex items-center space-x-2">
          <div className={`w-2 h-2 rounded-full ${status === 'connected' ? 'bg-green-500 animate-pulse' : status === 'completed' ? 'bg-blue-500' : 'bg-slate-300'}`}></div>
          <span className={`text-xs font-medium ${status === 'connected' ? 'text-green-700' : status === 'completed' ? 'text-blue-700' : 'text-slate-400'}`}>
            {status === 'connected' ? 'Connected · Live' : status === 'connecting' ? 'Connecting...' : status === 'completed' ? 'Session Complete' : 'Offline'}
          </span>
        </div>
      </div>

      {/* Mode Toggle */}
      <div className="p-4 pb-2">
        <div className="bg-slate-100 p-1 rounded-full flex relative">
          <button
            onClick={() => { setMode('signToVoice'); if(status !== 'disconnected') stopSession(); }}
            className={`flex-1 flex items-center justify-center space-x-2 py-2 rounded-full text-sm font-medium transition-all duration-300 ${
              mode === 'signToVoice' 
                ? 'bg-white text-primary-700 shadow-sm' 
                : 'text-slate-500 hover:text-slate-700'
            }`}
          >
            <Hand size={16} />
            <span>Sign → Voice</span>
          </button>
          <button
            onClick={() => { setMode('voiceToText'); if(status !== 'disconnected') stopSession(); }}
            className={`flex-1 flex items-center justify-center space-x-2 py-2 rounded-full text-sm font-medium transition-all duration-300 ${
              mode === 'voiceToText' 
                ? 'bg-white text-primary-700 shadow-sm' 
                : 'text-slate-500 hover:text-slate-700'
            }`}
          >
            <Mic size={16} />
            <span>Voice → Text</span>
          </button>
        </div>
      </div>

      {/* Body Layout */}
      <div className="p-4 space-y-4">
        
        {/* Input Panel */}
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider">
              Input · {mode === 'signToVoice' ? 'Sign' : 'Voice'}
            </span>
            <span className="bg-primary-50 text-primary-700 px-2 py-0.5 rounded text-[10px] font-medium border border-primary-100">
              {mode === 'signToVoice' ? 'Vietnamese Sign Language' : 'Spoken Vietnamese'}
            </span>
          </div>
          
          <div className="aspect-video bg-slate-900 rounded-xl overflow-hidden relative flex items-center justify-center group border border-slate-100 shadow-inner">
            {/* Logic for displaying media or placeholder */}
            {(status === 'connected' || status === 'completed') ? (
              mode === 'signToVoice' ? (
                <div className="relative w-full h-full">
                  <video 
                    ref={videoRef} 
                    autoPlay 
                    playsInline 
                    muted 
                    className={`w-full h-full object-cover transform scale-x-[-1] transition-opacity duration-500 ${status === 'completed' ? 'opacity-50 grayscale' : ''}`}
                  />
                  {status === 'connected' && (
                    <>
                      {/* Mock detection box */}
                      <div className="absolute top-1/4 left-1/4 w-1/2 h-1/2 border-2 border-primary-400/50 rounded-lg animate-pulse pointer-events-none"></div>
                      <div className="absolute bottom-2 right-2 px-1.5 py-0.5 bg-black/50 rounded text-[10px] text-white font-mono backdrop-blur-sm">
                        LIVE
                      </div>
                    </>
                  )}
                  {status === 'completed' && (
                     <div className="absolute inset-0 flex items-center justify-center">
                        <div className="bg-slate-900/80 px-4 py-2 rounded-full text-white text-sm font-medium flex items-center gap-2">
                           <Hand size={16} /> Conversation Ended
                        </div>
                     </div>
                  )}
                </div>
              ) : (
                <div className="w-full h-full flex flex-col items-center justify-center bg-slate-800 relative overflow-hidden">
                  {/* Abstract audio wave visualization */}
                  {status === 'connected' && (
                    <div className="flex items-end space-x-1 h-16 mb-4 z-10">
                       {[...Array(8)].map((_, i) => (
                         <div 
                          key={i} 
                          className="w-2 bg-primary-400 rounded-full animate-bounce" 
                          style={{ 
                            height: '40%',
                            animationDuration: `${0.8 + Math.random() * 0.5}s`,
                            animationDelay: `${i * 0.1}s` 
                          }}
                         ></div>
                       ))}
                    </div>
                  )}
                  {status === 'completed' && (
                    <div className="z-10 text-white font-medium mb-2">Translation Complete</div>
                  )}
                  <p className="text-xs text-slate-300 font-mono z-10">
                     {status === 'connected' ? "Listening..." : "Finished"}
                  </p>
                  
                  {/* Background concentric circles pulse */}
                  {status === 'connected' && <div className="absolute w-64 h-64 bg-primary-500/10 rounded-full animate-ping"></div>}
                </div>
              )
            ) : (
              // Placeholder when disconnected
              <div className="flex flex-col items-center justify-center text-slate-500 p-6 text-center">
                 {error ? (
                   <div className="text-red-400 mb-2 text-sm max-w-[200px]">{error}</div>
                 ) : (
                   <>
                    {mode === 'signToVoice' ? <Camera size={32} className="mb-2 opacity-50" /> : <Mic size={32} className="mb-2 opacity-50" />}
                    <p className="text-xs">Click Start to enable {mode === 'signToVoice' ? 'camera' : 'microphone'}</p>
                   </>
                 )}
              </div>
            )}
          </div>
        </div>

        {/* Output Panel */}
        <div className="space-y-2">
           <div className="flex items-center justify-between">
            <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider">
              Output · {mode === 'signToVoice' ? 'Voice' : 'Text'}
            </span>
            <span className="bg-slate-100 text-slate-600 px-2 py-0.5 rounded text-[10px] font-medium border border-slate-200">
              {mode === 'signToVoice' ? 'Spoken Vietnamese' : 'Vietnamese text'}
            </span>
          </div>

          <div className="bg-slate-50 rounded-xl p-4 border border-slate-100 min-h-[120px] flex flex-col justify-end transition-all overflow-hidden">
             {messages.length > 0 || interimResult ? (
               <div className="space-y-3">
                 {/* Interim result (streaming) */}
                 {interimResult && (
                    <div className="animate-pulse">
                      <p className="text-lg font-medium leading-tight text-slate-800">
                        "{interimResult}"
                      </p>
                      <div className="flex items-center gap-1 mt-1 text-slate-400">
                         <span className="w-1 h-1 bg-slate-400 rounded-full animate-bounce"></span>
                         <span className="w-1 h-1 bg-slate-400 rounded-full animate-bounce delay-100"></span>
                         <span className="w-1 h-1 bg-slate-400 rounded-full animate-bounce delay-200"></span>
                         <span className="text-[10px] font-bold uppercase ml-1">Processing</span>
                      </div>
                    </div>
                 )}
                 
                 {/* Finalized messages */}
                 {messages.map((msg, idx) => (
                   <div key={msg.id} className={`animate-in fade-in slide-in-from-bottom-2 duration-300 ${idx === 0 && !interimResult ? 'opacity-100' : 'opacity-60'}`}>
                     <p className={`text-lg font-medium leading-tight ${idx === 0 && !interimResult ? 'text-slate-800' : 'text-slate-400 text-base'}`}>
                       "{msg.text}"
                     </p>
                     {idx === 0 && !interimResult && mode === 'signToVoice' && (
                        <div className="flex items-center gap-2 text-primary-600 text-[10px] font-bold uppercase mt-1">
                          <Volume2 size={10} className="animate-pulse" />
                          Speaking
                        </div>
                     )}
                   </div>
                 ))}
               </div>
             ) : (
               <div className="text-center py-4">
                 <p className="text-sm text-slate-400 italic">
                   {status === 'connected' 
                     ? (mode === 'signToVoice' ? "Waiting for sign language..." : "Waiting for speech...") 
                     : "Ready to translate"}
                 </p>
               </div>
             )}
          </div>
        </div>

        {/* Controls Row */}
        <div className="pt-2 flex justify-center space-x-4">
          {status === 'connected' ? (
             <button 
               onClick={stopSession}
               className="w-14 h-14 rounded-full bg-red-50 text-red-600 border border-red-100 hover:bg-red-100 flex items-center justify-center transition-colors shadow-sm"
               title="Stop Session"
             >
               <StopCircle size={24} />
             </button>
          ) : status === 'completed' ? (
            <button 
               onClick={startSession}
               className="w-14 h-14 rounded-full bg-slate-900 text-white shadow-lg hover:bg-slate-800 flex items-center justify-center transition-all"
               title="Restart Session"
             >
               <RotateCcw size={24} />
             </button>
          ) : (
             <button 
               onClick={startSession}
               className="w-14 h-14 rounded-full bg-primary-600 text-white shadow-lg shadow-primary-500/30 hover:bg-primary-700 flex items-center justify-center transition-transform hover:scale-105"
               title="Start Session"
             >
               <Play size={24} className="ml-1" />
             </button>
          )}
          
           <button className="w-12 h-12 rounded-full bg-slate-100 text-slate-600 hover:bg-slate-200 flex items-center justify-center transition-colors">
            <Settings size={20} />
          </button>
        </div>

      </div>
    </div>
  );
};