import React, { useState, useCallback, useRef, useEffect } from 'react';
import {
  Mic,
  Hand,
  Globe,
  Camera,
  Smartphone,
  MessageSquare,
  Play,
  Menu,
  X,
  ChevronRight,
  Accessibility,
  Heart,
  Upload,
  Volume2,
  RefreshCw,
  ArrowLeft,
  User,
  Video
} from 'lucide-react';

/* --- LOGO COMPONENT --- */
const KinesisLogo = ({ darkMode = false }) => (
  <div className="flex items-center gap-3 cursor-pointer">
    <div className="relative w-12 h-12 flex-shrink-0">
      <div className="absolute inset-0 bg-[#2998b3] rounded-full flex items-center justify-center">
        <svg
          viewBox="0 0 24 24"
          fill="none"
          stroke="white"
          strokeWidth="2"
          strokeLinecap="round"
          strokeLinejoin="round"
          className="w-7 h-7 translate-y-[1px]"
        >
          <path d="M18 11V6a2 2 0 0 0-2-2v0a2 2 0 0 0-2 2v0" />
          <path d="M14 10V4a2 2 0 0 0-2-2v0a2 2 0 0 0-2 2v2" />
          <path d="M10 10.5V6a2 2 0 0 0-2-2v0a2 2 0 0 0-2 2v8" />
          <path d="M18 8a2 2 0 1 1 4 0v6a8 8 0 0 1-8 8h-2c-2.8 0-4.5-.86-5.99-2.34l-3.6-3.6a2 2 0 0 1 2.83-2.82L7 15" />
        </svg>
      </div>
    </div>
    <div className="flex flex-col">
      <span className={`text-2xl font-bold leading-none tracking-tight ${darkMode ? 'text-white' : 'text-[#14294a]'}`}>
        KINESIS
      </span>
      <span className={`text-[0.6rem] font-bold uppercase tracking-wide leading-tight ${darkMode ? 'text-slate-400' : 'text-[#14294a]'}`}>
        For Deaf Community<br/>in Da Nang
      </span>
    </div>
  </div>
);

/* --- LANDING PAGE COMPONENTS --- */
const SectionHeading = ({ children, className = "" }) => (
  <h2 className={`text-3xl md:text-4xl font-bold text-[#14294a] mb-6 ${className}`}>
    {children}
  </h2>
);

const FeatureCard = ({ icon: Icon, title, description }) => (
  <div className="bg-white p-8 rounded-2xl shadow-sm border border-slate-100 hover:shadow-md transition-shadow">
    <div className="w-12 h-12 bg-[#e6f4f7] rounded-xl flex items-center justify-center mb-6 text-[#2998b3]">
      <Icon size={24} strokeWidth={2.5} />
    </div>
    <h3 className="text-xl font-bold text-[#14294a] mb-3">{title}</h3>
    <p className="text-slate-600 leading-relaxed">{description}</p>
  </div>
);

const StepCard = ({ number, icon: Icon, title, description, color }) => (
  <div className="relative group">
    <div className="bg-white p-6 rounded-2xl border border-slate-100 shadow-sm relative z-10 h-full">
      <div className="flex items-center gap-4 mb-4">
        <span className="text-5xl font-black text-slate-100 absolute -top-4 -right-2 select-none group-hover:text-[#e6f4f7] transition-colors">
          {number}
        </span>
        <div className={`w-10 h-10 rounded-full flex items-center justify-center ${color} text-white`}>
          <Icon size={20} />
        </div>
        <h3 className="text-lg font-bold text-[#14294a]">{title}</h3>
      </div>
      <p className="text-slate-600 relative z-10">{description}</p>
    </div>
  </div>
);

/* --- NEW TRANSLATION APP INTERFACE --- */
const TranslationApp = ({ onBack }) => {
  const [activeTab, setActiveTab] = useState('live'); // 'live' or 'upload'
  const [translation, setTranslation] = useState(null);
  const [status, setStatus] = useState('Ready'); // 'Ready', 'Translating', 'Success', 'Error'
  const [error, setError] = useState(null);
  const [audioUrl, setAudioUrl] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [cameraError, setCameraError] = useState(null);
  const [isCameraReady, setIsCameraReady] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const fileInputRef = useRef(null);
  const audioRef = useRef(null);
  const videoRef = useRef(null);
  const mediaRecorderRef = useRef(null);
  const streamRef = useRef(null);

  const playAudio = useCallback(() => {
    if (audioRef.current) {
      audioRef.current.play();
    }
  }, []);

  const handleFileUpload = async (file) => {
    if (!file) return;
    // update video preview for both upload and live-recorded files
    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
    }
    setPreviewUrl(URL.createObjectURL(file));

    setStatus('Translating');
    setTranslation(null);
    setError(null);
    setAudioUrl(null);

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch('/predict/video', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Something went wrong');
      }

      const result = await response.json();
      setTranslation(result);
      if (result.audio_url) {
        setAudioUrl(result.audio_url);
      }
      setStatus('Success');
    } catch (err) {
      setError(err.message);
      setStatus('Error');
    }
  };

  useEffect(() => {
    const stopStream = () => {
      if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
        mediaRecorderRef.current.stop();
      }
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((track) => track.stop());
        streamRef.current = null;
      }
      setIsCameraReady(false);
    };

    if (activeTab !== 'live') {
      stopStream();
      return () => {};
    }

    let cancelled = false;
    const enableCamera = async () => {
      setCameraError(null);
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: true });
        if (cancelled) {
          stream.getTracks().forEach((track) => track.stop());
          return;
        }
        streamRef.current = stream;
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
        }
        setIsCameraReady(true);
      } catch (camErr) {
        setCameraError('Trình duyệt chặn quyền camera hoặc không có thiết bị phù hợp.');
        setIsCameraReady(false);
      }
    };

    enableCamera();

    return () => {
      cancelled = true;
      stopStream();
    };
  }, [activeTab]);

  useEffect(() => {
    if (status === 'Success' && audioUrl) {
      playAudio();
    }
  }, [status, audioUrl, playAudio]);

  const onFileSelect = (e) => {
    const file = e.target.files[0];
    handleFileUpload(file);
  };

  const onDrop = (e) => {
    e.preventDefault();
    const file = e.dataTransfer.files[0];
    handleFileUpload(file);
  };

  const onDragOver = (e) => {
    e.preventDefault();
  };

  const recordFromCamera = () => {
    if (!streamRef.current) {
      setCameraError('Không thể truy cập camera. Hãy kiểm tra quyền truy cập.');
      setStatus('Error');
      setError('Không thể truy cập camera.');
      return;
    }

    let recorder;
    try {
      recorder = new MediaRecorder(streamRef.current, { mimeType: 'video/webm' });
    } catch (recErr) {
      setCameraError('Trình duyệt không hỗ trợ ghi hình (MediaRecorder/video/webm).');
      setStatus('Error');
      setError('Trình duyệt không hỗ trợ ghi hình.');
      return;
    }

    const chunks = [];
    recorder.ondataavailable = (event) => {
      if (event.data.size > 0) {
        chunks.push(event.data);
      }
    };
    recorder.onstop = () => {
      setIsRecording(false);
      const blob = new Blob(chunks, { type: 'video/webm' });
      const file = new File([blob], `live-${Date.now()}.webm`, { type: 'video/webm' });
      mediaRecorderRef.current = null;
      handleFileUpload(file);
    };

    mediaRecorderRef.current = recorder;
    setIsRecording(true);
    setStatus('Translating');
    setTranslation(null);
    setError(null);
    setAudioUrl(null);
    recorder.start();
    // Auto-stop after 5 seconds to keep the clip small
    setTimeout(() => {
      if (recorder.state !== 'inactive') {
        recorder.stop();
      }
    }, 5000);
  };

  return (
    <div className="min-h-screen bg-slate-50 flex flex-col font-sans">
      {audioUrl && <audio ref={audioRef} src={audioUrl} />}
      {/* App Header */}
      <header className="bg-white border-b border-slate-200 px-6 py-4 flex items-center justify-between shadow-sm z-10">
        <div onClick={onBack} title="Back to Home">
          <KinesisLogo darkMode={false} />
        </div>
        <button
          onClick={onBack}
          className="flex items-center gap-2 text-slate-500 hover:text-[#14294a] transition-colors text-sm font-medium"
        >
          <ArrowLeft size={16} />
          Back to Website
        </button>
      </header>

      {/* Main App Content */}
      <main className="flex-1 max-w-7xl mx-auto w-full p-6 md:p-8 flex flex-col">

        {/* Title */}
        <h1 className="text-3xl font-bold text-[#14294a] text-center mb-8">Try Kinesis Now</h1>

        {/* Toggle Switch */}
        <div className="flex justify-center mb-10">
          <div className="bg-white p-1 rounded-full border border-slate-200 shadow-sm inline-flex">
            <button
              onClick={() => setActiveTab('live')}
              className={`flex items-center gap-2 px-6 py-2.5 rounded-full font-medium transition-all text-sm ${
                activeTab === 'live'
                  ? 'bg-[#e6f4f7] text-[#2998b3] shadow-sm'
                  : 'text-slate-500 hover:bg-slate-50'
              }`}
            >
              <Camera size={18} />
              Live Camera
            </button>
            <button
              onClick={() => setActiveTab('upload')}
              className={`flex items-center gap-2 px-6 py-2.5 rounded-full font-medium transition-all text-sm ${
                activeTab === 'upload'
                  ? 'bg-[#e6f4f7] text-[#2998b3] shadow-sm'
                  : 'text-slate-500 hover:bg-slate-50'
              }`}
            >
              <Upload size={18} />
              Upload Video
            </button>
          </div>
        </div>

        {/* Content Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 h-[600px]">

          {/* Left Panel: Camera/Input */}
          <div className="bg-[#0f172a] rounded-3xl overflow-hidden shadow-xl border border-slate-800 flex flex-col relative group">
             {activeTab === 'live' ? (
                <>
                  <div className="flex-1 relative bg-black">
                    <video
                      ref={videoRef}
                      autoPlay
                      muted
                      playsInline
                      className="w-full h-full object-cover bg-black"
                    />
                    {!isCameraReady && !cameraError && (
                      <div className="absolute inset-0 flex flex-col items-center justify-center gap-2 text-slate-300 bg-slate-900/70">
                        <Video size={32} />
                        <p>Đang mở camera...</p>
                      </div>
                    )}
                    {cameraError && (
                      <div className="absolute inset-0 flex flex-col items-center justify-center gap-2 text-red-200 bg-slate-900/80 px-6 text-center">
                        <Video size={32} />
                        <p>{cameraError}</p>
                      </div>
                    )}
                    {isRecording && (
                      <div className="absolute top-4 left-4 flex items-center gap-2 text-red-400 font-semibold">
                        <span className="w-3 h-3 rounded-full bg-red-500 animate-pulse"></span>
                        Đang ghi 5 giây...
                      </div>
                    )}
                  </div>
                  <div className="p-4 bg-slate-900 flex items-center justify-between text-sm">
                    <div className="text-slate-300">
                      {isCameraReady && !cameraError ? 'Camera đã sẵn sàng' : cameraError || 'Đang yêu cầu quyền camera...'}
                    </div>
                    <button
                      onClick={recordFromCamera}
                      disabled={!isCameraReady || isRecording}
                      className={`flex items-center gap-2 px-4 py-2 rounded-full font-semibold transition ${
                        !isCameraReady || isRecording
                          ? 'bg-slate-700 text-slate-400 cursor-not-allowed'
                          : 'bg-[#e11d48] text-white hover:bg-[#be123c]'
                      }`}
                    >
                      <Video size={18} />
                      {isRecording ? 'Đang ghi...' : 'Ghi 5s & dịch'}
                    </button>
                  </div>
                </>
             ) : (
                <div
                  className="flex-1 flex flex-col items-center justify-center text-slate-400 p-3 md:p-4 border-2 border-dashed border-slate-700 m-4 rounded-2xl cursor-pointer relative overflow-hidden bg-slate-900"
                  onClick={() => fileInputRef.current.click()}
                  onDrop={onDrop}
                  onDragOver={onDragOver}
                >
                   <input type="file" ref={fileInputRef} onChange={onFileSelect} className="hidden" accept="video/*" />
                   {previewUrl ? (
                     <video
                       src={previewUrl}
                       controls
                       autoPlay
                       loop
                       className="w-full h-full object-contain bg-black rounded-xl"
                     />
                   ) : (
                     <>
                       <Upload size={48} className="mb-4 text-slate-500" />
                       <p className="font-medium text-lg text-white">Upload Video</p>
                       <p className="text-sm mt-1">Drag & drop or click to browse</p>
                     </>
                   )}
                </div>
             )}
          </div>

          {/* Right Panel: Output */}
          <div className="bg-white rounded-3xl shadow-lg border border-slate-100 p-8 flex flex-col relative">
            <div className="flex items-center justify-between mb-8">
              <div className="flex items-center gap-3">
                <div className="w-2 h-2 rounded-full bg-[#2998b3]"></div>
                <div className="w-2 h-2 rounded-full bg-[#2998b3]/50"></div>
                <h2 className="text-xl font-bold text-[#14294a] flex items-center gap-2">
                  <Mic className="text-[#2998b3]" size={24} />
                  Live Translation
                </h2>
              </div>
              <div className="flex gap-1">
                 {[1,2,3,4,5,6].map(i => (
                   <div key={i} className="w-1.5 h-1.5 rounded-full bg-blue-100"></div>
                 ))}
              </div>
            </div>

            <div className="flex-1">
              {status === 'Translating' && (
                <p className="text-slate-500 text-2xl font-light leading-relaxed">Translating...</p>
              )}
              {status === 'Success' && translation && (
                <div>
                  <p className="text-5xl font-bold text-[#14294a]">{translation.label}</p>
                  <p className="text-lg text-slate-500 mt-2">Confidence: {(translation.confidence * 100).toFixed(2)}%</p>
                </div>
              )}
              {status === 'Error' && (
                 <p className="text-red-500 text-2xl font-light leading-relaxed">Error: {error}</p>
              )}
              {status === 'Ready' && (
                <p className="text-slate-300 text-2xl font-light italic leading-relaxed">
                  Waiting for input... Start camera or upload video to begin translation.
                </p>
              )}
            </div>

            <div className="mt-auto pt-6 border-t border-slate-100 flex items-center justify-between text-sm">
              <div className="flex items-center gap-2 text-slate-500 font-mono">
                <span>Status:</span>
                <span className={`font-medium px-2 py-0.5 rounded ${
                  status === 'Ready' ? 'text-[#2998b3] bg-[#e6f4f7]' :
                  status === 'Translating' ? 'text-amber-600 bg-amber-100' :
                  status === 'Success' ? 'text-green-600 bg-green-100' :
                  'text-red-600 bg-red-100'
                }`}>{status}</span>
              </div>
              <Volume2
                className={`cursor-pointer ${audioUrl ? 'text-[#2998b3]' : 'text-slate-400'}`}
                size={20}
                onClick={playAudio}
              />
            </div>
          </div>

        </div>
      </main>
    </div>
  );
};

/* --- MAIN APP COMPONENT --- */
export default function App() {
  const [currentView, setCurrentView] = useState('landing'); // 'landing' or 'app'
  const [isMenuOpen, setIsMenuOpen] = useState(false);

  const toggleMenu = () => setIsMenuOpen(!isMenuOpen);
  const startTranslating = () => {
    setCurrentView('app');
    setIsMenuOpen(false);
    window.scrollTo(0, 0);
  };

  if (currentView === 'app') {
    return <TranslationApp onBack={() => setCurrentView('landing')} />;
  }

  return (
    <div className="min-h-screen bg-slate-50 font-sans text-slate-900">
      {/* Navigation */}
      <nav className="sticky top-0 z-50 bg-white/90 backdrop-blur-md border-b border-slate-100">
        <div className="container mx-auto px-4 md:px-6 h-20 flex items-center justify-between">
          <KinesisLogo />

          {/* Desktop Nav */}
          <div className="hidden md:flex items-center gap-8">
            <a href="#how-it-works" className="text-slate-600 hover:text-[#2998b3] font-medium transition-colors">How it Works</a>
            <a href="#features" className="text-slate-600 hover:text-[#2998b3] font-medium transition-colors">Features</a>
            <a href="#demo" className="text-slate-600 hover:text-[#2998b3] font-medium transition-colors">Demo</a>
            <button
              onClick={startTranslating}
              className="bg-[#14294a] hover:bg-[#203c69] text-white px-6 py-2.5 rounded-full font-medium transition-colors shadow-lg shadow-blue-900/20"
            >
              Try now
            </button>
          </div>

          {/* Mobile Menu Button */}
          <button className="md:hidden text-[#14294a]" onClick={toggleMenu}>
            {isMenuOpen ? <X size={28} /> : <Menu size={28} />}
          </button>
        </div>

        {/* Mobile Nav Dropdown */}
        {isMenuOpen && (
          <div className="md:hidden bg-white border-t border-slate-100 absolute w-full shadow-lg h-screen">
            <div className="flex flex-col p-4 gap-4">
              <a href="#how-it-works" className="text-lg font-medium text-slate-700 py-2" onClick={toggleMenu}>How it Works</a>
              <a href="#features" className="text-lg font-medium text-slate-700 py-2" onClick={toggleMenu}>Features</a>
              <a href="#demo" className="text-lg font-medium text-slate-700 py-2" onClick={toggleMenu}>Demo</a>
              <button
                onClick={startTranslating}
                className="bg-[#14294a] text-white py-3 rounded-lg font-bold w-full"
              >
                Try now
              </button>
            </div>
          </div>
        )}
      </nav>

      {/* SECTION 1 - HERO */}
      <section className="relative pt-16 pb-24 md:pt-24 md:pb-32 overflow-hidden">
        <div className="absolute top-0 right-0 w-1/2 h-full bg-[#f0f9fb] -z-10 rounded-bl-[100px] hidden md:block" />
        <div className="container mx-auto px-4 md:px-6">
          <div className="max-w-3xl">
            <div className="inline-.flex items-center gap-2 px-3 py-1 rounded-full bg-[#e6f4f7] text-[#2998b3] font-semibold text-sm mb-6">
              <span className="w-2 h-2 rounded-full bg-[#2998b3] animate-pulse"></span>
              Now live in Da Nang
            </div>
            <h1 className="text-4xl md:text-6xl font-extrabold text-[#14294a] leading-tight mb-6">
              A Da Nang where every voice is heard, every sign is understood, and no one is left behind.
            </h1>
            <p className="text-lg md:text-xl text-slate-600 mb-10 max-w-xl leading-relaxed">
              Real-time & two-way sign–voice translation by Kinesis.
            </p>
            <div className="flex flex-col sm:flex-row gap-4">
              <button
                onClick={() => document.getElementById('footer')?.scrollIntoView({ behavior: 'smooth' })}
                className="bg-[#2998b3] hover:bg-[#23869e] text-white px-8 py-4 rounded-full font-bold text-lg transition-all transform hover:-translate-y-1 shadow-xl shadow-teal-500/20 flex items-center justify-center gap-2"
              >
                Contact
                <ChevronRight size={20} />
              </button>
              <button className="bg-white hover:bg-slate-50 text-[#14294a] border-2 border-[#14294a]/10 px-8 py-4 rounded-full font-bold text-lg transition-colors flex items-center justify-center gap-2">
                <Play size={20} className="fill-current" />
                Watch Demo
              </button>
            </div>
          </div>
        </div>
      </section>

      {/* SECTION 2 - HOW IT WORKS */}
      <section id="how-it-works" className="py-20 bg-white">
        <div className="container mx-auto px-4 md:px-6">
.
          <div className="text-center mb-16">
            <h2 className="text-[#2998b3] font-bold tracking-wider uppercase mb-2 text-sm">Simple Process</h2>
            <SectionHeading>How Kinesis Works</SectionHeading>
          </div>

          <div className="grid md:grid-cols-3 gap-8 max-w-5xl mx-auto">
            <StepCard
              number="01"
              icon={Smartphone}
              color="bg-[#2998b3]"
              title="Open Kinesis"
              description="Open the Kinesis website in your browser whenever you need to communicate."
            />
            <StepCard
              number="02"
              icon={MessageSquare}
              color="bg-[#14294a]"
              title="Choose Mode"
              description="Pick Sign → Voice or Voice → Text based on your needs."
            />
            <StepCard
              number="03"
              icon={Hand}
              color="bg-[#e05263]"
              title="Start Communicating"
              description="Kinesis translates instantly in real time right on the website."
            />
          </div>
        </div>
      </section>

      {/* SECTION 3 - FEATURES */}
      <section id="features" className="py-20 bg-[#f8fafc]">
        <div className="container mx-auto px-4 md:px-6">
          <div className="text-center mb-16">
            <h2 className="text-[#2998b3] font-bold tracking-wider uppercase mb-2 text-sm">Key Capabilities</h2>
            <SectionHeading>Designed for Connection</SectionHeading>
          </div>

          <div className="grid md:grid-cols-3 gap-8">
            <FeatureCard
              icon={Globe}
              title="Bi-Directional Translation"
              description="Kinesis offers real-time, bi-directional translation. Using your device’s camera, it recognizes Vietnamese Sign Language and converts it to spoken Vietnamese. It also listens to spoken Vietnamese—including Da Nang dialect accent—and turns it into text."
            />
            <FeatureCard
              icon={Heart}
              title="Vietnamese-Friendly Experience"
              description="Designed specifically for Việt Nam, supporting local Da Nang sign styles and accents for culturally accurate translations."
            />
            <FeatureCard
              icon={Camera}
              title="No Extra Devices"
              description="No special equipment, gloves, or sensors needed. All you need is a camera-enabled device and a browser to communicate with confidence anywhere."
            />
          </div>
        </div>
      </section>

      {/* SECTION 4 - DEMO VIDEO */}
      <section id="demo" className="py-20 bg-[#14294a] text-white">
        <div className="container mx-auto px-4 md:px-6 text-center">
          <h2 className="text-3xl md:text-4xl font-bold mb-6">See Kinesis in Action</h2>
          <p className="text-slate-300 mb-10 text-lg">See how Kinesis translates Vietnamese Sign Language instantly in your browser.</p>

          <div className="max-w-4xl mx-auto bg-black/30 rounded-2xl overflow-hidden aspect-video relative group cursor-pointer shadow-2xl border border-white/10">
            {/* Placeholder for Video */}
            <div className="absolute inset-0 flex items-center justify-center bg-gradient-to-tr from-[#14294a] to-[#2998b3] opacity-80 group-hover:opacity-90 transition-opacity">
              <div className="w-20 h-20 bg-white rounded-full flex items-center justify-center pl-2 shadow-lg group-hover:scale-110 transition-transform duration-300">
                <Play size={32} className="text-[#14294a] fill-current" />
              </div>
            </div>
            <div className="absolute bottom-0 left-0 right-0 p-6 bg-gradient-to-t from-black/80 to-transparent">
              <span className="text-sm font-medium text-white/90 uppercase tracking-wider">Demo Preview</span>
            </div>
          </div>
        </div>
      </section>

      {/* SECTION 5 - COMMUNITY / TESTIMONIAL */}
      <section className="py-20 bg-white">
        <div className="container mx-auto px-4 md:px-6">
          <div className="max-w-4xl mx-auto text-center">
            <div className="mb-8 flex justify-center">
              <div className="p-3 bg-[#e6f4f7] rounded-full text-[#2998b3]">
                <MessageSquare size={32} />
              </div>
            </div>
            <blockquote className="text-3xl md:text-4xl font-medium text-[#14294a] leading-tight mb-8">
              "Every time I use Kinesis, I feel the distance between me and hearing people slowly disappear."
            </blockquote>
            <div className="flex items-center justify-center gap-4">
              <div className="w-12 h-12 bg-slate-100 rounded-full flex items-center justify-center text-slate-400">
                 <User size={24} />
              </div>
              <div className="text-left">
                <div className="font-bold text-[#14294a]">Tùng Trần</div>
                <div className="text-slate-500 text-sm">a deaf user @ Da Nang</div>
              </div>
            </div>
            <p className="mt-8 text-slate-400 font-medium tracking-wide uppercase text-sm">Real stories. Real voices. Real connection.</p>
          </div>
        </div>
      </section>

      {/* SECTION 6 - FOOTER */}
      <footer id="footer" className="bg-[#f1f5f9] pt-16 pb-8 border-t border-slate-200">
        <div className="container mx-auto px-4 md:px-6">
          <div className="grid md:grid-cols-4 gap-12 mb-12">
            <div className="col-span-1 md:col-span-2">
              <KinesisLogo />
              <p className="mt-6 text-slate-600 max-w-sm">
                Bridging the gap between sign language and spoken Vietnamese, ensuring no one is left behind.
              </p>
            </div>
            <div>
              <h4 className="font-bold text-[#14294a] mb-4">Platform</h4>
              <ul className="space-y-3">
                <li><a href="#" className="text-slate-600 hover:text-[#2998b3] transition-colors">How it Works</a></li>
                <li><a href="#" className="text-slate-600 hover:text-[#2998b3] transition-colors">Features</a></li>
                <li><a href="#" className="text-slate-600 hover:text-[#2998b3] transition-colors">Live Demo</a></li>
              </ul>
            </div>
            <div>
              <h4 className="font-bold text-[#14294a] mb-4">Legal & Support</h4>
              <ul className="space-y-3">
                <li><a href="#" className="text-slate-600 hover:text-[#2998b3] transition-colors">Contact Support</a></li>
                <li><a href="#" className="text-slate-600 hover:text-[#2998b3] transition-colors">Privacy Policy</a></li>
                <li><a href="#" className="text-slate-600 hover:text-[#2998b3] transition-colors">Terms of Use</a></li>
                <li><a href="#" className="text-slate-600 hover:text-[#2998b3] transition-colors flex items-center gap-2">
                  <Accessibility size={16} />
                  Accessibility Statement
                </a></li>
              </ul>
            </div>
          </div>
          <div className="border-t border-slate-200 pt-8 flex flex-col md:flex-row justify-between items-center gap-4 text-slate-500 text-sm">
            <p>&copy; {new Date().getFullYear()} Kinesis. All rights reserved.</p>
            <p>Made with ❤️ in Đà Nẵng, Vietnam.</p>
          </div>
        </div>
      </footer>
    </div>
  );
}
