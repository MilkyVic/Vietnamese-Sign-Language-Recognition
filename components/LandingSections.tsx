import React from 'react';
import { Smartphone, Sliders, Hand, Video, Mic, Globe, Zap, Heart } from 'lucide-react';

// --- Section 2: How It Works ---
export const HowItWorks: React.FC = () => {
  const steps = [
    {
      icon: <Smartphone size={32} className="text-primary-600" />,
      emoji: "📱",
      title: "Open Kinesis",
      text: "Open the Kinesis website in your browser whenever you need to communicate."
    },
    {
      icon: <Sliders size={32} className="text-primary-600" />,
      emoji: "🎛️",
      title: "Choose translation mode",
      text: "Pick Sign → Voice or Voice → Text."
    },
    {
      icon: <Hand size={32} className="text-primary-600" />,
      emoji: "✋🎤",
      title: "Start signing or speaking",
      text: "Kinesis translates instantly in real time right on the website."
    }
  ];

  return (
    <section id="how-it-works" className="py-20 bg-slate-50">
      <div className="container mx-auto px-6 lg:px-12 text-center">
        <h2 className="text-3xl md:text-4xl font-bold text-slate-900 mb-16">How it works</h2>
        
        <div className="grid grid-cols-1 md:grid-cols-3 gap-10 relative">
          {/* Connector Line (Desktop) */}
          <div className="hidden md:block absolute top-12 left-[16%] right-[16%] h-0.5 bg-slate-200 -z-0"></div>

          {steps.map((step, idx) => (
            <div key={idx} className="flex flex-col items-center relative z-10">
              <div className="w-24 h-24 bg-white rounded-2xl shadow-md border border-slate-100 flex items-center justify-center text-4xl mb-6">
                {step.emoji}
              </div>
              <h3 className="text-xl font-bold text-slate-900 mb-3">{step.title}</h3>
              <p className="text-slate-600 leading-relaxed max-w-xs">{step.text}</p>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
};

// --- Section 3: Features ---
export const Features: React.FC = () => {
  const features = [
    {
      icon: <ArrowIcon />,
      title: "Bi-Directional Translation",
      text: "Kinesis offers real-time, bi-directional translation. Using your device’s camera, it recognizes Vietnamese Sign Language and converts it to spoken Vietnamese. It also listens to spoken Vietnamese—including Da Nang dialect accent—and turns it into text."
    },
    {
      icon: <Globe size={28} />,
      title: "Vietnamese-Friendly Experience",
      text: "Designed specifically for Việt Nam, supporting local Da Nang sign styles and accents for culturally accurate translations."
    },
    {
      icon: <Zap size={28} />,
      title: "No Extra Devices Needed",
      text: "Kinesis requires no special equipment — no gloves, no sensors, no extra hardware. All you need is a camera-enabled device and a browser. Simply open the website and communicate with confidence, anytime, anywhere."
    }
  ];

  return (
    <section id="features" className="py-24 bg-white">
      <div className="container mx-auto px-6 lg:px-12">
        <div className="text-center mb-16">
           <span className="text-primary-600 font-bold uppercase tracking-wider text-sm mb-2 block">Powerful Capabilities</span>
           <h2 className="text-3xl md:text-4xl font-bold text-slate-900">Features</h2>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
          {features.map((feature, idx) => (
            <div key={idx} className="bg-slate-50 rounded-3xl p-8 hover:shadow-lg transition-shadow duration-300 border border-slate-100">
              <div className="w-14 h-14 bg-white rounded-xl shadow-sm flex items-center justify-center text-primary-600 mb-6">
                {feature.icon}
              </div>
              <h3 className="text-xl font-bold text-slate-900 mb-4">{feature.title}</h3>
              <p className="text-slate-600 leading-relaxed text-sm md:text-base">{feature.text}</p>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
};

const ArrowIcon = () => (
    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <path d="M7 10h10" /><path d="M7 10l3-3" /><path d="M7 10l3 3" />
        <path d="M17 14H7" /><path d="M17 14l-3-3" /><path d="M17 14l-3 3" />
    </svg>
)

// --- Section 4: Demo Video ---
export const DemoVideo: React.FC = () => {
  return (
    <section id="demo" className="py-20 bg-slate-900 text-white text-center">
       <div className="container mx-auto px-6 lg:px-12">
         <h2 className="text-3xl md:text-4xl font-bold mb-10">Watch Kinesis in action</h2>
         
         <div className="max-w-4xl mx-auto bg-slate-800 rounded-3xl overflow-hidden shadow-2xl aspect-video relative flex items-center justify-center group cursor-pointer border border-slate-700">
           {/* Placeholder for Video */}
           <div className="absolute inset-0 bg-gradient-to-tr from-primary-900/40 to-transparent"></div>
           <div className="z-10 flex flex-col items-center">
             <div className="w-20 h-20 bg-white/10 backdrop-blur-sm rounded-full flex items-center justify-center mb-4 group-hover:scale-110 transition-transform duration-300 border border-white/20">
               <div className="w-0 h-0 border-t-[12px] border-t-transparent border-l-[20px] border-l-white border-b-[12px] border-b-transparent ml-1"></div>
             </div>
             <p className="font-semibold tracking-wide uppercase text-sm opacity-80">Play Demo</p>
           </div>
           <img 
              src="https://picsum.photos/1200/675?grayscale&blur=2" 
              alt="Video Thumbnail" 
              className="absolute inset-0 w-full h-full object-cover opacity-30 -z-0"
           />
         </div>
         
         <p className="mt-8 text-slate-400 text-lg max-w-2xl mx-auto">
           See how Kinesis translates Vietnamese Sign Language instantly in your browser.
         </p>
       </div>
    </section>
  );
};

// --- Section 5: Community ---
export const Community: React.FC = () => {
  return (
    <section id="community" className="py-24 bg-primary-50">
      <div className="container mx-auto px-6 lg:px-12 text-center">
        <h2 className="text-3xl font-bold text-slate-900 mb-12">Community voices</h2>
        
        <div className="max-w-3xl mx-auto">
          <blockquote className="text-2xl md:text-3xl font-medium text-slate-800 leading-relaxed mb-10">
            “Every time I use Kinesis, I feel the distance between me and hearing people slowly disappear.”
          </blockquote>
          
          <div className="flex flex-col items-center">
            {/* Avatar Placeholder */}
            <div className="w-16 h-16 rounded-full bg-slate-300 mb-4 overflow-hidden relative">
              <svg className="absolute w-full h-full text-slate-400" fill="currentColor" viewBox="0 0 24 24">
                  <path d="M24 20.993V24H0v-2.996A14.977 14.977 0 0112.004 15c4.904 0 9.26 2.354 11.996 5.993zM16.002 8.999a4 4 0 11-8 0 4 4 0 018 0z" />
              </svg>
            </div>
            
            <cite className="not-italic">
              <div className="font-bold text-slate-900 text-lg">Tùng Trần</div>
              <div className="text-slate-500">a deaf user @ Da Nang</div>
            </cite>
          </div>

          <div className="mt-12 pt-12 border-t border-primary-200">
            <p className="text-primary-800 font-semibold tracking-wide">Real stories. Real voices. Real connection.</p>
          </div>
        </div>
      </div>
    </section>
  );
};

// --- Footer ---
export const Footer: React.FC = () => {
  const contactEmail = "tam.nguyen171204@gmail.com";
  
  return (
    <footer className="bg-white border-t border-slate-100 py-12">
      <div className="container mx-auto px-6 lg:px-12 flex flex-col md:flex-row justify-between items-center gap-6">
        
        <div className="flex items-center gap-2">
           <div className="w-8 h-8 rounded-full bg-slate-900 text-white flex items-center justify-center font-bold text-sm">K</div>
           <span className="font-bold text-slate-900">Kinesis</span>
        </div>

        <nav className="flex flex-wrap justify-center gap-x-8 gap-y-4">
          <a href={`mailto:${contactEmail}`} className="text-slate-500 hover:text-primary-600 text-sm transition-colors">Contact</a>
          <a href="#" className="text-slate-500 hover:text-primary-600 text-sm transition-colors">Support</a>
          <a href="#" className="text-slate-500 hover:text-primary-600 text-sm transition-colors">Accessibility Statement</a>
          <a href="#" className="text-slate-500 hover:text-primary-600 text-sm transition-colors">Privacy Policy</a>
          <a href="#" className="text-slate-500 hover:text-primary-600 text-sm transition-colors">Terms of Use</a>
        </nav>

        <div className="text-slate-400 text-sm">
          © {new Date().getFullYear()} Kinesis. All rights reserved.
        </div>
      </div>
    </footer>
  );
};