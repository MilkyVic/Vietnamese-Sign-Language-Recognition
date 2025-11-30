import React from 'react';
import { TranslationMock } from './TranslationMock';
import { ArrowRight, Globe, CheckCircle2 } from 'lucide-react';

export const Hero: React.FC = () => {
  const contactEmail = "tam.nguyen171204@gmail.com";
  
  const scrollToMock = () => {
    const element = document.getElementById('translator-interface');
    if (element) {
      element.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }
  };

  return (
    <section className="relative pt-32 pb-20 lg:pt-40 lg:pb-32 overflow-hidden bg-white">
      {/* Background Decor */}
      <div className="absolute top-0 right-0 w-1/2 h-full bg-slate-50/50 -skew-x-12 translate-x-1/4 z-0 pointer-events-none" />
      <div className="absolute top-20 left-10 w-64 h-64 bg-primary-100 rounded-full blur-3xl opacity-30 pointer-events-none" />

      <div className="container mx-auto px-6 lg:px-12 relative z-10">
        <div className="flex flex-col lg:flex-row items-center gap-16">
          
          {/* Left Column: Text & CTAs */}
          <div className="flex-1 text-center lg:text-left">
            <div className="inline-flex items-center space-x-2 bg-slate-100 text-slate-700 px-3 py-1 rounded-full text-[10px] md:text-xs font-bold uppercase tracking-wider mb-6">
              <span className="w-2 h-2 rounded-full bg-primary-500"></span>
              <span>Now live in Da Nang</span>
            </div>

            <h1 className="text-4xl md:text-5xl lg:text-6xl font-extrabold text-slate-900 tracking-tight leading-[1.1] mb-6">
              A Da Nang where <span className="text-primary-600">every voice</span> is heard, <span className="text-primary-600">every sign</span> is understood.
            </h1>

            <p className="text-lg md:text-xl text-slate-600 mb-8 max-w-2xl mx-auto lg:mx-0 leading-relaxed">
              Real-time & two-way sign–voice translation by Kinesis. Breaking barriers so no one is left behind.
            </p>

            <div className="flex flex-col sm:flex-row items-center justify-center lg:justify-start gap-4 mb-10">
              <button 
                onClick={scrollToMock}
                className="w-full sm:w-auto px-8 py-4 bg-primary-600 text-white font-bold rounded-full shadow-lg shadow-primary-600/20 hover:bg-primary-700 transition-all hover:-translate-y-0.5 active:translate-y-0 flex items-center justify-center gap-2"
              >
                Try now <ArrowRight size={18} />
              </button>
              <a 
                href={`mailto:${contactEmail}`}
                className="w-full sm:w-auto px-8 py-4 bg-white text-slate-700 font-bold rounded-full border border-slate-200 hover:bg-slate-50 transition-colors flex items-center justify-center"
              >
                Contact
              </a>
            </div>

            {/* Badges */}
            <div className="flex flex-wrap justify-center lg:justify-start gap-4 md:gap-8 text-sm text-slate-500 font-medium">
              <div className="flex items-center gap-2">
                <Globe size={16} className="text-primary-500" />
                <span>Works right in Kinesis</span>
              </div>
              <div className="flex items-center gap-2">
                <CheckCircle2 size={16} className="text-primary-500" />
                <span>Supports deaf community in Da Nang</span>
              </div>
            </div>
          </div>

          {/* Right Column: Mock Interface */}
          <div className="flex-1 w-full max-w-lg lg:max-w-xl">
             <div className="relative">
                {/* Decorative blob behind the card */}
                <div className="absolute -inset-4 bg-gradient-to-tr from-primary-200 to-slate-200 rounded-[2.5rem] opacity-40 blur-lg"></div>
                <TranslationMock />
             </div>
          </div>

        </div>
      </div>
    </section>
  );
};