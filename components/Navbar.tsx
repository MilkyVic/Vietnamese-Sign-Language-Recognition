import React, { useState, useEffect } from 'react';
import { Menu, X } from 'lucide-react';

export const Navbar: React.FC = () => {
  const [isScrolled, setIsScrolled] = useState(false);
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const contactEmail = "tam.nguyen171204@gmail.com";

  useEffect(() => {
    const handleScroll = () => {
      setIsScrolled(window.scrollY > 10);
    };
    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  const navLinks = [
    { name: 'How it works', href: '#how-it-works' },
    { name: 'Features', href: '#features' },
    { name: 'Demo', href: '#demo' },
    { name: 'Community', href: '#community' },
  ];

  return (
    <header 
      className={`fixed top-0 left-0 right-0 z-50 transition-all duration-300 ${
        isScrolled || mobileMenuOpen ? 'bg-white/90 backdrop-blur-md shadow-sm py-4' : 'bg-transparent py-6'
      }`}
    >
      <div className="container mx-auto px-6 lg:px-12 flex items-center justify-between">
        
        {/* Brand / Logo */}
        <div className="flex items-center gap-3 md:gap-4">
          <div className="w-10 h-10 md:w-12 md:h-12 rounded-full bg-primary-600 flex items-center justify-center shadow-lg shadow-primary-500/20 text-white overflow-hidden p-1.5 md:p-2">
            {/* Custom Logo SVG: Two Interlocking Hands */}
            <svg viewBox="0 0 100 100" fill="none" stroke="currentColor" strokeWidth="6" strokeLinecap="round" strokeLinejoin="round" className="w-full h-full text-white">
              {/* Left Hand: F-shape/OK sign */}
              <circle cx="42" cy="55" r="14" />
              <path d="M32 45 L28 25" />
              <path d="M22 55 L10 45" />
              <path d="M18 65 L8 75" />
              
              {/* Right Hand: F-shape/OK sign (mirrored & interlocking) */}
              <circle cx="58" cy="55" r="14" />
              <path d="M68 45 L72 25" />
              <path d="M78 55 L90 45" />
              <path d="M82 65 L92 75" />
            </svg>
          </div>
          <div className="flex flex-col">
            <span className="font-bold text-lg md:text-xl uppercase tracking-wide text-slate-900 leading-none">Kinesis</span>
            <span className="text-[10px] md:text-xs font-medium text-slate-500 uppercase tracking-wider">Sign–Voice Translator</span>
          </div>
        </div>

        {/* Desktop Nav */}
        <div className="hidden md:flex items-center space-x-8">
          <nav className="flex items-center space-x-6">
            {navLinks.map((link) => (
              <a 
                key={link.name} 
                href={link.href} 
                className="text-sm font-medium text-slate-600 hover:text-primary-600 transition-colors"
              >
                {link.name}
              </a>
            ))}
          </nav>
          <a 
            href={`mailto:${contactEmail}`}
            className="px-5 py-2.5 bg-slate-900 text-white text-sm font-semibold rounded-full hover:bg-slate-800 transition-colors"
          >
            Contact
          </a>
        </div>

        {/* Mobile Toggle */}
        <div className="md:hidden">
          <button onClick={() => setMobileMenuOpen(!mobileMenuOpen)} className="text-slate-900 p-2">
            {mobileMenuOpen ? <X size={24} /> : <Menu size={24} />}
          </button>
        </div>
      </div>

      {/* Mobile Menu */}
      {mobileMenuOpen && (
        <div className="md:hidden bg-white border-t border-slate-100 absolute w-full left-0 top-full shadow-xl py-6 px-6 flex flex-col space-y-4">
           {navLinks.map((link) => (
              <a 
                key={link.name} 
                href={link.href}
                onClick={() => setMobileMenuOpen(false)} 
                className="text-lg font-medium text-slate-800 py-2 border-b border-slate-50"
              >
                {link.name}
              </a>
            ))}
            <a 
              href={`mailto:${contactEmail}`}
              className="w-full mt-4 px-6 py-3 bg-slate-900 text-white text-base font-bold rounded-xl text-center block"
            >
              Contact
            </a>
        </div>
      )}
    </header>
  );
};