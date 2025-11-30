import React from 'react';
import { Navbar } from './components/Navbar';
import { Hero } from './components/Hero';
import { HowItWorks, Features, DemoVideo, Community, Footer } from './components/LandingSections';

const App: React.FC = () => {
  return (
    <div className="min-h-screen bg-white">
      <Navbar />
      <main>
        <Hero />
        <HowItWorks />
        <Features />
        <DemoVideo />
        <Community />
      </main>
      <Footer />
    </div>
  );
};

export default App;
