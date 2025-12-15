import React, { useEffect, useState } from 'react';
import { Navbar } from './components/Navbar';
import { Hero } from './components/Hero';
import { HowItWorks, Features, DemoVideo, Community, Footer } from './components/LandingSections';
import LearningLibraryPage from './pages/LearningLibraryPage';

type Route = 'home' | 'learning';

const getRouteFromHash = (): Route => {
  return window.location.hash === '#/learning-library' ? 'learning' : 'home';
};

const App: React.FC = () => {
  const [route, setRoute] = useState<Route>(getRouteFromHash());

  useEffect(() => {
    const onHashChange = () => setRoute(getRouteFromHash());
    window.addEventListener('hashchange', onHashChange);
    return () => window.removeEventListener('hashchange', onHashChange);
  }, []);

  return (
    <div className="min-h-screen bg-white">
      <Navbar />
      <main>
        {route === 'home' ? (
          <>
            <Hero />
            <HowItWorks />
            <Features />
            <DemoVideo />
            <Community />
          </>
        ) : (
          <LearningLibraryPage />
        )}
      </main>
      <Footer />
    </div>
  );
};

export default App;
