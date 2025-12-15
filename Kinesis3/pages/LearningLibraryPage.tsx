import React, { useEffect, useState } from 'react';

type LibraryItem = {
  id: string | number;
  label: string;
  videoUrl: string;
  filename: string;
  posterUrl?: string;
};

const PAGE_SIZE = 12;

type VideoCardProps = {
  item: LibraryItem;
  isLoaded: boolean;
  onPlay: (id: string | number) => void;
};

const VideoCard: React.FC<VideoCardProps> = ({ item, isLoaded, onPlay }) => {
  const handlePlay = () => onPlay(item.id);
  return (
    <div className="rounded-2xl border border-slate-100 bg-slate-50 p-4 shadow-sm hover:shadow-md transition-shadow">
      <div className="aspect-video rounded-xl overflow-hidden bg-slate-200 flex items-center justify-center relative">
        {isLoaded ? (
          <video
            src={item.videoUrl}
            controls
            preload="metadata"
            poster={item.posterUrl || '/logo.png'}
            className="w-full h-full object-cover"
          />
        ) : (
          <button
            onClick={handlePlay}
            className="absolute inset-0 w-full h-full flex flex-col items-center justify-center bg-gradient-to-b from-slate-900/10 to-slate-900/30 text-white"
          >
            {item.posterUrl && (
              <img
                src={item.posterUrl}
                alt={item.label || 'Poster'}
                className="absolute inset-0 w-full h-full object-cover"
              />
            )}
            <div className="w-14 h-14 rounded-full bg-white/90 text-slate-900 flex items-center justify-center shadow-lg mb-3 relative z-10">
              <div className="w-0 h-0 border-t-[12px] border-t-transparent border-l-[20px] border-l-slate-900 border-b-[12px] border-b-transparent ml-1"></div>
            </div>
            <span className="font-semibold text-sm relative z-10 bg-black/30 px-3 py-1 rounded-full">
              Bam de tai & xem
            </span>
          </button>
        )}
      </div>
      <div className="mt-4 text-left">
        <p className="font-semibold text-slate-900 text-lg leading-snug">{item.label || 'Khong co tieu de'}</p>
      </div>
    </div>
  );
};

const LearningLibraryPage: React.FC = () => {
  const [items, setItems] = useState<LibraryItem[]>([]);
  const [status, setStatus] = useState<'idle' | 'loading' | 'error'>('loading');
  const [error, setError] = useState<string | null>(null);
  const [page, setPage] = useState(1);
  const [totalPages, setTotalPages] = useState(1);
  const [loadedIds, setLoadedIds] = useState<Record<string | number, boolean>>({});
  const [search, setSearch] = useState('');
  const [pendingSearch, setPendingSearch] = useState('');

  useEffect(() => {
    const load = async () => {
      try {
        const apiBase = import.meta.env.VITE_API_BASE || '';
        const params = new URLSearchParams({
          page: String(page),
          limit: String(PAGE_SIZE),
          q: pendingSearch,
        });
        const res = await fetch(`${apiBase}/learning-library?${params.toString()}`);
        if (!res.ok) {
          throw new Error(`Server returned ${res.status}`);
        }
        const data = await res.json();
        setItems(data.items || []);
        setPage(data.page || page);
        setTotalPages(data.total_pages || 1);
        setStatus('idle');
      } catch (err: any) {
        setError(err?.message || 'Khong the tai thu vien hoc tap');
        setStatus('error');
      }
    };
    load();
  }, [page, pendingSearch]);

  const playVideo = (id: string | number) => {
    setLoadedIds((prev) => ({ ...prev, [id]: true }));
  };

  // Debounce search input
  useEffect(() => {
    const handle = setTimeout(() => {
      setPendingSearch(search.trim());
      setPage(1);
    }, 300);
    return () => clearTimeout(handle);
  }, [search]);

  const canPrev = page > 1;
  const canNext = page < totalPages;

  const pageNumbers = (() => {
    const windowSize = 5;
    let start = Math.max(1, page - 2);
    let end = Math.min(totalPages, start + windowSize - 1);
    // ensure we always show up to windowSize pages if possible
    if (end - start + 1 < windowSize) {
      start = Math.max(1, end - windowSize + 1);
    }
    return Array.from({ length: end - start + 1 }, (_, i) => start + i);
  })();

  return (
    <section className="pt-28 pb-24 bg-white min-h-screen">
      <div className="container mx-auto px-6 lg:px-12">
        <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-6 mb-10">
          <div>
            <span className="text-primary-600 font-bold uppercase tracking-wider text-sm mb-2 block">
              Learning Library
            </span>
          </div>
          <div className="w-full md:w-80">
            <div className="relative">
              <input
                type="text"
                placeholder="Tim theo tieu de..."
                value={search}
                onChange={(e) => setSearch(e.target.value)}
                className="w-full pl-10 pr-4 py-2.5 rounded-xl border border-slate-200 focus:border-primary-500 focus:ring-2 focus:ring-primary-100 outline-none text-sm text-slate-700"
              />
              <svg
                className="w-5 h-5 text-slate-400 absolute left-3 top-1/2 -translate-y-1/2 pointer-events-none"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
              >
                <circle cx="11" cy="11" r="7" />
                <line x1="16.65" y1="16.65" x2="21" y2="21" />
              </svg>
            </div>
          </div>
        </div>

        {status === 'loading' && (
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            {[1, 2, 3].map((i) => (
              <div key={i} className="rounded-2xl border border-slate-100 bg-slate-50 p-4 animate-pulse">
                <div className="aspect-video rounded-xl bg-slate-200 mb-4" />
                <div className="h-4 bg-slate-200 rounded w-2/3 mb-2" />
                <div className="h-3 bg-slate-200 rounded w-1/3" />
              </div>
            ))}
          </div>
        )}

        {status === 'error' && (
          <div className="bg-red-50 border border-red-100 text-red-700 rounded-xl p-4">
            {error || 'Khong the tai video. Vui long thu lai sau.'}
          </div>
        )}

        {status === 'idle' && !items.length && (
          <div className="bg-slate-50 border border-slate-100 text-slate-600 rounded-xl p-6 text-center">
            Chua co video trong Learning Library.
          </div>
        )}

        {status === 'idle' && items.length > 0 && (
          <>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              {items.map((item) => (
                <VideoCard
                  key={item.id}
                  item={item}
                  isLoaded={!!loadedIds[item.id]}
                  onPlay={(id) => playVideo(id)}
                />
              ))}
            </div>

            <div className="mt-10 flex flex-col items-center gap-3">
              <div className="flex items-center gap-2">
                <span className="text-sm text-slate-600">Trang:</span>
                <input
                  type="number"
                  min={1}
                  max={totalPages}
                  value={page}
                  onChange={(e) => {
                    const next = Number(e.target.value) || 1;
                    if (next >= 1 && next <= totalPages) {
                      setPage(next);
                    }
                  }}
                  className="w-20 px-3 py-2 rounded-lg border border-slate-200 text-sm text-slate-700 focus:border-primary-500 focus:ring-2 focus:ring-primary-100 outline-none"
                />
                <span className="text-sm text-slate-500">/ {totalPages}</span>
              </div>

              <div className="flex flex-wrap items-center justify-center gap-2">
                <button
                  onClick={() => canPrev && setPage((p) => Math.max(1, p - 1))}
                  disabled={!canPrev}
                  className="px-4 py-2 rounded-lg border border-slate-200 text-slate-700 disabled:opacity-50 disabled:cursor-not-allowed hover:bg-slate-50"
                >
                  Trang truoc
                </button>
                {pageNumbers.map((num) => (
                  <button
                    key={num}
                    onClick={() => setPage(num)}
                    className={`min-w-[38px] px-3 py-2 rounded-lg border text-sm ${
                      num === page
                        ? 'bg-primary-600 text-white border-primary-600'
                        : 'border-slate-200 text-slate-700 hover:bg-slate-50'
                    }`}
                  >
                    {num}
                  </button>
                ))}
                <button
                  onClick={() => canNext && setPage((p) => p + 1)}
                  disabled={!canNext}
                  className="px-4 py-2 rounded-lg border border-slate-200 text-slate-700 disabled:opacity-50 disabled:cursor-not-allowed hover:bg-slate-50"
                >
                  Trang tiep
                </button>
              </div>
            </div>

            {/* Old controls removed */}
            {/* 
            <div className="mt-10 flex items-center justify-center gap-3">
              <button
                onClick={() => canPrev && setPage((p) => Math.max(1, p - 1))}
                disabled={!canPrev}
                className="px-4 py-2 rounded-lg border border-slate-200 text-slate-700 disabled:opacity-50 disabled:cursor-not-allowed hover:bg-slate-50"
              >
                Trang truoc
              </button>
              <span className="text-sm text-slate-600">
                Trang {page} / {totalPages}
              </span>
              <button
                onClick={() => canNext && setPage((p) => p + 1)}
                disabled={!canNext}
                className="px-4 py-2 rounded-lg border border-slate-200 text-slate-700 disabled:opacity-50 disabled:cursor-not-allowed hover:bg-slate-50"
              >
                Trang tiep
              </button>
            </div>
            */}
          </>
        )}
      </div>
    </section>
  );
};

export default LearningLibraryPage;
