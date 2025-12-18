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
  useEffect(() => {
    // Redirect to external dictionary
    if (typeof window !== 'undefined') {
      window.location.href = 'https://qipedc.moet.gov.vn/dictionary';
    }
  }, []);

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
    <section className="pt-28 pb-24 bg-white min-h-screen flex items-center">
      <div className="container mx-auto px-6 lg:px-12 text-center space-y-4">
        <p className="text-lg text-slate-700">
          Đang chuyển hướng tới Learning Hub...
        </p>
        <a
          href="https://qipedc.moet.gov.vn/dictionary"
          target="_blank"
          rel="noopener noreferrer"
          className="text-primary-600 underline"
        >
          Nhấn vào đây nếu không tự chuyển hướng
        </a>
      </div>
    </section>
  );
};

export default LearningLibraryPage;
