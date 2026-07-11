import { useEffect, useRef, useState } from "react";
import { usePlayer } from "../contexts/PlayerContext";

const fmtTime = (s: number) => {
  if (!Number.isFinite(s)) return "0:00";
  const m = Math.floor(s / 60);
  const sec = Math.floor(s % 60);
  return `${m}:${sec.toString().padStart(2, "0")}`;
};

export function MiniPlayer() {
  const { current, playing, setPlaying, next, prev } = usePlayer();
  const audioRef = useRef<HTMLAudioElement>(null);
  const [elapsed, setElapsed] = useState(0);
  const [duration, setDuration] = useState(0);

  // Load source when the current track changes.
  useEffect(() => {
    const audio = audioRef.current;
    if (!audio || !current) return;
    audio.src = `/api/audio/${encodeURIComponent(current.rating_key ?? "")}`;
    audio.load();
    if (playing) audio.play().catch(() => setPlaying(false));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [current?.rating_key]);

  // React to play/pause state changes.
  useEffect(() => {
    const audio = audioRef.current;
    if (!audio || !current) return;
    if (playing) audio.play().catch(() => setPlaying(false));
    else audio.pause();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [playing]);

  if (!current) return null;

  return (
    <div
      data-testid="mini-player"
      className="fixed bottom-16 lg:bottom-4 right-4 z-50 flex items-center gap-3 bg-panel border border-accent rounded-lg px-4 py-2 shadow-2xl min-w-[300px]"
    >
      <audio
        ref={audioRef}
        onTimeUpdate={(e) => setElapsed(e.currentTarget.currentTime)}
        onLoadedMetadata={(e) => setDuration(e.currentTarget.duration)}
        onEnded={next}
        onPlay={() => setPlaying(true)}
        onPause={() => setPlaying(false)}
      />
      <button
        onClick={prev}
        className="text-faint hover:text-text text-sm inline-flex items-center justify-center pointer-coarse:min-w-11 pointer-coarse:min-h-11"
        title="Previous"
      >
        ⏮
      </button>
      <button
        onClick={() => setPlaying(!playing)}
        className="bg-accent text-bg font-bold text-sm px-2 py-1 rounded inline-flex items-center justify-center pointer-coarse:min-w-11 pointer-coarse:min-h-11"
        title={playing ? "Pause" : "Play"}
      >
        {playing ? "❚❚" : "▶"}
      </button>
      <button
        onClick={next}
        className="text-faint hover:text-text text-sm inline-flex items-center justify-center pointer-coarse:min-w-11 pointer-coarse:min-h-11"
        title="Next"
      >
        ⏭
      </button>
      <div className="flex-1 min-w-0">
        <div className="text-text text-[10px] truncate">{current.title} — {current.artist}</div>
        {/* Padded hit area around the 2px visual track (discipline T1). */}
        <div
          data-testid="seek-bar"
          className="py-2 cursor-pointer"
          onClick={(e) => {
            const audio = audioRef.current;
            if (!audio || !duration) return;
            const rect = e.currentTarget.getBoundingClientRect();
            audio.currentTime = ((e.clientX - rect.left) / rect.width) * duration;
          }}
        >
          <div className="h-0.5 bg-border rounded">
            <div className="h-full bg-accent rounded" style={{ width: `${duration ? (elapsed / duration) * 100 : 0}%` }} />
          </div>
        </div>
      </div>
      <span className="text-faint text-[10px] font-mono whitespace-nowrap">
        {fmtTime(elapsed)} / {fmtTime(duration)}
      </span>
    </div>
  );
}
