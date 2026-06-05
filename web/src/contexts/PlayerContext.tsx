import { createContext, useContext, useState, useCallback, type ReactNode } from "react";
import type { TrackOut } from "../lib/types";

interface PlayerState {
  playlist: TrackOut[];
  currentIndex: number; // -1 = nothing loaded
  playing: boolean;
}

interface PlayerContextValue extends PlayerState {
  load: (playlist: TrackOut[], index: number) => void;
  setPlaying: (p: boolean) => void;
  next: () => void;
  prev: () => void;
  current: TrackOut | null;
}

const PlayerContext = createContext<PlayerContextValue | null>(null);

export function PlayerProvider({ children }: { children: ReactNode }) {
  const [state, setState] = useState<PlayerState>({ playlist: [], currentIndex: -1, playing: false });

  const load = useCallback((playlist: TrackOut[], index: number) => {
    setState({ playlist, currentIndex: index, playing: true });
  }, []);
  const setPlaying = useCallback((p: boolean) => setState((s) => ({ ...s, playing: p })), []);
  const next = useCallback(() => setState((s) => {
    if (s.playlist.length === 0) return s;
    return { ...s, currentIndex: (s.currentIndex + 1) % s.playlist.length, playing: true };
  }), []);
  const prev = useCallback(() => setState((s) => {
    if (s.playlist.length === 0) return s;
    return { ...s, currentIndex: (s.currentIndex - 1 + s.playlist.length) % s.playlist.length, playing: true };
  }), []);

  const current = state.currentIndex >= 0 ? state.playlist[state.currentIndex] ?? null : null;

  return (
    <PlayerContext.Provider value={{ ...state, load, setPlaying, next, prev, current }}>
      {children}
    </PlayerContext.Provider>
  );
}

export function usePlayer(): PlayerContextValue {
  const ctx = useContext(PlayerContext);
  if (!ctx) throw new Error("usePlayer must be used within PlayerProvider");
  return ctx;
}
