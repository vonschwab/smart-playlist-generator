import { useEffect, useRef, useState } from "react";
import { api } from "../lib/api";
import type { AxisValue, GenerateRequestBody, Mode } from "../lib/types";

const AXES: { key: string; label: string; values: string[] }[] = [
  { key: "genre_mode", label: "genre", values: ["off", "discover", "dynamic", "narrow", "strict"] },
  { key: "sonic_mode", label: "sonic", values: ["off", "dynamic", "narrow", "strict"] },
  { key: "pace_mode", label: "pace", values: ["dynamic", "narrow", "strict"] },
];

export function GenerateControls(props: { onSubmit: (body: GenerateRequestBody) => void; busy: boolean }) {
  const [mode, setMode] = useState<Mode>("artist");
  const [seed, setSeed] = useState("");
  const [tracks, setTracks] = useState(30);
  const [axes, setAxes] = useState<Record<string, string>>({
    genre_mode: "dynamic",
    sonic_mode: "dynamic",
    pace_mode: "dynamic",
  });
  const [cohesion, setCohesion] = useState("dynamic");
  const [suggestions, setSuggestions] = useState<string[]>([]);
  const timer = useRef<number>();

  useEffect(() => {
    if (mode !== "artist" || seed.length < 2) {
      setSuggestions([]);
      return;
    }
    window.clearTimeout(timer.current);
    timer.current = window.setTimeout(async () => {
      setSuggestions(await api.autocomplete(seed).catch(() => []));
    }, 180);
  }, [seed, mode]);

  function submit() {
    const body: GenerateRequestBody = {
      mode,
      tracks,
      artist: mode === "artist" ? seed : undefined,
      genre: mode === "genre" ? seed : undefined,
      seed_tracks:
        mode === "seeds"
          ? seed
              .split(",")
              .map((s) => s.trim())
              .filter(Boolean)
          : undefined,
      genre_mode: axes.genre_mode as AxisValue,
      sonic_mode: axes.sonic_mode as AxisValue,
      pace_mode: axes.pace_mode as "strict" | "narrow" | "dynamic",
    };
    props.onSubmit(body);
  }

  return (
    <div className="flex flex-wrap items-center gap-2 px-3 py-2 bg-panel2 border-b border-border">
      <select
        value={mode}
        onChange={(e) => setMode(e.target.value as Mode)}
        className="bg-[#0c0e12] border border-border rounded text-xs text-text px-2 py-1.5"
      >
        <option value="artist">artist</option>
        <option value="seeds">seeds</option>
        <option value="genre">genre</option>
        <option value="history">history</option>
      </select>

      <div className="relative flex-1 min-w-[180px]">
        <input
          value={seed}
          onChange={(e) => setSeed(e.target.value)}
          placeholder="Acetone, Mazzy Star"
          className="w-full bg-[#0c0e12] border border-border rounded text-xs text-text px-2.5 py-1.5"
        />
        {suggestions.length > 0 && (
          <ul className="absolute z-10 mt-1 w-full bg-panel border border-border rounded shadow-xl max-h-48 overflow-auto">
            {suggestions.map((s) => (
              <li
                key={s}
                onClick={() => {
                  setSeed(s);
                  setSuggestions([]);
                }}
                className="px-2.5 py-1.5 text-xs text-text hover:bg-border cursor-pointer"
              >
                {s}
              </li>
            ))}
          </ul>
        )}
      </div>

      <input
        type="number"
        min={1}
        max={200}
        value={tracks}
        onChange={(e) => setTracks(Number(e.target.value))}
        className="w-16 bg-[#0c0e12] border border-border rounded text-xs text-text px-2 py-1.5 text-center"
      />

      <select
        value={cohesion}
        onChange={(e) => setCohesion(e.target.value)}
        title="cohesion mode"
        className="bg-[#0c0e12] border border-border rounded text-xs text-muted px-2 py-1.5"
      >
        {["strict", "narrow", "dynamic", "discover"].map((v) => (
          <option key={v} value={v}>
            cohesion · {v}
          </option>
        ))}
      </select>

      {AXES.map((a) => (
        <select
          key={a.key}
          value={axes[a.key]}
          onChange={(e) => setAxes({ ...axes, [a.key]: e.target.value })}
          className="bg-[#0c0e12] border border-border rounded text-xs text-muted px-2 py-1.5"
        >
          {a.values.map((v) => (
            <option key={v} value={v}>
              {a.label} · {v}
            </option>
          ))}
        </select>
      ))}

      <button
        onClick={submit}
        disabled={props.busy}
        className="bg-accent text-bg font-semibold text-xs px-3.5 py-1.5 rounded disabled:opacity-50"
      >
        {props.busy ? "Generating…" : "▸ Generate"}
      </button>
    </div>
  );
}
