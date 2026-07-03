import { useEffect, useRef, useState } from "react";
import { api } from "../lib/api";
import { useInfiniteSearch } from "../lib/useInfiniteSearch";
import { useLocalStorage } from "../lib/useLocalStorage";
import type { AxisValue, GenerateRequestBody, Mode } from "../lib/types";

// ── Constants ────────────────────────────────────────────────────────────────

const DIVERSITY_GAMMAS = [0.00, 0.02, 0.04, 0.06, 0.08, 0.08];
const DIVERSITY_LABELS = ["very low", "low", "normal", "high", "very high", "one each"];

// ── Shared style helpers ──────────────────────────────────────────────────────

const SEL = "bg-[#0c0e12] border border-[#23262d] rounded text-[11px] text-[#8b939d] py-[3px] pl-[5px] pr-[18px] appearance-none cursor-pointer disabled:opacity-30 disabled:cursor-default"
  + " [background-image:url(\"data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='8' height='5'%3E%3Cpath d='M0 0l4 5 4-5z' fill='%235b6470'/%3E%3C/svg%3E\")] [background-repeat:no-repeat] [background-position:right_5px_center]";

function Lbl({ title, children }: { title?: string; children: React.ReactNode }) {
  return (
    <span title={title} className="text-[9px] uppercase tracking-[.08em] text-[#5b6470] font-medium whitespace-nowrap select-none">
      {children}
    </span>
  );
}

// A bordered cell within a toolbar row
function Cell({ children, grow, push, className = "" }: {
  children: React.ReactNode;
  grow?: boolean;
  push?: boolean;
  className?: string;
}) {
  return (
    <div className={[
      "flex items-center gap-[6px] px-3 py-[6px]",
      "border-r border-[#1e2128] last:border-r-0",
      grow ? "flex-1 min-w-0" : "",
      push ? "ml-auto border-l border-r-0 !border-r-0" : "",
      className,
    ].join(" ")}>
      {children}
    </div>
  );
}

// ── Component ─────────────────────────────────────────────────────────────────

export function GenerateControls({
  mode,
  onModeChange,
  seedTrackIds,
  seedDisplays,
  onSubmit,
  busy,
  initialValues,
}: {
  mode: Mode;
  onModeChange: (m: Mode) => void;
  seedTrackIds: string[];
  // Display strings ("Title - Artist") parallel to seedTrackIds. The backend
  // requires seed_tracks (the fuzzy-match fallback) alongside seed_track_ids
  // (the exact-match optimization); sending IDs alone fails generation.
  seedDisplays: string[];
  onSubmit: (body: GenerateRequestBody) => void;
  busy: boolean;
  initialValues?: Partial<GenerateRequestBody>;
}) {
  const [seed, setSeed] = useLocalStorage("pg_seed_text", "");
  const [tracks, setTracks] = useLocalStorage("pg_tracks", 30);
  const [cohesion, setCohesion] = useLocalStorage("pg_cohesion", "dynamic");
  const [axes, setAxes] = useLocalStorage<Record<string, string>>("pg_axes", {
    genre_mode: "dynamic",
    sonic_mode: "dynamic",
    pace_mode: "dynamic",
  });

  // Row 3 — freshness + spacing
  const [recencyEnabled, setRecencyEnabled] = useLocalStorage("pg_recency_enabled", true);
  const [recencyDays, setRecencyDays] = useLocalStorage("pg_recency_days", 14);
  const [recencyPlays, setRecencyPlays] = useLocalStorage("pg_recency_plays", 1);
  const [excludeRecentSeeds, setExcludeRecentSeeds] = useLocalStorage("pg_exclude_recent_seeds", false);
  const [artistSpacing, setArtistSpacing] = useLocalStorage("pg_artist_spacing", "normal");
  const [diversityLevel, setDiversityLevel] = useLocalStorage("pg_diversity_level", 2);

  // Artist-mode Row 1 extras
  const [artistPresence, setArtistPresence] = useLocalStorage("pg_artist_presence", "medium");
  const [artistVariety, setArtistVariety] = useLocalStorage("pg_artist_variety", "balanced");
  const [includeCollabs, setIncludeCollabs] = useLocalStorage("pg_include_collabs", false);
  const [popularSeedsMode, setPopularSeedsMode] = useLocalStorage<"off" | "on" | "fire">("pg_popular_seeds_mode", "off");
  const [popularityMode, setPopularityMode] = useLocalStorage<"off" | "on" | "oops">("pg_popularity_mode", "off");
  const [seedEpoch, setSeedEpoch] = useState(0);

  // On a narrow *container* (<@md = 448px) Rows 2–3 + artist extras collapse
  // behind this toggle. At >=448px they are always shown regardless of state.
  const [showMore, setShowMore] = useState(false);

  // Autocomplete (artist mode) — bounded-page infinite scroll
  const artistSearch = useInfiniteSearch<string>({ fetchPage: api.autocomplete, pageSize: 30 });
  // Initialize from the persisted seed so a remount (e.g. mobile tab switch
  // unmounting this component) treats the stored name as already-selected and
  // does NOT reopen the autocomplete dropdown.
  const selectedRef = useRef<string | null>(seed);
  const dropdownRef = useRef<HTMLDivElement>(null);
  const listRef = useRef<HTMLUListElement>(null);

  // Tag steering (artist mode): the artist's published genres as selectable chips.
  const [artistTags, setArtistTags] = useState<
    { name: string; release_count: number; confidence: number }[]
  >([]);
  const [steeringTags, setSteeringTags] = useState<string[]>([]);
  const [tagsFetched, setTagsFetched] = useState(false);

  // Fetch chips whenever a confirmed artist is selected; reset both on change.
  useEffect(() => {
    setSteeringTags([]);
    setArtistTags([]);
    setTagsFetched(false);
    if (mode !== "artist" || !seed.trim() || seed !== selectedRef.current) return;
    let cancelled = false;
    api.artistGenres(seed)
      .then((r) => { if (!cancelled) { setArtistTags(r.genres); setTagsFetched(true); } })
      .catch(() => { /* chips are best-effort; generation works without them */ });
    return () => { cancelled = true; };
  }, [seed, mode]);

  function toggleSteeringTag(name: string) {
    setSteeringTags((prev) =>
      prev.includes(name) ? prev.filter((t) => t !== name)
        : prev.length >= 3 ? prev
        : [...prev, name],
    );
  }

  // Apply initialValues (re-run prefill) once per change
  useEffect(() => {
    if (!initialValues) return;
    if (initialValues.cohesion_mode) setCohesion(initialValues.cohesion_mode);
    if (initialValues.genre_mode || initialValues.sonic_mode || initialValues.pace_mode) {
      setAxes({
        genre_mode: initialValues.genre_mode ?? axes.genre_mode,
        sonic_mode: initialValues.sonic_mode ?? axes.sonic_mode,
        pace_mode: initialValues.pace_mode ?? axes.pace_mode,
      });
    }
    if (typeof initialValues.tracks === "number") setTracks(initialValues.tracks);
    if (typeof initialValues.artist === "string") setSeed(initialValues.artist);
    else if (typeof initialValues.genre === "string") setSeed(initialValues.genre);
    if (initialValues.artist_spacing) setArtistSpacing(initialValues.artist_spacing);
    if (initialValues.artist_presence) setArtistPresence(initialValues.artist_presence);
    if (initialValues.artist_variety) setArtistVariety(initialValues.artist_variety);
    if (typeof initialValues.include_collaborations === "boolean") setIncludeCollabs(initialValues.include_collaborations);
    if (typeof initialValues.recency_enabled === "boolean") setRecencyEnabled(initialValues.recency_enabled);
    if (typeof initialValues.recency_days === "number") setRecencyDays(initialValues.recency_days);
    if (typeof initialValues.recency_plays_threshold === "number") setRecencyPlays(initialValues.recency_plays_threshold);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [initialValues]);

  // Close autocomplete on outside click
  useEffect(() => {
    function onOutside(e: MouseEvent) {
      if (dropdownRef.current && !dropdownRef.current.contains(e.target as Node)) {
        artistSearch.reset();
      }
    }
    document.addEventListener("mousedown", onOutside);
    return () => document.removeEventListener("mousedown", onOutside);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Drive the artist autocomplete from the seed input. Skip one cycle after a
  // selection so picking a name doesn't immediately re-open the dropdown.
  useEffect(() => {
    if (mode !== "artist") { artistSearch.reset(); selectedRef.current = null; return; }
    if (seed === selectedRef.current) return; // just selected this name — don't reopen the dropdown
    artistSearch.setQuery(seed);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [seed, mode]);

  function submit(epoch: number = seedEpoch) {
    const body: GenerateRequestBody = {
      mode,
      tracks,
      artist: mode === "artist" ? seed : undefined,
      genre: mode === "genre" ? seed : undefined,
      seed_tracks: mode === "seeds" ? seedDisplays : undefined,
      seed_track_ids: mode === "seeds" ? seedTrackIds : undefined,
      cohesion_mode: cohesion as "strict" | "narrow" | "dynamic" | "discover",
      genre_mode: axes.genre_mode as AxisValue,
      sonic_mode: axes.sonic_mode as AxisValue,
      pace_mode: axes.pace_mode as "strict" | "narrow" | "dynamic" | "off",
      recency_enabled: recencyEnabled,
      recency_days: recencyDays,
      recency_plays_threshold: recencyPlays,
      exclude_seed_tracks_from_recency: excludeRecentSeeds,
      artist_spacing: artistSpacing,
      diversity_gamma: DIVERSITY_GAMMAS[diversityLevel],
      artist_diversity_mode: diversityLevel === DIVERSITY_GAMMAS.length - 1 ? "one_per_artist" : "weighted",
      artist_presence: artistPresence,
      artist_variety: artistVariety,
      include_collaborations: includeCollabs,
      popular_seeds_mode: popularityMode === "oops" ? "fire" : popularSeedsMode,
      popularity_mode: popularityMode,
      seed_epoch: epoch,
      steering_tags: mode === "artist" && steeringTags.length ? steeringTags : undefined,
    };
    onSubmit(body);
  }

  return (
    <div className="@container border border-[#23262d] rounded-none border-x-0 border-t-0 overflow-hidden">

      {/* ── ROW 1: mode-specific ──────────────────────────────────────────── */}
      <div className="flex flex-wrap items-center bg-[#16181d] border-b border-[#1e2128]">

        {/* Mode selector */}
        <Cell>
          <select
            value={mode}
            onChange={(e) => { onModeChange(e.target.value as Mode); setSeed(""); artistSearch.reset(); }}
            className={SEL}
            title="Generation mode"
          >
            <option value="artist">ARTIST</option>
            <option value="seeds">SEEDS</option>
            <option value="genre">GENRE</option>
            <option value="history">HISTORY</option>
          </select>
        </Cell>

        {/* Text input: artist or genre mode */}
        {(mode === "artist" || mode === "genre") && (
          <Cell className="flex-1 min-w-[220px]">
            <div ref={dropdownRef} className="relative w-full">
              <input
                data-testid="seed-input"
                value={seed}
                onChange={(e) => { selectedRef.current = null; setSeed(e.target.value); }}
                onKeyDown={(e) => e.key === "Enter" && submit()}
                placeholder={mode === "artist" ? "Artist name…" : "Genre…"}
                className="w-full bg-[#0c0e12] border border-[#23262d] rounded text-[11px] text-[#e6e9ec] px-2.5 py-[3px]"
              />
              {mode === "artist" && artistSearch.items.length > 0 && (
                <ul
                  ref={listRef}
                  onScroll={() => {
                    const el = listRef.current;
                    if (el && el.scrollHeight - el.scrollTop - el.clientHeight < 48) artistSearch.loadMore();
                  }}
                  className="absolute z-10 mt-1 w-full bg-[#16181d] border border-[#23262d] rounded shadow-xl max-h-48 overflow-auto"
                >
                  {artistSearch.items.map((s) => (
                    <li
                      key={s}
                      onClick={() => { selectedRef.current = s; setSeed(s); artistSearch.reset(); }}
                      className="px-2.5 py-1.5 text-[11px] text-[#e6e9ec] hover:bg-[#1e2229] cursor-pointer"
                    >
                      {s}
                    </li>
                  ))}
                  {(artistSearch.loading || artistSearch.hasMore) && (
                    <li className="px-2.5 py-1.5 text-[10px] text-[#5b6470]">
                      {artistSearch.loading ? "Loading…" : "Scroll for more"}
                    </li>
                  )}
                </ul>
              )}
            </div>
          </Cell>
        )}

        {/* Seeds mode: show count badge */}
        {mode === "seeds" && seedTrackIds.length > 0 && (
          <Cell grow>
            <span className="text-[10px] text-[#5b6470]">
              {seedTrackIds.length} seed{seedTrackIds.length !== 1 ? "s" : ""} — manage below ↓
            </span>
          </Cell>
        )}

        {/* Tracks */}
        <Cell>
          <Lbl title="Number of tracks to generate.">tracks</Lbl>
          <input
            type="number"
            min={1}
            max={200}
            value={tracks}
            onChange={(e) => setTracks(Number(e.target.value))}
            className="w-12 bg-[#0c0e12] border border-[#23262d] rounded text-[11px] text-[#e6e9ec] px-2 py-[3px] text-center [appearance:textfield] [&::-webkit-outer-spin-button]:appearance-none [&::-webkit-inner-spin-button]:appearance-none"
          />
        </Cell>

        {/* Artist-mode extras */}
        {mode === "artist" && (
          <>
            <Cell>
              <Lbl title="Target share of the playlist from the seed artist. Very Low ≈ 5%, Low ≈ 10%, Medium ≈ 12%, High ≈ 20%, Very High ≈ 33%.">
                artist presence
              </Lbl>
              <select
                value={artistPresence}
                onChange={(e) => setArtistPresence(e.target.value)}
                className={SEL}
                title="Target share of the playlist from the seed artist. Very Low ≈ 5%, Low ≈ 10%, Medium ≈ 12%, High ≈ 20%, Very High ≈ 33%."
              >
                <option value="very_low">very low</option>
                <option value="low">low</option>
                <option value="medium">medium</option>
                <option value="high">high</option>
                <option value="very_high">very high</option>
              </select>
            </Cell>
            <Cell>
              <Lbl title="How much of the seed artist's stylistic range to draw from. Some artists span many styles across their catalog — Focused draws from one corner, Sprawling draws from across the full range.">
                style spread
              </Lbl>
              <select
                value={artistVariety}
                onChange={(e) => setArtistVariety(e.target.value)}
                className={SEL}
                title="How much of the seed artist's stylistic range to draw from. Some artists span many styles across their catalog — Focused draws from one corner, Sprawling draws from across the full range."
              >
                <option value="focused">focused</option>
                <option value="balanced">balanced</option>
                <option value="sprawling">sprawling</option>
              </select>
            </Cell>
            <Cell>
              <label
                className="flex items-center gap-1.5 cursor-pointer select-none"
                title="Include collaboration tracks (e.g. 'Miles Davis & John Coltrane', 'Bill Evans Trio') in the seed pool alongside solo tracks."
              >
                <input
                  type="checkbox"
                  checked={includeCollabs}
                  onChange={(e) => setIncludeCollabs(e.target.checked)}
                  className="accent-[#5eead4] cursor-pointer"
                />
                <Lbl>include collaborations</Lbl>
              </label>
            </Cell>
            <Cell>
              <Lbl title="Seed piers from this artist's most popular tracks. Off = cluster-medoid selection; On = popularity-weighted medoids; 🔥 Pure Hits = top-N most popular tracks only. Forced 🔥 when Bangers = Oops.">
                popular seeds
              </Lbl>
              <select
                value={popularityMode === "oops" ? "fire" : popularSeedsMode}
                onChange={(e) => setPopularSeedsMode(e.target.value as "off" | "on" | "fire")}
                disabled={popularityMode === "oops"}
                className={SEL}
                title="Seed piers from this artist's most popular tracks. Off = cluster-medoid selection; On = popularity-weighted medoids; 🔥 Pure Hits = top-N most popular tracks only. Forced 🔥 when Bangers = Oops."
              >
                <option value="off">Off</option>
                <option value="on">On</option>
                <option value="fire">🔥</option>
              </select>
            </Cell>
          </>
        )}

        {/* Generate */}
        <Cell push>
          <button
            onClick={() => submit()}
            disabled={busy}
            className="bg-[#5eead4] text-[#0f1115] font-bold text-[11px] px-4 py-[4px] rounded disabled:opacity-50 whitespace-nowrap"
          >
            {busy ? "Generating…" : "▸ Generate"}
          </button>
        </Cell>
        {mode === "artist" && (
          <Cell>
            <button onClick={() => { setSeedEpoch((e) => e + 1); submit(seedEpoch + 1); }}
              disabled={busy}
              className="border border-[#5eead4] text-[#5eead4] text-[11px] px-3 py-[4px] rounded disabled:opacity-50 whitespace-nowrap"
              title="Re-roll: same settings, fresh seed tracks.">
              ↻ New Seeds
            </button>
          </Cell>
        )}
        <Cell>
          <button
            type="button"
            data-testid="more-controls"
            aria-expanded={showMore}
            aria-controls="advanced-controls-region"
            onClick={() => setShowMore((v) => !v)}
            className="@md:hidden border border-[#23262d] text-[#8b939d] text-[11px] px-3 py-[4px] rounded whitespace-nowrap"
          >
            {showMore ? "Less controls ▴" : "More controls ▾"}
          </button>
        </Cell>
      </div>

      {/* ── Tag steering (artist mode): the seed artist's published genres ── */}
      {mode === "artist" && artistTags.length > 0 && (
        <div className="flex flex-wrap items-center gap-1.5 px-3 py-[6px] bg-[#16181d] border-b border-[#1e2128]" data-testid="steering-chips">
          {artistTags.map((t) => {
            const on = steeringTags.includes(t.name);
            const capped = !on && steeringTags.length >= 3;
            return (
              <button
                key={t.name}
                type="button"
                disabled={capped}
                onClick={() => toggleSteeringTag(t.name)}
                title={`${t.release_count} release${t.release_count === 1 ? "" : "s"}`}
                className={
                  "rounded-full border px-2 py-0.5 text-[11px] transition-colors " +
                  (on
                    ? "border-[#5eead4] bg-[#5eead4]/15 text-[#5eead4]"
                    : capped
                      ? "border-[#23262d] text-[#8b939d] opacity-40"
                      : "border-[#23262d] text-[#8b939d] hover:bg-[#1e2229]")
                }
              >
                {t.name}
              </button>
            );
          })}
        </div>
      )}
      {mode === "artist" && tagsFetched && artistTags.length === 0 && seed.trim() !== "" && (
        <div className="px-3 py-[6px] bg-[#16181d] border-b border-[#1e2128]">
          <p className="text-[11px] text-[#5b6470]">
            No published genres for this artist — run enrichment publish to enable tag steering.
          </p>
        </div>
      )}

      {/* ── Advanced controls: Rows 2–3 (collapse below @md container width) ── */}
      <div id="advanced-controls-region" data-testid="advanced-controls" className={showMore ? "" : "@max-md:hidden"}>

      {/* ── ROW 2: cohesion + matching ──────────────────────────────────────── */}
      <div className="flex flex-wrap items-center bg-[#13151a] border-b border-[#1e2128]">
        <Cell>
          <Lbl title="Overall beam tightness — how strictly the playlist stays within a sonic-genre neighbourhood. Strict = very cohesive; Discover = wide-ranging.">
            cohesion
          </Lbl>
          <select value={cohesion} onChange={(e) => setCohesion(e.target.value)} className={SEL}
            title="Overall beam tightness — how strictly the playlist stays within a sonic-genre neighbourhood. Strict = very cohesive; Discover = wide-ranging.">
            {["strict", "narrow", "dynamic", "discover"].map((v) => <option key={v} value={v}>{v}</option>)}
          </select>
        </Cell>
        <Cell>
          <Lbl title="How strictly genre tags must match. Off = ignore genre entirely.">genre</Lbl>
          <select value={axes.genre_mode} onChange={(e) => setAxes({ ...axes, genre_mode: e.target.value })} className={SEL}
            title="How strictly genre tags must match. Off = ignore genre entirely.">
            {["off", "discover", "dynamic", "narrow", "strict"].map((v) => <option key={v} value={v}>{v}</option>)}
          </select>
        </Cell>
        <Cell>
          <Lbl title="How closely the sonic fingerprint (timbre, rhythm, harmony) must match. Off = ignore sonic similarity.">sonic</Lbl>
          <select value={axes.sonic_mode} onChange={(e) => setAxes({ ...axes, sonic_mode: e.target.value })} className={SEL}
            title="How closely the sonic fingerprint (timbre, rhythm, harmony) must match. Off = ignore sonic similarity.">
            {["off", "dynamic", "narrow", "strict"].map((v) => <option key={v} value={v}>{v}</option>)}
          </select>
        </Cell>
        <Cell>
          <Lbl title="How closely the rhythmic feel must match. Strict = very similar BPM and groove; Dynamic = allows more variation.">pace</Lbl>
          <select value={axes.pace_mode} onChange={(e) => setAxes({ ...axes, pace_mode: e.target.value })} className={SEL}
            title="How closely the rhythmic feel must match. Strict = very similar BPM and groove; Dynamic = allows more variation.">
            {["off", "dynamic", "narrow", "strict"].map((v) => <option key={v} value={v}>{v}</option>)}
          </select>
        </Cell>
        <Cell>
          <Lbl title="Oops, All Bangers — bias the bridge tracks toward each artist's most popular (Last.fm) songs. Off = today; On = lean popular; Oops, All Bangers = library greatest-hits.">bangers</Lbl>
          <select value={popularityMode} onChange={(e) => setPopularityMode(e.target.value as "off" | "on" | "oops")} className={SEL}
            title="Bias the bridge tracks toward each artist's most popular (Last.fm) songs.">
            <option value="off">Off</option>
            <option value="on">On</option>
            <option value="oops">Oops, All Bangers</option>
          </select>
        </Cell>
      </div>

      {/* ── ROW 3: freshness + spacing ──────────────────────────────────────── */}
      <div className="flex flex-wrap items-center bg-[#111317]">
        <Cell>
          <label
            className="flex items-center gap-1.5 cursor-pointer select-none"
            title="Exclude tracks played recently. Uncheck to ignore playback history entirely."
          >
            <input
              type="checkbox"
              checked={recencyEnabled}
              onChange={(e) => setRecencyEnabled(e.target.checked)}
              className="accent-[#5eead4] cursor-pointer"
            />
            <Lbl>freshness</Lbl>
          </label>
        </Cell>
        <Cell>
          <Lbl title="How far back to look for recently-played tracks.">within</Lbl>
          <select value={recencyDays} onChange={(e) => setRecencyDays(Number(e.target.value))} disabled={!recencyEnabled} className={SEL}
            title="How far back to look for recently-played tracks.">
            {[7, 14, 30, 60, 90].map((d) => <option key={d} value={d}>{d}d</option>)}
          </select>
        </Cell>
        <Cell>
          <Lbl title="Minimum play count to consider a track recently played.">played</Lbl>
          <select value={recencyPlays} onChange={(e) => setRecencyPlays(Number(e.target.value))} disabled={!recencyEnabled} className={SEL}
            title="Minimum play count to consider a track recently played.">
            {[1, 2, 3].map((n) => <option key={n} value={n}>{n}+</option>)}
          </select>
          <Lbl>times</Lbl>
        </Cell>
        <Cell>
          <label
            className="flex items-center gap-1.5 cursor-pointer select-none"
            title="Also apply the recency filter to seed candidates, not just bridge tracks."
          >
            <input
              type="checkbox"
              checked={excludeRecentSeeds}
              onChange={(e) => setExcludeRecentSeeds(e.target.checked)}
              disabled={!recencyEnabled}
              className="accent-[#5eead4] cursor-pointer disabled:opacity-30"
            />
            <Lbl>skip recent seeds</Lbl>
          </label>
        </Cell>
        <Cell>
          <Lbl title="Minimum tracks between the same artist appearing again. Loose = 3, Normal = 6, Strong = 9, Very Strong = 12.">
            artist gap
          </Lbl>
          <select value={artistSpacing} onChange={(e) => setArtistSpacing(e.target.value)} className={SEL}
            title="Minimum tracks between the same artist appearing again. Loose = 3, Normal = 6, Strong = 9, Very Strong = 12.">
            <option value="loose">loose</option>
            <option value="normal">normal</option>
            <option value="strong">strong</option>
            <option value="very_strong">very strong</option>
          </select>
        </Cell>
        <Cell>
          <Lbl title="Scoring bonus for picking tracks from artists not yet in the playlist. Higher = more variety across artists. 'One each' hard-caps each non-seed artist to one track.">
            artist diversity
          </Lbl>
          <select value={diversityLevel} onChange={(e) => setDiversityLevel(Number(e.target.value))} className={SEL}
            title="Scoring bonus for picking tracks from artists not yet in the playlist. Higher = more variety across artists. 'One each' hard-caps each non-seed artist to one track.">
            {DIVERSITY_LABELS.map((label, i) => (
              <option key={i} value={i}>{label}</option>
            ))}
          </select>
        </Cell>
      </div>
      </div>

    </div>
  );
}
