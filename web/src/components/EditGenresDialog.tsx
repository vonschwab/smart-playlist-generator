import * as Dialog from "@radix-ui/react-dialog";
import { useEffect, useState } from "react";
import { api } from "../lib/api";
import type { CanonicalGenre } from "../lib/types";

export interface EditGenresDialogProps {
  open: boolean;
  onOpenChange: (o: boolean) => void;
  artist: string;
  album: string;
  initialGenres: string[];
  onSaved: (artist: string, album: string, genres: string[]) => void;
}

export function EditGenresDialog(props: EditGenresDialogProps) {
  const [genres, setGenres] = useState<string[]>([]);
  const [input, setInput] = useState("");
  const [err, setErr] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);
  const [suggestions, setSuggestions] = useState<CanonicalGenre[]>([]);
  const [unknown, setUnknown] = useState<string[]>([]);

  // On open, seed chips from the props, then refresh from the live authority.
  useEffect(() => {
    if (!props.open) return;
    setInput(""); setErr(null); setUnknown([]); setSuggestions([]);
    setGenres([...props.initialGenres]);
    api.albumGenres(props.artist, props.album)
      .then((r) => { if (r.genres.length) setGenres(r.genres); })
      .catch(() => {});
  }, [props.open, props.artist, props.album, props.initialGenres]);

  // Debounced autocomplete over the canonical taxonomy vocabulary.
  useEffect(() => {
    const q = input.trim();
    if (!q) { setSuggestions([]); return; }
    const h = setTimeout(() => {
      api.genresSearch(q, 8).then((r) => setSuggestions(r.items)).catch(() => setSuggestions([]));
    }, 150);
    return () => clearTimeout(h);
  }, [input]);

  const addGenre = (raw?: string) => {
    const g = (raw ?? input).trim();
    if (g && !genres.some((x) => x.toLowerCase() === g.toLowerCase())) {
      setGenres([...genres, g]);
    }
    setInput(""); setSuggestions([]);
  };
  const removeGenre = (g: string) => setGenres(genres.filter((x) => x !== g));

  const save = async () => {
    // Flush a typed-but-not-committed genre so it isn't silently dropped.
    const pending = input.trim();
    const finalGenres = pending && !genres.some((x) => x.toLowerCase() === pending.toLowerCase())
      ? [...genres, pending] : genres;
    setSaving(true); setErr(null); setUnknown([]);
    try {
      const res = await api.editGenres({
        artist: props.artist, album: props.album,
        genres: finalGenres, base_genres: props.initialGenres,
      });
      props.onSaved(props.artist, props.album, res.resolved);
      if (res.unknown.length) {
        setUnknown(res.unknown);
        // Keep only the genres that were actually saved.
        setGenres(res.resolved);
      } else {
        props.onOpenChange(false);
      }
    } catch (e) { setErr(String(e)); } finally { setSaving(false); }
  };

  return (
    <Dialog.Root open={props.open} onOpenChange={props.onOpenChange}>
      <Dialog.Portal>
        <Dialog.Overlay className="fixed inset-0 bg-black/60 z-40" />
        <Dialog.Content
          data-testid="edit-genres-dialog"
          className="fixed left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 z-50 w-[480px] max-w-[90vw] bg-panel border border-border rounded-lg shadow-2xl"
        >
          <div className="px-5 py-3 border-b border-border flex items-baseline justify-between">
            <div>
              <Dialog.Title className="text-text text-sm font-semibold">Edit genres</Dialog.Title>
              <div className="text-muted text-[11px] mt-0.5"><span className="text-text">{props.album}</span> · {props.artist}</div>
            </div>
            <Dialog.Close className="text-faint text-lg leading-none">×</Dialog.Close>
          </div>

          <div className="px-5 py-4">
            <div className="text-faint text-[9px] uppercase tracking-wide mb-2">Genres (click × to remove)</div>
            <div className="flex flex-wrap gap-1.5 p-2.5 bg-panel2 border border-border rounded-md min-h-[42px]">
              {genres.map((g) => (
                <span key={g} className="bg-chip text-chipText text-[11px] px-2 py-0.5 rounded-full flex items-center gap-1">
                  {g}
                  <span onClick={() => removeGenre(g)} className="text-faint cursor-pointer">×</span>
                </span>
              ))}
              <input
                data-testid="genre-input"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={(e) => { if (e.key === "Enter") { e.preventDefault(); addGenre(); } }}
                placeholder="Add genre…"
                className="bg-transparent outline-none text-text text-[11px] min-w-[100px] px-1"
              />
            </div>
            {suggestions.length > 0 && (
              <div data-testid="genre-suggestions" className="mt-1 bg-panel2 border border-border rounded-md max-h-40 overflow-auto">
                {suggestions.map((s) => (
                  <div key={s.genre_id} onClick={() => addGenre(s.name)}
                       className="px-2.5 py-1 text-[11px] text-text hover:bg-border cursor-pointer">
                    {s.name}
                  </div>
                ))}
              </div>
            )}
            <div className="text-faint text-[9px] mt-1.5">Pick from the list (Enter to add) · applies to all tracks on this album</div>
            {unknown.length > 0 && (
              <div className="text-danger text-[11px] mt-2">
                Not in the genre vocabulary (not saved): {unknown.join(", ")}
              </div>
            )}
            {err && <div className="text-danger text-xs mt-2">{err}</div>}
          </div>

          <div className="px-5 py-3 border-t border-border flex items-center justify-between bg-panel2">
            <div className="text-faint text-[10px]">Saved to the genre authority · run “Refresh genres” to affect generation</div>
            <div className="flex gap-2">
              <Dialog.Close className="border border-border text-muted text-xs px-3.5 py-1.5 rounded">Cancel</Dialog.Close>
              <button onClick={save} disabled={saving} className="bg-accent text-bg font-semibold text-xs px-3.5 py-1.5 rounded disabled:opacity-50">
                {saving ? "Saving…" : "Save"}
              </button>
            </div>
          </div>
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  );
}
