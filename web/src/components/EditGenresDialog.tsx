import * as Dialog from "@radix-ui/react-dialog";
import { useEffect, useState } from "react";
import { api } from "../lib/api";

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

  useEffect(() => {
    if (props.open) { setGenres([...props.initialGenres]); setInput(""); setErr(null); }
  }, [props.open, props.initialGenres]);

  const addGenre = () => {
    const g = input.trim();
    if (g && !genres.some((x) => x.toLowerCase() === g.toLowerCase())) setGenres([...genres, g]);
    setInput("");
  };
  const removeGenre = (g: string) => setGenres(genres.filter((x) => x !== g));

  const save = async () => {
    setSaving(true); setErr(null);
    try {
      await api.editGenres({ artist: props.artist, album: props.album, genres });
      props.onSaved(props.artist, props.album, genres);
      props.onOpenChange(false);
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
            <div className="text-faint text-[9px] mt-1.5">Type a genre and press Enter · applies to all tracks on this album</div>
            {err && <div className="text-danger text-xs mt-2">{err}</div>}
          </div>

          <div className="px-5 py-3 border-t border-border flex items-center justify-between bg-panel2">
            <div className="text-faint text-[10px]">Saves a user override · does not affect source tags</div>
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
