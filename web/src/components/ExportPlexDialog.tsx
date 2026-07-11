import * as Dialog from "@radix-ui/react-dialog";
import { useEffect, useState } from "react";
import { api } from "../lib/api";
import type { TrackOut } from "../lib/types";

export interface ExportPlexDialogProps {
  open: boolean;
  onOpenChange: (o: boolean) => void;
  tracks: TrackOut[];
  defaultName: string;
}

export function ExportPlexDialog(props: ExportPlexDialogProps) {
  const [name, setName] = useState("");
  const [status, setStatus] = useState<"idle" | "exporting" | "done" | "error">("idle");
  const [msg, setMsg] = useState("");

  useEffect(() => {
    if (props.open) { setName(props.defaultName); setStatus("idle"); setMsg(""); }
  }, [props.open, props.defaultName]);

  const doExport = async () => {
    setStatus("exporting"); setMsg("");
    try {
      const r = await api.exportPlex({ title: name, tracks: props.tracks });
      setStatus("done"); setMsg(`Exported to Plex (key ${r.playlist_key}).`);
    } catch (e) { setStatus("error"); setMsg(String(e)); }
  };

  return (
    <Dialog.Root open={props.open} onOpenChange={props.onOpenChange}>
      <Dialog.Portal>
        <Dialog.Overlay className="fixed inset-0 bg-black/60 z-40" />
        <Dialog.Content
          data-testid="export-plex-dialog"
          className="fixed left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 z-50 w-[420px] max-w-[90vw] max-h-[85dvh] overflow-y-auto bg-panel border border-border rounded-lg shadow-2xl"
        >
          <div className="px-5 py-3 border-b border-border flex items-baseline justify-between">
            <Dialog.Title className="text-text text-sm font-semibold">Export to Plex</Dialog.Title>
            <Dialog.Close className="text-faint text-lg leading-none p-1 -m-1 inline-flex items-center justify-center pointer-coarse:min-w-11 pointer-coarse:min-h-11">×</Dialog.Close>
          </div>
          <div className="px-5 py-4">
            <div className="text-faint text-[9px] uppercase tracking-wide mb-2">Playlist name</div>
            <input
              data-testid="plex-name"
              value={name}
              onChange={(e) => setName(e.target.value)}
              className="w-full bg-panel2 border border-border rounded text-xs text-text px-2.5 py-1.5"
            />
            {status === "error" && <div className="text-danger text-xs mt-2">{msg}</div>}
            {status === "done" && <div className="text-accent text-xs mt-2">{msg}</div>}
          </div>
          <div className="px-5 py-3 border-t border-border flex justify-end gap-2 bg-panel2">
            <Dialog.Close className="border border-border text-muted text-xs px-3.5 py-1.5 pointer-coarse:min-h-11 rounded">Close</Dialog.Close>
            <button
              onClick={doExport}
              disabled={status === "exporting" || !name.trim()}
              className="bg-accent text-bg font-semibold text-xs px-3.5 py-1.5 pointer-coarse:min-h-11 rounded disabled:opacity-50"
            >
              {status === "exporting" ? "Exporting…" : "Export"}
            </button>
          </div>
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  );
}
