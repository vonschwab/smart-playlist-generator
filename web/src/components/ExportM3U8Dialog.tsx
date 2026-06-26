import * as Dialog from "@radix-ui/react-dialog";
import { useEffect, useState } from "react";
import { downloadM3U8 } from "../lib/m3u";
import type { TrackOut } from "../lib/types";

export interface ExportM3U8DialogProps {
  open: boolean;
  onOpenChange: (o: boolean) => void;
  tracks: TrackOut[];
  defaultName: string;
}

export function ExportM3U8Dialog(props: ExportM3U8DialogProps) {
  const [name, setName] = useState("");

  useEffect(() => {
    if (props.open) setName(props.defaultName);
  }, [props.open, props.defaultName]);

  const doDownload = () => {
    downloadM3U8(props.tracks, name);
    props.onOpenChange(false);
  };

  return (
    <Dialog.Root open={props.open} onOpenChange={props.onOpenChange}>
      <Dialog.Portal>
        <Dialog.Overlay className="fixed inset-0 bg-black/60 z-40" />
        <Dialog.Content
          data-testid="export-m3u8-dialog"
          aria-describedby={undefined}
          className="fixed left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 z-50 w-[420px] max-w-[90vw] bg-panel border border-border rounded-lg shadow-2xl"
        >
          <div className="px-5 py-3 border-b border-border flex items-baseline justify-between">
            <Dialog.Title className="text-text text-sm font-semibold">Export as M3U8</Dialog.Title>
            <Dialog.Close className="text-faint text-lg leading-none">×</Dialog.Close>
          </div>
          <div className="px-5 py-4">
            <div className="text-faint text-[9px] uppercase tracking-wide mb-2">Playlist name</div>
            <input
              data-testid="m3u8-name"
              value={name}
              onChange={(e) => setName(e.target.value)}
              className="w-full bg-panel2 border border-border rounded text-xs text-text px-2.5 py-1.5"
            />
          </div>
          <div className="px-5 py-3 border-t border-border flex justify-end gap-2 bg-panel2">
            <Dialog.Close className="border border-border text-muted text-xs px-3.5 py-1.5 rounded">
              Cancel
            </Dialog.Close>
            <button
              data-testid="m3u8-download"
              onClick={doDownload}
              disabled={!name.trim()}
              className="bg-accent text-bg font-semibold text-xs px-3.5 py-1.5 rounded disabled:opacity-50"
            >
              Download
            </button>
          </div>
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  );
}
