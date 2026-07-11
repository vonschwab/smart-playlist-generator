import * as Dialog from "@radix-ui/react-dialog";
import { useEffect, useState } from "react";
import { api } from "../lib/api";
import { btnGhost, btnPrimary } from "../lib/ui";
import type { CandidateOut } from "../lib/types";

export interface ReplaceDialogProps {
  open: boolean;
  onOpenChange: (o: boolean) => void;
  jobId: string;
  position: number;
  prevTitle: string;
  nextTitle: string;
  onConfirm: (position: number, candidate: CandidateOut) => void;
}

export function ReplaceDialog(props: ReplaceDialogProps) {
  const [candidates, setCandidates] = useState<CandidateOut[]>([]);
  const [selected, setSelected] = useState<number>(-1);
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    if (!props.open) return;
    setLoading(true); setErr(null); setSelected(-1); setCandidates([]);
    api.replaceSuggestions(props.jobId, props.position)
      .then((r) => setCandidates(r.candidates))
      .catch((e) => setErr(String(e)))
      .finally(() => setLoading(false));
  }, [props.open, props.jobId, props.position]);

  return (
    <Dialog.Root open={props.open} onOpenChange={props.onOpenChange}>
      <Dialog.Portal>
        <Dialog.Overlay className="fixed inset-0 bg-black/60 z-40" />
        <Dialog.Content
          data-testid="replace-dialog"
          className="fixed left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 z-50 w-[560px] max-w-[90vw] max-h-[85dvh] overflow-y-auto bg-panel border border-border rounded-lg shadow-2xl"
        >
          <div className="px-5 py-3 border-b border-border flex items-baseline justify-between">
            <div>
              <Dialog.Title className="text-text text-sm font-semibold">Replace track</Dialog.Title>
              <div className="text-muted text-xs mt-0.5">
                Position {props.position + 1} · between <span className="text-text">{props.prevTitle}</span> and <span className="text-text">{props.nextTitle}</span>
              </div>
            </div>
            <Dialog.Close className="text-faint text-lg leading-none p-1 -m-1 inline-flex items-center justify-center pointer-coarse:min-w-11 pointer-coarse:min-h-11">×</Dialog.Close>
          </div>

          <div className="px-5 py-2 bg-panel2 border-b border-border text-2xs text-faint">
            {loading ? "Loading candidates…" : `${candidates.length} candidates · ranked by transition fit to neighbors`}
          </div>

          {err && <div className="px-5 py-3 text-danger text-xs">{err}</div>}

          <div className="max-h-[260px] overflow-y-auto">
            <table className="w-full text-xs">
              <tbody>
                {candidates.map((c, i) => (
                  <tr
                    key={c.track_id}
                    data-testid="replace-candidate"
                    onClick={() => setSelected(i)}
                    style={{ opacity: 1 - i * 0.06 }}
                    className={`border-b border-hairline cursor-pointer ${selected === i ? "bg-rowsel" : "hover:bg-rowsel"}`}
                  >
                    <td className="px-3 py-2 font-mono text-faint text-2xs w-8">{String(i + 1).padStart(2, "0")}</td>
                    <td className="px-3 py-2">
                      <div className="text-text">
                        {c.title}
                        {c.genres.slice(0, 2).map((g) => (
                          <span key={g} className="ml-1.5 bg-chip text-chipText text-2xs px-1.5 py-0.5 rounded-full">{g}</span>
                        ))}
                      </div>
                      <div className="text-muted text-2xs">{c.artist} · {c.album}</div>
                    </td>
                    <td className="px-3 py-2 font-mono text-accent">{c.fit_score.toFixed(2)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <div className="px-5 py-3 border-t border-border flex items-center justify-between bg-panel2">
            <div className="text-faint text-2xs">Click a row to select, then Replace</div>
            <div className="flex gap-2">
              <Dialog.Close className={btnGhost}>Cancel</Dialog.Close>
              <button
                disabled={selected < 0}
                onClick={() => { props.onConfirm(props.position, candidates[selected]); props.onOpenChange(false); }}
                className={btnPrimary}
              >
                Replace
              </button>
            </div>
          </div>
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  );
}
