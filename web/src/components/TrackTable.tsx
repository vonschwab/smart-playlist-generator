import {
  createColumnHelper,
  flexRender,
  getCoreRowModel,
  getSortedRowModel,
  useReactTable,
  type SortingState,
} from "@tanstack/react-table";
import { useState } from "react";
import type { TrackOut } from "../lib/types";
import { usePlayer } from "../contexts/PlayerContext";

const fmt = (n?: number | null) => (n == null ? "—" : n.toFixed(2));

export interface TrackTableProps {
  tracks: TrackOut[];
  blacklisted?: Set<string>;
  // x/y are viewport coordinates used to anchor the context menu at the cursor.
  onContextAction?: (track: TrackOut, index: number, x: number, y: number) => void;
}

export function TrackTable({ tracks, blacklisted, onContextAction }: TrackTableProps) {
  const [sorting, setSorting] = useState<SortingState>([]);
  const player = usePlayer();

  const col = createColumnHelper<TrackOut>();
  const columns = [
    col.display({
      id: "play",
      header: "",
      cell: (c) => {
        const idx = c.row.index;
        const isCurrent = player.current?.rating_key === c.row.original.rating_key;
        return (
          <button
            data-testid="play-btn"
            onClick={(e) => {
              e.stopPropagation();
              if (isCurrent) player.setPlaying(!player.playing);
              else player.load(tracks, idx);
            }}
            className={`text-sm ${isCurrent ? "text-accent" : "text-faint opacity-40 hover:opacity-100"}`}
            title={isCurrent && player.playing ? "Pause" : "Play"}
          >
            {isCurrent && player.playing ? "❚❚" : "▶"}
          </button>
        );
      },
    }),
    col.accessor("position", {
      header: "#",
      cell: (c) => {
        const isCurrent = player.current?.rating_key === c.row.original.rating_key;
        return (
          <span className={`font-mono text-[10px] ${isCurrent ? "text-accent" : "text-faint"}`}>
            {String(c.getValue() + 1).padStart(2, "0")}
          </span>
        );
      },
    }),
    col.accessor("title", {
      header: "Track",
      cell: (c) => {
        const bl = blacklisted?.has(c.row.original.rating_key ?? "");
        return (
          <div>
            <div className={`text-xs ${bl ? "text-text line-through opacity-60" : "text-text"}`}>
              {c.getValue()}
              {bl && (
                <span className="ml-1.5 bg-[#2a1a1a] text-danger text-[9px] px-1.5 py-0.5 rounded-full">
                  blacklisted
                </span>
              )}
              {c.row.original.genres.slice(0, 2).map((g) => (
                <span key={g} className="ml-1.5 bg-chip text-chipText text-[9px] px-1.5 py-0.5 rounded-full">
                  {g}
                </span>
              ))}
            </div>
            <div className="text-muted text-[10px]">{c.row.original.artist}</div>
          </div>
        );
      },
    }),
    col.accessor("sonic_similarity", {
      header: "T",
      cell: (c) => <span className="font-mono text-accent text-[11px]">{fmt(c.getValue())}</span>,
    }),
    col.display({
      id: "kebab",
      header: "",
      cell: (c) => (
        <button
          data-testid="kebab-btn"
          onClick={(e) => {
            e.stopPropagation();
            onContextAction?.(c.row.original, c.row.index, e.clientX, e.clientY);
          }}
          className="text-muted opacity-0 group-hover:opacity-100 hover:text-text text-sm"
          title="Actions"
        >
          ⋯
        </button>
      ),
    }),
  ];

  const table = useReactTable({
    data: tracks,
    columns,
    state: { sorting },
    onSortingChange: setSorting,
    getCoreRowModel: getCoreRowModel(),
    getSortedRowModel: getSortedRowModel(),
  });

  if (tracks.length === 0) {
    return <div className="p-6 text-center text-muted text-xs">No playlist yet — generate one.</div>;
  }

  return (
    <table className="w-full text-left" data-testid="track-table">
      <thead>
        {table.getHeaderGroups().map((hg) => (
          <tr key={hg.id} className="border-b border-border">
            {hg.headers.map((h) => (
              <th
                key={h.id}
                onClick={h.column.getToggleSortingHandler()}
                className="px-3 py-2 text-[9px] uppercase tracking-wide text-faint cursor-pointer select-none"
              >
                {flexRender(h.column.columnDef.header, h.getContext())}
              </th>
            ))}
          </tr>
        ))}
      </thead>
      <tbody>
        {table.getRowModel().rows.map((r) => {
          const isCurrent = player.current?.rating_key === r.original.rating_key;
          return (
            <tr
              key={r.id}
              onContextMenu={(e) => {
                e.preventDefault();
                onContextAction?.(r.original, r.index, e.clientX, e.clientY);
              }}
              className={`group border-b border-[#181b21] ${
                isCurrent ? "bg-[#15202b]" : "odd:bg-panel2 hover:bg-[#15202b]"
              }`}
            >
              {r.getVisibleCells().map((cell) => (
                <td key={cell.id} className="px-3 py-2 align-top">
                  {flexRender(cell.column.columnDef.cell, cell.getContext())}
                </td>
              ))}
            </tr>
          );
        })}
      </tbody>
    </table>
  );
}
