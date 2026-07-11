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
import { GenreChips } from "./GenreChips";

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

  const hasPopularity = tracks.some((t) => t.popularity_rank != null);
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
          <span className={`font-mono text-2xs ${isCurrent ? "text-accent" : "text-faint"}`}>
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
                <span className="ml-1.5 bg-danger/10 text-danger text-2xs px-1.5 py-0.5 rounded-full">
                  blacklisted
                </span>
              )}
              <GenreChips
                genres={c.row.original.genres}
                chipClass="ml-1.5 bg-chip text-chipText text-2xs px-1.5 py-0.5 rounded-full"
              />
            </div>
            <div className="text-muted text-2xs">{c.row.original.artist}</div>
          </div>
        );
      },
    }),
    col.accessor("sonic_similarity", {
      header: "T",
      cell: (c) => <span className="font-mono text-accent text-xs">{fmt(c.getValue())}</span>,
    }),
    ...(hasPopularity
      ? [
          col.accessor("popularity_rank", {
            header: "Last.fm",
            meta: { cellClass: "@max-md:hidden" },
            cell: (c) => {
              const r = c.getValue();
              return (
                <span
                  className="font-mono text-xs text-faint"
                  title="Last.fm popularity rank within the artist's top tracks (lower = more popular)"
                >
                  {r == null ? "—" : `#${r}`}
                </span>
              );
            },
          }),
        ]
      : []),
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
          className="text-muted opacity-0 group-hover:opacity-100 pointer-coarse:opacity-60 hover:text-text text-sm"
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
    <div className="@container">
      <table className="w-full text-left" data-testid="track-table">
        <thead>
          {table.getHeaderGroups().map((hg) => (
            <tr key={hg.id} className="border-b border-border">
              {hg.headers.map((h) => {
                const extra = (h.column.columnDef.meta as { cellClass?: string } | undefined)?.cellClass ?? "";
                return (
                  <th
                    key={h.id}
                    onClick={h.column.getToggleSortingHandler()}
                    className={`px-3 py-2 text-2xs uppercase tracking-wide text-faint cursor-pointer select-none ${extra}`}
                  >
                    {flexRender(h.column.columnDef.header, h.getContext())}
                  </th>
                );
              })}
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
                className={`group border-b border-hairline ${
                  isCurrent ? "bg-rowsel" : "odd:bg-panel2 hover:bg-rowsel"
                }`}
              >
                {r.getVisibleCells().map((cell) => {
                  const extra = (cell.column.columnDef.meta as { cellClass?: string } | undefined)?.cellClass ?? "";
                  return (
                    <td key={cell.id} className={`px-3 py-2 align-top ${extra}`}>
                      {flexRender(cell.column.columnDef.cell, cell.getContext())}
                    </td>
                  );
                })}
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
