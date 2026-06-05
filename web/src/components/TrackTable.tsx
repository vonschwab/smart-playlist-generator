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

const col = createColumnHelper<TrackOut>();
const fmt = (n?: number | null) => (n == null ? "—" : n.toFixed(2));

const columns = [
  col.accessor("position", {
    header: "#",
    cell: (c) => (
      <span className="font-mono text-faint text-[10px]">
        {String(c.getValue() + 1).padStart(2, "0")}
      </span>
    ),
  }),
  col.accessor("title", {
    header: "Track",
    cell: (c) => (
      <div>
        <div className="text-text text-xs">
          {c.getValue()}
          {c.row.original.genres.slice(0, 2).map((g) => (
            <span
              key={g}
              className="ml-1.5 bg-chip text-chipText text-[9px] px-1.5 py-0.5 rounded-full"
            >
              {g}
            </span>
          ))}
        </div>
        <div className="text-muted text-[10px]">{c.row.original.artist}</div>
      </div>
    ),
  }),
  col.accessor("sonic_similarity", {
    header: "T",
    cell: (c) => (
      <span className="font-mono text-accent text-[11px]">{fmt(c.getValue())}</span>
    ),
  }),
];

export function TrackTable({ tracks }: { tracks: TrackOut[] }) {
  const [sorting, setSorting] = useState<SortingState>([]);
  const table = useReactTable({
    data: tracks,
    columns,
    state: { sorting },
    onSortingChange: setSorting,
    getCoreRowModel: getCoreRowModel(),
    getSortedRowModel: getSortedRowModel(),
  });

  if (tracks.length === 0) {
    return (
      <div className="p-6 text-center text-muted text-xs">
        No playlist yet — generate one.
      </div>
    );
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
        {table.getRowModel().rows.map((r) => (
          <tr
            key={r.id}
            className="border-b border-[#181b21] odd:bg-panel2 hover:bg-[#15202b]"
          >
            {r.getVisibleCells().map((cell) => (
              <td key={cell.id} className="px-3 py-2 align-top">
                {flexRender(cell.column.columnDef.cell, cell.getContext())}
              </td>
            ))}
          </tr>
        ))}
      </tbody>
    </table>
  );
}
