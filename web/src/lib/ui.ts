// Shared className recipes for the small set of recurring control kinds
// (dialog footer buttons, tool-run buttons, genre chips). One definition per
// kind so token/spacing tweaks land in a single place (discipline: reuse-first).

export const btnPrimary =
  "bg-accent text-bg font-semibold text-xs px-3.5 py-1.5 pointer-coarse:min-h-11 rounded disabled:opacity-50";

export const btnGhost =
  "border border-border text-muted text-xs px-3.5 py-1.5 pointer-coarse:min-h-11 rounded disabled:opacity-50";

export const chip =
  "text-2xs bg-chip text-chipText px-1.5 py-0.5 rounded-full shrink-0";
