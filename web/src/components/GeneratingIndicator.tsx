// Shown in the results area while a generation is in flight. The signature is
// an audio-equalizer: bars that breathe in the accent color — thematic to the
// subject (music) rather than a generic spinner. Under prefers-reduced-motion
// the global rule stops the animation and the bars rest at staggered heights.
const BARS = [0.55, 0.8, 1, 0.65, 0.85];

export function GeneratingIndicator({ label = "Building your playlist…", status }: { label?: string; status?: string }) {
  return (
    <div
      data-testid="generating-indicator"
      className="h-full flex flex-col items-center justify-center gap-4 px-6 py-16 text-center"
    >
      <div className="flex items-end gap-1 h-9" aria-hidden="true">
        {BARS.map((h, i) => (
          <span
            key={i}
            className="eq-bar w-1.5 rounded-full bg-accent"
            style={{ height: `${h * 100}%`, animationDelay: `${i * 0.12}s` }}
          />
        ))}
      </div>
      <div className="text-muted text-sm" role="status">
        {label}
      </div>
      {status && (
        <div className="text-faint text-2xs font-mono max-w-full truncate">{status}</div>
      )}
    </div>
  );
}
