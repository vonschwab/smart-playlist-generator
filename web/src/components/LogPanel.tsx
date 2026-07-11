import { useEffect, useRef } from "react";

export function LogPanel({ lines }: { lines: string[] }) {
  const end = useRef<HTMLDivElement>(null);
  useEffect(() => { end.current?.scrollIntoView(); }, [lines.length]);
  return (
    <div className="h-full overflow-auto px-3 py-2 font-mono text-2xs leading-relaxed text-faint" data-testid="log-panel">
      {lines.length === 0 && (
        <div className="py-6 text-center font-sans">Logs stream here during generation.</div>
      )}
      {lines.map((l, i) => (
        <div key={i} className={l.startsWith("ERROR") ? "text-danger" : l.startsWith("WARNING") ? "text-warn" : ""}>{l}</div>
      ))}
      <div ref={end} />
    </div>
  );
}
