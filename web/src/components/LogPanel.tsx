import { useEffect, useRef } from "react";

export function LogPanel({ lines }: { lines: string[] }) {
  const end = useRef<HTMLDivElement>(null);
  useEffect(() => { end.current?.scrollIntoView(); }, [lines.length]);
  return (
    <div className="h-full overflow-auto px-3 py-2 font-mono text-[10px] leading-relaxed text-faint" data-testid="log-panel">
      {lines.map((l, i) => (
        <div key={i} className={l.startsWith("ERROR") ? "text-danger" : l.startsWith("WARNING") ? "text-warn" : ""}>{l}</div>
      ))}
      <div ref={end} />
    </div>
  );
}
