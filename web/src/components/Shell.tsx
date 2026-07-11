import { Group, Panel, Separator } from "react-resizable-panels";
import { useState, type ReactNode } from "react";
import { useMediaQuery } from "../lib/useMediaQuery";

// Separator className: v4 uses data-separator for hover/active; no resize-handle-state attr
const handleV =
  "w-1.5 bg-bg hover:bg-accent transition-colors cursor-col-resize data-[disabled]:cursor-default";
const handleH =
  "h-1.5 bg-bg hover:bg-accent transition-colors cursor-row-resize data-[disabled]:cursor-default";

interface ShellProps {
  topBar: ReactNode;
  jobs: ReactNode;
  center: ReactNode;
  right: ReactNode;
  logs: ReactNode;
}

export function Shell(props: ShellProps) {
  // Single viewport breakpoint: drag-resizable panes only make sense with a pointer
  // and real width. Below it we render a touch-friendly single-column shell.
  const isDesktop = useMediaQuery("(min-width: 1024px)");
  return (
    <div className="h-screen supports-[height:100dvh]:h-dvh flex flex-col bg-bg text-text">
      <header className="flex flex-wrap items-center justify-between gap-2 px-4 py-2.5 bg-panel border-b border-border">
        {props.topBar}
      </header>
      {isDesktop ? <DesktopBody {...props} /> : <MobileBody {...props} />}
    </div>
  );
}

function DesktopBody(props: ShellProps) {
  return (
    <Group orientation="vertical" className="flex-1" id="pg-vert">
      <Panel defaultSize={78} minSize={40}>
        <Group orientation="horizontal" id="pg-horiz">
          <Panel
            defaultSize={16}
            minSize={10}
            collapsible
            className="bg-panel border-r border-border"
          >
            {props.jobs}
          </Panel>
          <Separator className={handleV} />
          <Panel defaultSize={62} minSize={30} className="bg-bg overflow-hidden">
            {props.center}
          </Panel>
          <Separator className={handleV} />
          <Panel
            defaultSize={22}
            minSize={14}
            collapsible
            className="bg-panel border-l border-border"
          >
            {props.right}
          </Panel>
        </Group>
      </Panel>
      <Separator className={handleH} />
      <Panel
        defaultSize={22}
        minSize={8}
        collapsible
        className="bg-well border-t border-border"
      >
        {props.logs}
      </Panel>
    </Group>
  );
}

type Region = "playlist" | "jobs" | "diag" | "logs";

function MobileBody(props: ShellProps) {
  const [region, setRegion] = useState<Region>("playlist");
  const active =
    region === "playlist" ? props.center
    : region === "jobs" ? props.jobs
    : region === "diag" ? props.right
    : props.logs;

  const tabs: Array<{ id: Region; label: string }> = [
    { id: "playlist", label: "Playlist" },
    { id: "jobs", label: "Jobs" },
    { id: "diag", label: "Advanced" },
    { id: "logs", label: "Logs" },
  ];

  return (
    <>
      <div className="flex-1 min-h-0 overflow-hidden bg-bg">{active}</div>
      {/* viewport-fit=cover: pb keeps the tabs clear of the home indicator. */}
      <nav
        data-testid="mobile-tabbar"
        className="flex shrink-0 border-t border-border bg-panel pb-[env(safe-area-inset-bottom)]"
      >
        {tabs.map((t) => (
          <button
            key={t.id}
            data-testid={`tab-mobile-${t.id}`}
            onClick={() => setRegion(t.id)}
            aria-current={region === t.id}
            className={[
              "flex-1 py-3 min-h-11 text-xs font-medium transition-colors",
              region === t.id ? "text-accent" : "text-faint hover:text-muted",
            ].join(" ")}
          >
            {t.label}
          </button>
        ))}
      </nav>
    </>
  );
}
