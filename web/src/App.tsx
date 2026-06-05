import { Shell } from "./components/Shell";
import { AdvancedPanel } from "./components/AdvancedPanel";

export default function App() {
  return (
    <Shell
      topBar={
        <div className="font-bold text-sm">
          <span className="text-accent">◆</span> Playlist Generator
        </div>
      }
      jobs={<div className="p-3 text-xs text-muted">Jobs</div>}
      center={<div className="p-3 text-xs text-muted">Center</div>}
      right={<AdvancedPanel />}
      logs={<div className="p-3 font-mono text-[11px] text-faint">Logs</div>}
    />
  );
}
