import { Group, Panel, Separator } from "react-resizable-panels";
import type { ReactNode } from "react";

// Separator className: v4 uses data-separator for hover/active; no resize-handle-state attr
const handleV =
  "w-1.5 bg-bg hover:bg-accent transition-colors cursor-col-resize data-[disabled]:cursor-default";
const handleH =
  "h-1.5 bg-bg hover:bg-accent transition-colors cursor-row-resize data-[disabled]:cursor-default";

export function Shell(props: {
  topBar: ReactNode;
  jobs: ReactNode;
  center: ReactNode;
  right: ReactNode;
  logs: ReactNode;
}) {
  return (
    <div className="h-screen flex flex-col bg-bg text-text">
      <header className="flex items-center justify-between px-4 py-2.5 bg-panel border-b border-border">
        {props.topBar}
      </header>
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
          className="bg-[#0c0e12] border-t border-border"
        >
          {props.logs}
        </Panel>
      </Group>
    </div>
  );
}
