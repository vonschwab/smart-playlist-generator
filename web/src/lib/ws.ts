import { useEffect, useRef } from "react";
import type { WsEvent } from "./types";

export function useWorkerEvents(onEvent: (e: WsEvent) => void) {
  const handler = useRef(onEvent);
  handler.current = onEvent;
  useEffect(() => {
    const proto = location.protocol === "https:" ? "wss" : "ws";
    const ws = new WebSocket(`${proto}://${location.host}/ws`);
    ws.onmessage = (m) => {
      try { handler.current(JSON.parse(m.data) as WsEvent); } catch { /* ignore */ }
    };
    const ping = setInterval(() => ws.readyState === 1 && ws.send("ping"), 20000);
    return () => { clearInterval(ping); ws.close(); };
  }, []);
}
