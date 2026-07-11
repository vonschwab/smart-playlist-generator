import { useEffect, useRef } from "react";
import type { WsEvent } from "./types";

// The socket is mortal: iOS Safari kills background-tab WebSockets and never
// resumes them. Reconnect with exponential backoff after any close, and
// immediately when the tab returns to the foreground (visibility/pageshow/
// online). Missed events are healed by useJobReconcile polling, not replayed.
const BASE_RETRY_MS = 1000;
const MAX_RETRY_MS = 15000;

export function useWorkerEvents(onEvent: (e: WsEvent) => void) {
  const handler = useRef(onEvent);
  handler.current = onEvent;
  useEffect(() => {
    const proto = location.protocol === "https:" ? "wss" : "ws";
    const url = `${proto}://${location.host}/ws`;
    let ws: WebSocket | null = null;
    let retryTimer: ReturnType<typeof setTimeout> | null = null;
    let failures = 0;
    let disposed = false;

    const connect = () => {
      if (disposed) return;
      if (retryTimer) {
        clearTimeout(retryTimer);
        retryTimer = null;
      }
      ws = new WebSocket(url);
      ws.onopen = () => {
        failures = 0;
      };
      ws.onmessage = (m) => {
        try { handler.current(JSON.parse(m.data) as WsEvent); } catch { /* ignore */ }
      };
      ws.onclose = () => {
        if (disposed || retryTimer) return;
        const delay = Math.min(BASE_RETRY_MS * 2 ** failures, MAX_RETRY_MS);
        failures += 1;
        retryTimer = setTimeout(() => {
          retryTimer = null;
          connect();
        }, delay);
      };
    };

    // Numeric readyState (0 CONNECTING, 1 OPEN): the codebase convention, and
    // safe when the WebSocket global is a test double without the constants.
    const wake = () => {
      if (disposed || document.visibilityState !== "visible") return;
      if (ws && (ws.readyState === 0 || ws.readyState === 1)) return;
      connect();
    };
    document.addEventListener("visibilitychange", wake);
    window.addEventListener("pageshow", wake);
    window.addEventListener("online", wake);

    const ping = setInterval(() => { if (ws?.readyState === 1) ws.send("ping"); }, 20000);
    connect();
    return () => {
      disposed = true;
      clearInterval(ping);
      if (retryTimer) clearTimeout(retryTimer);
      document.removeEventListener("visibilitychange", wake);
      window.removeEventListener("pageshow", wake);
      window.removeEventListener("online", wake);
      ws?.close();
    };
  }, []);
}
