import { describe, it, expect, beforeEach, afterEach, vi } from "vitest";
import { renderHook, cleanup } from "@testing-library/react";
import { useWorkerEvents } from "./ws";
import type { WsEvent } from "./types";

// Controllable WebSocket double. readyState numbers mirror the real constants
// (0 CONNECTING, 1 OPEN, 3 CLOSED).
class FakeWebSocket {
  static instances: FakeWebSocket[] = [];
  url: string;
  readyState = 0;
  onopen: (() => void) | null = null;
  onmessage: ((ev: { data: string }) => void) | null = null;
  onclose: (() => void) | null = null;
  onerror: (() => void) | null = null;
  sent: string[] = [];
  constructor(url: string) {
    this.url = url;
    FakeWebSocket.instances.push(this);
  }
  send(d: string) {
    this.sent.push(d);
  }
  close() {
    this.readyState = 3;
  }
  // Test helpers: drive the server side of the socket.
  open() {
    this.readyState = 1;
    this.onopen?.();
  }
  serverClose() {
    this.readyState = 3;
    this.onclose?.();
  }
}

const sockets = () => FakeWebSocket.instances;

beforeEach(() => {
  vi.useFakeTimers();
  FakeWebSocket.instances = [];
  vi.stubGlobal("WebSocket", FakeWebSocket);
});

afterEach(() => {
  cleanup();
  vi.useRealTimers();
  vi.unstubAllGlobals();
});

describe("useWorkerEvents reconnect", () => {
  it("opens one socket and delivers parsed events", () => {
    const seen: WsEvent[] = [];
    renderHook(() => useWorkerEvents((e) => seen.push(e)));
    expect(sockets()).toHaveLength(1);
    sockets()[0].open();
    sockets()[0].onmessage?.({ data: JSON.stringify({ type: "log", msg: "hi" }) });
    expect(seen).toHaveLength(1);
    expect(seen[0].type).toBe("log");
  });

  it("reconnects after the socket closes", () => {
    renderHook(() => useWorkerEvents(() => {}));
    sockets()[0].open();
    sockets()[0].serverClose();
    expect(sockets()).toHaveLength(1);
    vi.advanceTimersByTime(1000);
    expect(sockets()).toHaveLength(2);
  });

  it("backs off exponentially while reconnects keep failing", () => {
    renderHook(() => useWorkerEvents(() => {}));
    sockets()[0].serverClose(); // first failure -> retry in 1000ms
    vi.advanceTimersByTime(1000);
    expect(sockets()).toHaveLength(2);
    sockets()[1].serverClose(); // second consecutive failure -> retry in 2000ms
    vi.advanceTimersByTime(1999);
    expect(sockets()).toHaveLength(2);
    vi.advanceTimersByTime(1);
    expect(sockets()).toHaveLength(3);
  });

  it("resets the backoff after a successful connection", () => {
    renderHook(() => useWorkerEvents(() => {}));
    sockets()[0].serverClose();
    vi.advanceTimersByTime(1000);
    sockets()[1].open(); // success resets the failure count
    sockets()[1].serverClose();
    vi.advanceTimersByTime(1000); // back to the base delay, not 2000ms
    expect(sockets()).toHaveLength(3);
  });

  it("reconnects immediately when the tab becomes visible while disconnected", () => {
    renderHook(() => useWorkerEvents(() => {}));
    sockets()[0].open();
    sockets()[0].serverClose();
    // iOS Safari killed the socket in the background; user returns before the
    // backoff timer fires.
    document.dispatchEvent(new Event("visibilitychange"));
    expect(sockets()).toHaveLength(2);
    // The pending backoff must not produce a duplicate connection on top.
    sockets()[1].open();
    vi.advanceTimersByTime(60000);
    expect(sockets()).toHaveLength(2);
  });

  it("delivers events through the reconnected socket", () => {
    const seen: WsEvent[] = [];
    renderHook(() => useWorkerEvents((e) => seen.push(e)));
    sockets()[0].serverClose();
    vi.advanceTimersByTime(1000);
    sockets()[1].open();
    sockets()[1].onmessage?.({ data: JSON.stringify({ type: "done", job_id: "j9" }) });
    expect(seen).toHaveLength(1);
    expect(seen[0].job_id).toBe("j9");
  });

  it("stops reconnecting after unmount", () => {
    const { unmount } = renderHook(() => useWorkerEvents(() => {}));
    sockets()[0].open();
    unmount();
    sockets()[0].serverClose();
    vi.advanceTimersByTime(60000);
    expect(sockets()).toHaveLength(1);
  });
});
