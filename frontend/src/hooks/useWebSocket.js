import { useState, useEffect, useRef } from "react";

export function useWebSocket(url, onFrame) {
  const [connected, setConnected] = useState(false);
  const [lastFrame, setLastFrame] = useState(null);
  const wsRef = useRef(null);
  const reconnectTimer = useRef(null);
  const backoffRef = useRef(3000); // start at 3s
  const onFrameRef = useRef(onFrame);
  const unmountingRef = useRef(false);

  // Keep onFrame callback ref current without triggering reconnects
  useEffect(() => {
    onFrameRef.current = onFrame;
  }, [onFrame]);

  useEffect(() => {
    if (!url) return;

    unmountingRef.current = false;

    // Guard against reconnect during page unload
    const handleBeforeUnload = () => {
      unmountingRef.current = true;
      if (reconnectTimer.current) clearTimeout(reconnectTimer.current);
      if (wsRef.current) wsRef.current.close();
    };
    window.addEventListener("beforeunload", handleBeforeUnload);

    function connect() {
      if (unmountingRef.current) return;

      const ws = new WebSocket(url);
      wsRef.current = ws;

      ws.onopen = () => {
        setConnected(true);
        backoffRef.current = 3000; // reset backoff on successful connect
        // Start ping loop
        const ping = setInterval(() => {
          if (ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify({ type: "ping" }));
          }
        }, 15000);
        ws._pingInterval = ping;
      };

      ws.onmessage = (event) => {
        try {
          const msg = JSON.parse(event.data);
          if (msg.type === "frame") {
            setLastFrame(msg.data);
            onFrameRef.current?.(msg.data);
          }
        } catch {
          // ignore parse errors
        }
      };

      ws.onclose = () => {
        setConnected(false);
        if (ws._pingInterval) clearInterval(ws._pingInterval);
        // Don't reconnect if unmounting or page is unloading
        if (unmountingRef.current) return;
        // Exponential backoff: 3s, 6s, 12s, 24s, max 30s
        const delay = backoffRef.current;
        backoffRef.current = Math.min(backoffRef.current * 2, 30000);
        reconnectTimer.current = setTimeout(connect, delay);
      };

      ws.onerror = () => {
        ws.close();
      };
    }

    connect();

    return () => {
      unmountingRef.current = true;
      window.removeEventListener("beforeunload", handleBeforeUnload);
      if (reconnectTimer.current) clearTimeout(reconnectTimer.current);
      if (wsRef.current) wsRef.current.close();
    };
  }, [url]); // Only reconnect when the URL itself changes

  return { connected, lastFrame };
}
