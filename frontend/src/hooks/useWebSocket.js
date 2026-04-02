import { useState, useEffect, useRef, useCallback } from "react";

export function useWebSocket(url, onFrame) {
  const [connected, setConnected] = useState(false);
  const [lastFrame, setLastFrame] = useState(null);
  const wsRef = useRef(null);
  const reconnectTimer = useRef(null);

  const connect = useCallback(() => {
    if (!url) return;

    const ws = new WebSocket(url);
    wsRef.current = ws;

    ws.onopen = () => {
      setConnected(true);
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
          onFrame?.(msg.data);
        }
      } catch {}
    };

    ws.onclose = () => {
      setConnected(false);
      if (ws._pingInterval) clearInterval(ws._pingInterval);
      // Auto-reconnect after 3s
      reconnectTimer.current = setTimeout(connect, 3000);
    };

    ws.onerror = () => {
      ws.close();
    };
  }, [url, onFrame]);

  useEffect(() => {
    connect();
    return () => {
      if (reconnectTimer.current) clearTimeout(reconnectTimer.current);
      if (wsRef.current) wsRef.current.close();
    };
  }, [connect]);

  return { connected, lastFrame };
}
