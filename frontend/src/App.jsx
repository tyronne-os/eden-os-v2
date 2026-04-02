import React, { useState, useCallback, useRef, useEffect } from "react";
import { useWebSocket } from "./hooks/useWebSocket.js";
import { useAudioRecorder } from "./hooks/useAudioRecorder.js";
import EveAlive from "./components/EveAlive.jsx";
import ChatPanel from "./components/ChatPanel.jsx";
import StatusBar from "./components/StatusBar.jsx";
import "./App.css";

const API_BASE = "/api";
const FRAME_INTERVAL_MS = 1000 / 30; // 30fps = ~33.3ms per frame

export default function App() {
  const [currentFrame, setCurrentFrame] = useState(null);
  const [messages, setMessages] = useState([]);
  const [pipelineInfo, setPipelineInfo] = useState(null);
  const [loading, setLoading] = useState(false);
  const [speaking, setSpeaking] = useState(false);

  // Ref to cancel any in-progress frame animation
  const animationRef = useRef(null);

  // WebSocket connects immediately — no click needed
  const { connected, lastFrame } = useWebSocket(
    `ws://${window.location.host}/ws`,
    (frame) => setCurrentFrame(frame)
  );

  // Audio recorder for voice input
  const { recording, startRecording, stopRecording } = useAudioRecorder();

  /**
   * Play an array of base64 JPEG frames at 30fps using requestAnimationFrame,
   * synchronized with audio playback. Returns a cancel function.
   */
  const playFramesWithAudio = useCallback((frames, audioB64) => {
    // Cancel any previous animation
    if (animationRef.current) {
      animationRef.current();
      animationRef.current = null;
    }

    let cancelled = false;
    const cancel = () => {
      cancelled = true;
      setSpeaking(false);
      setCurrentFrame(null); // return to idle alive
    };
    animationRef.current = cancel;

    // Prepare audio element (if present)
    let audio = null;
    if (audioB64) {
      const audioBytes = Uint8Array.from(atob(audioB64), (c) => c.charCodeAt(0));
      const blob = new Blob([audioBytes], { type: "audio/wav" });
      audio = new Audio(URL.createObjectURL(blob));
    }

    if (frames && frames.length > 0) {
      setSpeaking(true);
      // Show first frame immediately
      setCurrentFrame(frames[0]);

      let frameIdx = 0;
      let startTime = null;

      const tick = (timestamp) => {
        if (cancelled) return;
        if (startTime === null) {
          startTime = timestamp;
          // Start audio at the same time as the first rAF tick
          if (audio) audio.play().catch(() => {});
        }

        // Determine which frame should be shown based on elapsed time
        const elapsed = timestamp - startTime;
        const targetIdx = Math.min(
          Math.floor(elapsed / FRAME_INTERVAL_MS),
          frames.length - 1
        );

        if (targetIdx !== frameIdx) {
          frameIdx = targetIdx;
          setCurrentFrame(frames[frameIdx]);
        }

        // Keep going until we've shown the last frame
        if (frameIdx < frames.length - 1) {
          requestAnimationFrame(tick);
        } else {
          // Animation done — return to idle alive
          animationRef.current = null;
          setSpeaking(false);
          setCurrentFrame(null);
        }
      };

      requestAnimationFrame(tick);
    } else {
      // No frames — just play audio, Eve stays alive in idle
      setSpeaking(true);
      if (audio) {
        audio.play().catch(() => {});
        audio.addEventListener("ended", () => setSpeaking(false));
        // Fallback timeout in case ended event doesn't fire
        setTimeout(() => setSpeaking(false), 15000);
      } else {
        setSpeaking(false);
      }
    }

    return cancel;
  }, []);

  // ── Auto-welcome on mount (no click needed) ──────────────────
  const welcomeSent = useRef(false);
  useEffect(() => {
    if (welcomeSent.current) return;
    welcomeSent.current = true;

    (async () => {
      setLoading(true);
      try {
        const resp = await fetch(`${API_BASE}/welcome`, { method: "POST" });
        const data = await resp.json();

        if (data.text) {
          setMessages([{ role: "eve", text: data.text }]);
        }

        setPipelineInfo({
          pipeline: data.pipeline_used,
          frames: data.frame_count,
          elapsed: data.elapsed_s,
        });

        playFramesWithAudio(
          data.frame_count > 0 ? data.frames : null,
          data.audio_b64
        );
      } catch (e) {
        // Welcome failed — Eve is still alive and idle, just no greeting
        console.warn("Welcome fetch failed:", e.message);
      } finally {
        setLoading(false);
      }
    })();
  }, [playFramesWithAudio]);

  const handleSendMessage = useCallback(async (text) => {
    if (!text.trim()) return;

    setMessages((prev) => [...prev, { role: "user", text }]);
    setLoading(true);

    try {
      const resp = await fetch(`${API_BASE}/chat?message=${encodeURIComponent(text)}`, {
        method: "POST",
      });
      const data = await resp.json();

      if (data.response) {
        setMessages((prev) => [...prev, { role: "eve", text: data.response }]);
      }

      setPipelineInfo({
        pipeline: data.pipeline_used,
        frames: data.frame_count,
        elapsed: data.elapsed_s,
      });

      playFramesWithAudio(
        data.frame_count > 0 ? data.frames : null,
        data.audio_b64
      );
    } catch (e) {
      setMessages((prev) => [...prev, { role: "system", text: `Error: ${e.message}` }]);
    } finally {
      setLoading(false);
    }
  }, [playFramesWithAudio]);

  const handleVoiceInput = useCallback(async () => {
    if (recording) {
      const audioBlob = await stopRecording();
      if (!audioBlob) return;
      setLoading(true);
      try {
        const formData = new FormData();
        formData.append("audio", audioBlob, "recording.wav");
        const resp = await fetch(`${API_BASE}/chat`, { method: "POST", body: formData });
        const data = await resp.json();
        if (data.user_message) {
          setMessages((prev) => [...prev, { role: "user", text: data.user_message }]);
        }
        if (data.response) {
          setMessages((prev) => [...prev, { role: "eve", text: data.response }]);
        }

        playFramesWithAudio(
          data.frame_count > 0 ? data.frames : null,
          data.audio_b64
        );
      } catch (e) {
        setMessages((prev) => [...prev, { role: "system", text: `Error: ${e.message}` }]);
      } finally {
        setLoading(false);
      }
    } else {
      startRecording();
    }
  }, [recording, startRecording, stopRecording, playFramesWithAudio]);

  // No splash screen — Eve is alive from frame zero
  return (
    <div className="eden-app">
      <EveAlive
        frame={currentFrame}
        speaking={speaking || loading}
      />
      <ChatPanel
        messages={messages}
        onSend={handleSendMessage}
        onVoice={handleVoiceInput}
        recording={recording}
        loading={loading}
      />
      <StatusBar
        connected={connected}
        pipeline={pipelineInfo}
        cssFallback={!currentFrame}
      />
    </div>
  );
}
