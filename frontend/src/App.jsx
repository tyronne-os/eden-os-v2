import React, { useState, useCallback } from "react";
import { useWebSocket } from "./hooks/useWebSocket.js";
import { useAudioRecorder } from "./hooks/useAudioRecorder.js";
import SplashScreen from "./components/SplashScreen.jsx";
import AvatarView from "./components/AvatarView.jsx";
import ChatPanel from "./components/ChatPanel.jsx";
import StatusBar from "./components/StatusBar.jsx";
import "./App.css";

const API_BASE = "/api";

export default function App() {
  const [started, setStarted] = useState(false);
  const [loading, setLoading] = useState(false);
  const [currentFrame, setCurrentFrame] = useState(null);
  const [messages, setMessages] = useState([]);
  const [pipelineInfo, setPipelineInfo] = useState(null);
  const [error, setError] = useState(null);

  // WebSocket for real-time frame streaming
  const { connected, lastFrame } = useWebSocket(
    started ? `ws://${window.location.host}/ws` : null,
    (frame) => setCurrentFrame(frame)
  );

  // Audio recorder for voice input
  const { recording, startRecording, stopRecording } = useAudioRecorder();

  // Feature 5: CSS fallback animation state
  const [useCssFallback, setUseCssFallback] = useState(false);

  const handleStart = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const resp = await fetch(`${API_BASE}/welcome`, { method: "POST" });
      const data = await resp.json();

      if (data.text) {
        setMessages([{ role: "eve", text: data.text }]);
      }

      // Play audio
      if (data.audio_b64) {
        const audioBytes = Uint8Array.from(atob(data.audio_b64), (c) => c.charCodeAt(0));
        const blob = new Blob([audioBytes], { type: "audio/wav" });
        const audio = new Audio(URL.createObjectURL(blob));
        audio.play().catch(() => {});
      }

      setPipelineInfo({
        pipeline: data.pipeline_used,
        frames: data.frame_count,
        elapsed: data.elapsed_s,
      });

      // Feature 5: if server says force strong pipeline or no frames, use CSS fallback
      if (data.frame_count === 0 || data.force_strong_pipeline) {
        setUseCssFallback(true);
      }

      setStarted(true);
    } catch (e) {
      setError(`Connection failed: ${e.message}`);
      setUseCssFallback(true);
      setStarted(true);
    } finally {
      setLoading(false);
    }
  }, []);

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

      if (data.audio_b64) {
        const audioBytes = Uint8Array.from(atob(data.audio_b64), (c) => c.charCodeAt(0));
        const blob = new Blob([audioBytes], { type: "audio/wav" });
        const audio = new Audio(URL.createObjectURL(blob));
        audio.play().catch(() => {});
      }

      setPipelineInfo({
        pipeline: data.pipeline_used,
        frames: data.frame_count,
        elapsed: data.elapsed_s,
      });
    } catch (e) {
      setMessages((prev) => [...prev, { role: "system", text: `Error: ${e.message}` }]);
    } finally {
      setLoading(false);
    }
  }, []);

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
        if (data.audio_b64) {
          const audioBytes = Uint8Array.from(atob(data.audio_b64), (c) => c.charCodeAt(0));
          const blob = new Blob([audioBytes], { type: "audio/wav" });
          new Audio(URL.createObjectURL(blob)).play().catch(() => {});
        }
      } catch (e) {
        setMessages((prev) => [...prev, { role: "system", text: `Error: ${e.message}` }]);
      } finally {
        setLoading(false);
      }
    } else {
      startRecording();
    }
  }, [recording, startRecording, stopRecording]);

  if (!started) {
    return <SplashScreen onStart={handleStart} loading={loading} error={error} />;
  }

  return (
    <div className="eden-app">
      <AvatarView
        frame={currentFrame}
        useCssFallback={useCssFallback}
        speaking={loading}
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
        cssFallback={useCssFallback}
      />
    </div>
  );
}
