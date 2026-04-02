import React, { useState, useCallback, useRef } from "react";
import { useWebSocket } from "./hooks/useWebSocket.js";
import { useAudioRecorder } from "./hooks/useAudioRecorder.js";
import SplashScreen from "./components/SplashScreen.jsx";
import AvatarView from "./components/AvatarView.jsx";
import ChatPanel from "./components/ChatPanel.jsx";
import StatusBar from "./components/StatusBar.jsx";
import "./App.css";

const API_BASE = "/api";
const FRAME_INTERVAL_MS = 1000 / 30; // 30fps = ~33.3ms per frame

export default function App() {
  const [started, setStarted] = useState(false);
  const [loading, setLoading] = useState(false);
  const [currentFrame, setCurrentFrame] = useState(null);
  const [messages, setMessages] = useState([]);
  const [pipelineInfo, setPipelineInfo] = useState(null);
  const [error, setError] = useState(null);
  const [useCssFallback, setUseCssFallback] = useState(false);

  // Ref to cancel any in-progress frame animation
  const animationRef = useRef(null);

  // WebSocket for real-time frame streaming
  const { connected, lastFrame } = useWebSocket(
    started ? `ws://${window.location.host}/ws` : null,
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
    const cancel = () => { cancelled = true; };
    animationRef.current = cancel;

    // Prepare audio element (if present)
    let audio = null;
    if (audioB64) {
      const audioBytes = Uint8Array.from(atob(audioB64), (c) => c.charCodeAt(0));
      const blob = new Blob([audioBytes], { type: "audio/wav" });
      audio = new Audio(URL.createObjectURL(blob));
    }

    if (frames && frames.length > 0) {
      setUseCssFallback(false);
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
          // Animation done — switch to CSS fallback for idle breathing
          animationRef.current = null;
          setUseCssFallback(true);
        }
      };

      requestAnimationFrame(tick);
    } else {
      // No frames — just play audio and show CSS fallback
      setUseCssFallback(true);
      if (audio) audio.play().catch(() => {});
    }

    return cancel;
  }, []);

  const handleStart = useCallback(async () => {
    setLoading(true);
    setError(null);
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

      // Play frames + audio simultaneously
      playFramesWithAudio(
        data.frame_count > 0 ? data.frames : null,
        data.audio_b64
      );

      setStarted(true);
    } catch (e) {
      setError(`Connection failed: ${e.message}`);
      setUseCssFallback(true);
      setStarted(true);
    } finally {
      setLoading(false);
    }
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

      // Play frames + audio simultaneously (works for chat responses too)
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

        // Play frames + audio simultaneously for voice responses too
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
