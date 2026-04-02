import { useState, useRef, useCallback } from "react";

export function useAudioRecorder() {
  const [recording, setRecording] = useState(false);
  const mediaRecorder = useRef(null);
  const chunks = useRef([]);

  const startRecording = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const recorder = new MediaRecorder(stream, { mimeType: "audio/webm" });
      mediaRecorder.current = recorder;
      chunks.current = [];

      recorder.ondataavailable = (e) => {
        if (e.data.size > 0) chunks.current.push(e.data);
      };

      recorder.start();
      setRecording(true);
    } catch (err) {
      console.error("Microphone access denied:", err);
    }
  }, []);

  const stopRecording = useCallback(() => {
    return new Promise((resolve) => {
      const recorder = mediaRecorder.current;
      if (!recorder || recorder.state === "inactive") {
        resolve(null);
        return;
      }

      recorder.onstop = () => {
        const blob = new Blob(chunks.current, { type: "audio/webm" });
        // Stop all tracks
        recorder.stream.getTracks().forEach((t) => t.stop());
        setRecording(false);
        resolve(blob);
      };

      recorder.stop();
    });
  }, []);

  return { recording, startRecording, stopRecording };
}
