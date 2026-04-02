import React, { useState, useEffect, useRef } from "react";

export default function AvatarView({ frame, useCssFallback, speaking }) {
  const [breathePhase, setBreathePhase] = useState(0);
  const breatheTimerRef = useRef(null);

  // Only run breathing animation when in CSS fallback mode
  useEffect(() => {
    if (useCssFallback) {
      breatheTimerRef.current = setInterval(() => {
        setBreathePhase((p) => p + 1);
      }, 50);
    } else {
      if (breatheTimerRef.current) {
        clearInterval(breatheTimerRef.current);
        breatheTimerRef.current = null;
      }
    }
    return () => {
      if (breatheTimerRef.current) clearInterval(breatheTimerRef.current);
    };
  }, [useCssFallback]);

  // If we have real pipeline frames, show them
  if (frame && !useCssFallback) {
    return (
      <div className="avatar-container">
        <img
          className="avatar-frame"
          src={`data:image/jpeg;base64,${frame}`}
          alt="Eve"
          draggable={false}
        />
      </div>
    );
  }

  // Eve's reference image with CSS animation (breathing, speaking)
  const breatheScale = 1.0 + Math.sin(breathePhase * 0.03) * 0.004;
  const breatheY = Math.sin(breathePhase * 0.02) * 1.5;
  const speakScale = speaking ? 1.0 + Math.sin(breathePhase * 0.15) * 0.003 : 0;

  return (
    <div className="avatar-container">
      <div className="avatar-eve-wrapper">
        <img
          src="/eve-NATURAL.png"
          alt="Eve"
          className={`avatar-eve-live ${speaking ? "speaking" : ""}`}
          style={{
            transform: `scale(${breatheScale + speakScale}) translateY(${breatheY}px)`,
          }}
          draggable={false}
        />
        <div className={`avatar-eve-aura ${speaking ? "speaking" : ""}`} />
      </div>
    </div>
  );
}
