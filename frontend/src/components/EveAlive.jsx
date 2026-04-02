import React, { useState, useEffect, useRef, useCallback } from "react";

/**
 * EveAlive — Always-alive idle animation for Eve's avatar.
 *
 * She breathes, blinks, micro-moves, and holds direct eye contact from
 * frame zero. No click-to-start. When `frame` prop supplies pipeline
 * data she switches to real animation, then smoothly returns to idle.
 *
 * Props:
 *   speaking : boolean         — true while waiting for / playing response
 *   frame    : string | null   — base64 JPEG from the animation pipeline
 */

// ── Utility: seeded-ish noise for organic feel ──────────────────────
function noise(t, freq, amp) {
  // Sum two sine waves at incommensurate frequencies for pseudo-random feel
  return (
    Math.sin(t * freq) * amp +
    Math.sin(t * freq * 1.7 + 2.3) * amp * 0.4 +
    Math.sin(t * freq * 0.6 + 5.1) * amp * 0.2
  );
}

// ── Constants ───────────────────────────────────────────────────────
const BLINK_DURATION_MS = 170;
const DOUBLE_BLINK_GAP_MS = 120;
const DOUBLE_BLINK_CHANCE = 0.1;

// Eye region positions relative to the image (fraction of height/width)
const EYE_TOP = 0.32;
const EYE_BOTTOM = 0.42;
const EYE_LEFT = 0.2;
const EYE_RIGHT = 0.8;

export default function EveAlive({ speaking = false, frame = null }) {
  const rafRef = useRef(null);
  const startTimeRef = useRef(null);
  const containerRef = useRef(null);
  const imgRef = useRef(null);

  // ── Animation state (refs to avoid re-renders) ──────────────────
  const animState = useRef({
    // Transform values applied each frame
    breatheScale: 1,
    breatheY: 0,
    headRotateX: 0, // nod
    headRotateY: 0, // turn
    headRotateZ: 0, // tilt
    browRaise: 0,
    jawOpen: 0,

    // Blink state
    blinkOpacity: 0,
    nextBlinkAt: 2000 + Math.random() * 2000, // first blink 2-4s in
    blinkStartedAt: -1,
    isDoubleBlink: false,
    doubleBlinkPhase: 0, // 0 = first blink, 1 = gap, 2 = second blink

    // Micro-expression timers
    nextBrowAt: 8000 + Math.random() * 7000,
    nextJawAt: 5000 + Math.random() * 5000,
  });

  // ── Transition blend (0 = idle, 1 = pipeline frame) ─────────────
  const [blend, setBlend] = useState(0);
  const blendTarget = useRef(0);
  const blendRef = useRef(0);

  // Track whether we have a real frame
  const hasFrame = frame !== null;

  useEffect(() => {
    blendTarget.current = hasFrame ? 1 : 0;
  }, [hasFrame]);

  // ── Core animation loop ─────────────────────────────────────────
  const tick = useCallback((timestamp) => {
    if (startTimeRef.current === null) startTimeRef.current = timestamp;
    const t = timestamp - startTimeRef.current; // ms elapsed
    const s = animState.current;

    // ── Blend interpolation (smooth transition to/from pipeline) ───
    const blendSpeed = 0.04;
    blendRef.current += (blendTarget.current - blendRef.current) * blendSpeed;
    if (Math.abs(blendRef.current - blendTarget.current) < 0.005) {
      blendRef.current = blendTarget.current;
    }
    // Only update React state when blend changes meaningfully
    const roundedBlend = Math.round(blendRef.current * 100) / 100;
    setBlend((prev) => (prev !== roundedBlend ? roundedBlend : prev));

    // Skip idle computation when fully in pipeline mode
    if (blendRef.current >= 0.99) {
      rafRef.current = requestAnimationFrame(tick);
      return;
    }

    // ── 1. BREATHING ──────────────────────────────────────────────
    // Primary: ~3.8s cycle with organic irregularity
    const breatheCycle = t / 1000; // seconds
    const primaryBreath = Math.sin(breatheCycle * (2 * Math.PI / 3.8));
    const breatheNoise = noise(breatheCycle, 1.1, 0.15);
    s.breatheScale = 1 + (primaryBreath + breatheNoise) * 0.0035;
    s.breatheY = (primaryBreath + breatheNoise * 0.5) * 1.2;

    // ── 2. BLINKING ──────────────────────────────────────────────
    if (s.blinkStartedAt < 0) {
      // Not currently blinking — check if it is time
      if (t >= s.nextBlinkAt) {
        s.blinkStartedAt = t;
        s.isDoubleBlink = Math.random() < DOUBLE_BLINK_CHANCE;
        s.doubleBlinkPhase = 0;
      }
      s.blinkOpacity = 0;
    } else {
      // Currently in a blink sequence
      const blinkElapsed = t - s.blinkStartedAt;

      if (s.doubleBlinkPhase === 0) {
        // First blink
        if (blinkElapsed < BLINK_DURATION_MS) {
          // Ease in-out: fast close, slightly slower open
          const p = blinkElapsed / BLINK_DURATION_MS;
          s.blinkOpacity = Math.sin(p * Math.PI);
        } else if (s.isDoubleBlink) {
          s.doubleBlinkPhase = 1;
          s.blinkStartedAt = t; // reset for gap timing
          s.blinkOpacity = 0;
        } else {
          // Single blink done
          s.blinkOpacity = 0;
          s.blinkStartedAt = -1;
          s.nextBlinkAt = t + 3000 + Math.random() * 3000;
        }
      } else if (s.doubleBlinkPhase === 1) {
        // Gap between double blinks
        const gapElapsed = t - s.blinkStartedAt;
        s.blinkOpacity = 0;
        if (gapElapsed >= DOUBLE_BLINK_GAP_MS) {
          s.doubleBlinkPhase = 2;
          s.blinkStartedAt = t;
        }
      } else if (s.doubleBlinkPhase === 2) {
        // Second blink
        const blink2Elapsed = t - s.blinkStartedAt;
        if (blink2Elapsed < BLINK_DURATION_MS) {
          const p = blink2Elapsed / BLINK_DURATION_MS;
          s.blinkOpacity = Math.sin(p * Math.PI);
        } else {
          s.blinkOpacity = 0;
          s.blinkStartedAt = -1;
          s.nextBlinkAt = t + 3000 + Math.random() * 3000;
        }
      }
    }

    // ── 3. DIRECT GAZE + MICRO HEAD MOVEMENT ─────────────────────
    // Very subtle head micro-movements to break uncanny stillness
    s.headRotateX = noise(breatheCycle, 0.4, 0.18); // nod +-0.18 deg
    s.headRotateY = noise(breatheCycle, 0.3, 0.22); // turn +-0.22 deg
    s.headRotateZ = noise(breatheCycle, 0.2, 0.12); // tilt +-0.12 deg

    // ── 4. MICRO-EXPRESSIONS ─────────────────────────────────────
    // Eyebrow raise (every 8-15s, lasts ~1.5s)
    if (t >= s.nextBrowAt) {
      s.nextBrowAt = t + 8000 + Math.random() * 7000;
    }
    const browCyclePos = (s.nextBrowAt - t) / 1500;
    if (browCyclePos > 0 && browCyclePos < 1) {
      s.browRaise = Math.sin(browCyclePos * Math.PI) * 0.6; // px shift up
    } else {
      s.browRaise = 0;
    }

    // Subtle jaw movement (as if about to speak)
    if (t >= s.nextJawAt) {
      s.nextJawAt = t + 5000 + Math.random() * 5000;
    }
    const jawCyclePos = (s.nextJawAt - t) / 800;
    if (jawCyclePos > 0 && jawCyclePos < 1) {
      s.jawOpen = Math.sin(jawCyclePos * Math.PI) * 0.3; // px
    } else {
      s.jawOpen = 0;
    }

    // ── Speaking amplification ────────────────────────────────────
    let speakJitter = 0;
    if (speaking && blendRef.current < 0.5) {
      // When speaking but no pipeline frames, add more jaw/breath motion
      speakJitter = Math.sin(t * 0.025) * 0.003;
      s.jawOpen += Math.abs(Math.sin(t * 0.018)) * 1.2;
    }

    // ── Apply transforms to DOM directly (no React re-render) ────
    if (imgRef.current) {
      const scale = s.breatheScale + speakJitter;
      const tx = s.headRotateY * 0.3; // subtle horizontal shift
      const ty = s.breatheY + s.browRaise * -0.3 + s.jawOpen * 0.15;
      imgRef.current.style.transform =
        `scale(${scale}) ` +
        `translate(${tx}px, ${ty}px) ` +
        `rotateX(${s.headRotateX}deg) ` +
        `rotateY(${s.headRotateY}deg) ` +
        `rotateZ(${s.headRotateZ}deg)`;
    }

    // Blink overlay
    const blinkEl = containerRef.current?.querySelector(".eve-blink-overlay");
    if (blinkEl) {
      blinkEl.style.opacity = s.blinkOpacity;
    }

    // Jaw overlay (subtle shadow under mouth)
    const jawEl = containerRef.current?.querySelector(".eve-jaw-overlay");
    if (jawEl) {
      jawEl.style.transform = `translateY(${s.jawOpen}px)`;
      jawEl.style.opacity = s.jawOpen > 0.1 ? 0.15 : 0;
    }

    rafRef.current = requestAnimationFrame(tick);
  }, [speaking]);

  // ── Start/stop animation loop ──────────────────────────────────
  useEffect(() => {
    rafRef.current = requestAnimationFrame(tick);
    return () => {
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
    };
  }, [tick]);

  // ── Render ─────────────────────────────────────────────────────
  const idleOpacity = 1 - blend;
  const frameOpacity = blend;

  return (
    <div className="eve-alive-container" ref={containerRef}>
      {/* Idle layer — always-alive Eve */}
      <div
        className="eve-alive-idle-layer"
        style={{ opacity: idleOpacity, visibility: idleOpacity < 0.01 ? "hidden" : "visible" }}
      >
        <div className="eve-alive-image-wrapper">
          <img
            ref={imgRef}
            src="/eve-NATURAL.png"
            alt="Eve"
            className={`eve-alive-image ${speaking ? "eve-alive-speaking" : ""}`}
            draggable={false}
          />

          {/* Blink overlay — positioned over the eye region */}
          <div
            className="eve-blink-overlay"
            style={{
              top: `${EYE_TOP * 100}%`,
              height: `${(EYE_BOTTOM - EYE_TOP) * 100}%`,
              left: `${EYE_LEFT * 100}%`,
              width: `${(EYE_RIGHT - EYE_LEFT) * 100}%`,
              opacity: 0,
            }}
          />

          {/* Jaw micro-movement overlay */}
          <div
            className="eve-jaw-overlay"
            style={{
              top: "68%",
              height: "10%",
              left: "30%",
              width: "40%",
              opacity: 0,
            }}
          />

          {/* Ambient aura */}
          <div className={`eve-alive-aura ${speaking ? "speaking" : ""}`} />
        </div>
      </div>

      {/* Pipeline frame layer — fades in when real frames arrive */}
      {frame && (
        <div
          className="eve-alive-frame-layer"
          style={{ opacity: frameOpacity, visibility: frameOpacity < 0.01 ? "hidden" : "visible" }}
        >
          <img
            className="eve-alive-frame"
            src={`data:image/jpeg;base64,${frame}`}
            alt="Eve (animated)"
            draggable={false}
          />
        </div>
      )}
    </div>
  );
}
