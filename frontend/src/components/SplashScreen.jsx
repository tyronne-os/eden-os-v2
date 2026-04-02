import React from "react";

export default function SplashScreen({ onStart, loading, error }) {
  return (
    <div className="splash" onClick={!loading ? onStart : undefined}>
      <div className="splash-avatar">
        <img src="/eve-NATURAL.png" alt="Eve" className="splash-eve" />
        <div className="splash-glow" />
        {loading && <div className="splash-pulse-ring" />}
      </div>
      <h1>EDEN</h1>
      {loading ? (
        <p className="loading-text">Waking Eve...</p>
      ) : (
        <p>Click anywhere to begin</p>
      )}
      {error && <p className="error-text">{error}</p>}
    </div>
  );
}
