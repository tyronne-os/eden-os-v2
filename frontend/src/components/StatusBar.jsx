import React from "react";

export default function StatusBar({ connected, pipeline, cssFallback }) {
  return (
    <div className="status-bar">
      <span>
        <span className={`status-dot ${connected ? "connected" : "disconnected"}`} />
        {connected ? "Connected" : "Disconnected"}
      </span>
      {pipeline && (
        <span>
          Pipeline: {pipeline.pipeline} | {pipeline.frames} frames | {pipeline.elapsed}s
        </span>
      )}
      {cssFallback && <span>CSS Fallback Active</span>}
      <span style={{ marginLeft: "auto" }}>EDEN OS V2</span>
    </div>
  );
}
