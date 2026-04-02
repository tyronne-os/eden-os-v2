import React from "react";

export default function AvatarView({ frame, useCssFallback, speaking }) {
  if (useCssFallback || !frame) {
    return (
      <div className="avatar-container">
        <div className={`avatar-fallback ${speaking ? "speaking" : ""}`}>
          <div className="eyes">
            <div className="eye" />
            <div className="eye" />
          </div>
          <div className="mouth" />
        </div>
      </div>
    );
  }

  return (
    <div className="avatar-container">
      <img
        className="avatar-frame"
        src={`data:image/jpeg;base64,${frame}`}
        alt="Eve"
      />
    </div>
  );
}
