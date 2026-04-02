import React, { useState, useRef, useEffect } from "react";

export default function ChatPanel({ messages, onSend, onVoice, recording, loading }) {
  const [input, setInput] = useState("");
  const messagesEnd = useRef(null);

  useEffect(() => {
    messagesEnd.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const handleSubmit = (e) => {
    e.preventDefault();
    if (input.trim() && !loading) {
      onSend(input.trim());
      setInput("");
    }
  };

  return (
    <div className="chat-panel">
      <div className="chat-messages">
        {messages.map((msg, i) => (
          <div key={i} className={`msg ${msg.role}`}>
            {msg.text}
          </div>
        ))}
        <div ref={messagesEnd} />
      </div>
      <form className="chat-input" onSubmit={handleSubmit}>
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Talk to Eve..."
          disabled={loading}
        />
        <button type="submit" disabled={loading || !input.trim()}>
          Send
        </button>
        <button
          type="button"
          className={recording ? "recording" : ""}
          onClick={onVoice}
          disabled={loading}
        >
          {recording ? "Stop" : "Mic"}
        </button>
      </form>
    </div>
  );
}
