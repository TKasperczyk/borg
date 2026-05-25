import { useState } from "react";

import type { TurnStakes } from "../../api/types";

export type ChatInputProps = {
  audience: string;
  running: boolean;
  onSend: (input: { message: string; stakes: TurnStakes }) => void;
};

export function ChatInput({ audience, running, onSend }: ChatInputProps) {
  const [input, setInput] = useState("");
  const [stakes, setStakes] = useState<TurnStakes>("low");

  const send = () => {
    const message = input.trim();
    if (message.length === 0 || running) {
      return;
    }
    onSend({ message, stakes });
    setInput("");
  };

  return (
    <div className="chat-input-wrap">
      <div className="chat-input-bar">
        <span className="prompt">{">"}</span>
        <textarea
          placeholder={running ? "borg is thinking..." : "send a turn"}
          value={input}
          onChange={(event) => setInput(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === "Enter" && (event.metaKey || event.ctrlKey || !event.shiftKey)) {
              event.preventDefault();
              send();
            }
          }}
          rows={1}
          style={{ height: Math.min(96, Math.max(20, input.split("\n").length * 18)) }}
          disabled={running}
        />
      </div>
      <div className="chat-input-flags">
        <span className="flag">
          <span className="k">--audience</span> <span className="v acc">{audience}</span>
        </span>
        <span className="flag">
          <span className="k">--stakes</span>{" "}
          <select
            value={stakes}
            onChange={(event) => setStakes(event.target.value as TurnStakes)}
            style={{
              background: "transparent",
              color: "var(--text-dim)",
              border: "0",
              fontFamily: "var(--mono)",
              fontSize: "10.5px",
              outline: "none",
              cursor: "pointer"
            }}
            disabled={running}
          >
            <option value="low">low</option>
            <option value="medium">medium</option>
            <option value="high">high</option>
          </select>
        </span>
        <span className="flag">
          <span className="k">--mode</span> <span className="v">auto</span>
        </span>
        <span className="spacer"></span>
        <button className="btn sm primary" onClick={send} disabled={running || input.trim().length === 0} type="button">
          send
        </button>
        <span className="hint">↵ send · ⇧↵ newline</span>
      </div>
    </div>
  );
}
