import { useRef, useState } from "react";

import type { TurnStakes } from "../../api/types";
import { ImagePlaceholder } from "../../components/ImagePlaceholder";

export type ChatInputProps = {
  audience: string;
  running: boolean;
  onSend: (input: {
    message: string;
    stakes: TurnStakes;
    attachments?: readonly File[];
  }) => Promise<boolean>;
};

type StagedAttachment = {
  id: string;
  file: File;
};

const BYTES_PER_KIB = 1024;
const BYTES_PER_MIB = BYTES_PER_KIB * BYTES_PER_KIB;

function formatAttachmentBytes(bytes: number): string {
  if (bytes < BYTES_PER_KIB) {
    return `${bytes} B`;
  }
  if (bytes < BYTES_PER_MIB) {
    return `${Math.round(bytes / BYTES_PER_KIB)} KiB`;
  }
  return `${(bytes / BYTES_PER_MIB).toFixed(1)} MiB`;
}

export function ChatInput({ audience, running, onSend }: ChatInputProps) {
  const [input, setInput] = useState("");
  const [stakes, setStakes] = useState<TurnStakes>("low");
  const [staged, setStaged] = useState<StagedAttachment[]>([]);
  const [draggingOver, setDraggingOver] = useState(false);
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const nextAttachmentIdRef = useRef(0);

  const stageFiles = (files: readonly File[] | FileList) => {
    const imageFiles = Array.from(files).filter((file) => file.type.startsWith("image/"));
    if (imageFiles.length === 0) {
      return;
    }

    setStaged((current) => [
      ...current,
      ...imageFiles.map((file) => {
        const id = `staged-${nextAttachmentIdRef.current}`;
        nextAttachmentIdRef.current += 1;
        return { id, file };
      }),
    ]);
  };

  const removeStaged = (id: string) => {
    setStaged((current) => current.filter((attachment) => attachment.id !== id));
  };

  const send = () => {
    const trimmed = input.trim();
    const message = trimmed.length === 0 ? "(attached image)" : trimmed;
    if ((trimmed.length === 0 && staged.length === 0) || running) {
      return;
    }
    void (async () => {
      const accepted = await onSend({
        message,
        stakes,
        ...(staged.length === 0
          ? {}
          : { attachments: staged.map((attachment) => attachment.file) }),
      });
      if (accepted) {
        setInput("");
        setStaged([]);
      }
    })();
  };

  return (
    <div
      className="chat-input-wrap"
      onDragEnter={(event) => {
        if (!running) {
          event.preventDefault();
          setDraggingOver(true);
        }
      }}
      onDragOver={(event) => {
        if (!running) {
          event.preventDefault();
          setDraggingOver(true);
        }
      }}
      onDragLeave={(event) => {
        const nextTarget = event.relatedTarget;
        if (nextTarget instanceof Node && event.currentTarget.contains(nextTarget)) {
          return;
        }
        setDraggingOver(false);
      }}
      onDrop={(event) => {
        event.preventDefault();
        setDraggingOver(false);
        if (!running) {
          stageFiles(event.dataTransfer.files);
        }
      }}
    >
      {draggingOver ? <div className="composer-dropzone">▸ drop to attach</div> : null}
      {staged.length > 0 ? (
        <div className="composer-staged">
          {staged.map((attachment) => (
            <div key={attachment.id} className="staged-att">
              <ImagePlaceholder
                mediaType={attachment.file.type}
                size="xs"
              />
              <div className="meta">
                <span className="h">{attachment.file.name}</span>
                <span>
                  pending · {attachment.file.type || "image"} ·{" "}
                  {formatAttachmentBytes(attachment.file.size)}
                </span>
              </div>
              <button
                className="staged-att-x"
                onClick={() => removeStaged(attachment.id)}
                type="button"
                aria-label={`remove ${attachment.file.name}`}
                disabled={running}
              >
                ×
              </button>
            </div>
          ))}
        </div>
      ) : null}
      <div className="chat-input-bar">
        <span className="prompt">{">"}</span>
        <textarea
          placeholder={running ? "borg is thinking..." : "send a turn"}
          value={input}
          onChange={(event) => setInput(event.target.value)}
          onPaste={(event) => {
            const files = Array.from(event.clipboardData.items)
              .filter((item) => item.kind === "file" && item.type.startsWith("image/"))
              .map((item) => item.getAsFile())
              .filter((file): file is File => file !== null);

            if (files.length > 0) {
              event.preventDefault();
              stageFiles(files);
            }
          }}
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
        <span className="send-hint" aria-hidden="true">
          <span className="kbd">⌘</span>
          <span className="kbd">↵</span>
        </span>
      </div>
      <div className="chat-input-flags">
        <span className="flag">
          <span className="k">--audience</span>{" "}
          <span className="v acc">{audience}</span>
        </span>
        <span className="flag">
          <span className="k">--stakes</span>{" "}
          <select
            value={stakes}
            onChange={(event) => setStakes(event.target.value as TurnStakes)}
            style={{
              background: "transparent",
              color: "var(--text-mute)",
              border: "0",
              fontFamily: "var(--mono)",
              fontSize: "var(--fs-micro)",
              textTransform: "uppercase",
              letterSpacing: "var(--eyebrow-ls)",
              outline: "none",
              cursor: "pointer",
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
        <input
          ref={fileInputRef}
          data-testid="attachment-file-input"
          type="file"
          accept="image/*"
          multiple
          hidden
          onChange={(event) => {
            if (event.currentTarget.files !== null) {
              stageFiles(event.currentTarget.files);
            }
            event.currentTarget.value = "";
          }}
        />
        <button
          className="btn sm ghost"
          onClick={() => fileInputRef.current?.click()}
          disabled={running}
          type="button"
        >
          + attach
        </button>
        <button
          className="btn sm primary"
          onClick={send}
          disabled={running || (input.trim().length === 0 && staged.length === 0)}
          type="button"
        >
          send
        </button>
        <span className="hint">
          <span className="kbd">↵</span> send
          <span className="sep" style={{ color: "var(--text-ghost)", margin: "0 4px" }}>
            ·
          </span>
          <span className="kbd">⇧</span>
          <span className="kbd">↵</span> newline
        </span>
      </div>
    </div>
  );
}
