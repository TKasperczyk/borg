import { useRef, useState } from "react";

import { ImagePlaceholder } from "../../components/ImagePlaceholder";

export type ChatInputProps = {
  audience: string;
  onSend: (input: {
    message: string;
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

export function ChatInput({ audience, onSend }: ChatInputProps) {
  const [input, setInput] = useState("");
  const [staged, setStaged] = useState<StagedAttachment[]>([]);
  const [draggingOver, setDraggingOver] = useState(false);
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const nextAttachmentIdRef = useRef(0);
  const submittingDraftKeysRef = useRef<Set<string>>(new Set());

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

  const draftKey = (message: string, attachments: readonly StagedAttachment[]): string => {
    return `${message}\0${attachments.map((attachment) => attachment.id).join("\0")}`;
  };

  const send = () => {
    const trimmed = input.trim();
    const message = trimmed.length === 0 ? "(attached image)" : trimmed;
    if (trimmed.length === 0 && staged.length === 0) {
      return;
    }
    const stagedSnapshot = staged;
    const key = draftKey(message, stagedSnapshot);
    if (submittingDraftKeysRef.current.has(key)) {
      return;
    }
    submittingDraftKeysRef.current.add(key);
    setInput("");
    setStaged([]);
    void (async () => {
      try {
        const accepted = await onSend({
          message,
          ...(stagedSnapshot.length === 0
            ? {}
            : { attachments: stagedSnapshot.map((attachment) => attachment.file) }),
        });
        if (!accepted) {
          setInput((current) => (current.length === 0 ? input : current));
          setStaged((current) => (current.length === 0 ? stagedSnapshot : current));
        }
      } finally {
        submittingDraftKeysRef.current.delete(key);
      }
    })();
  };

  return (
    <div
      className="chat-input-wrap"
      onDragEnter={(event) => {
        event.preventDefault();
        setDraggingOver(true);
      }}
      onDragOver={(event) => {
        event.preventDefault();
        setDraggingOver(true);
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
        stageFiles(event.dataTransfer.files);
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
          placeholder="send a turn"
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
          style={{ height: Math.min(120, Math.max(18, input.split("\n").length * 18)) }}
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
          type="button"
        >
          + attach
        </button>
        <button
          className="btn sm primary"
          onClick={send}
          disabled={input.trim().length === 0 && staged.length === 0}
          type="button"
        >
          send
        </button>
        <span className="hint">
          <span className="kbd">↵</span> send
          <span className="sep">·</span>
          <span className="kbd">⇧</span>
          <span className="kbd">↵</span> newline
        </span>
      </div>
    </div>
  );
}
