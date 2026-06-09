export async function copyText(text: string): Promise<void> {
  if (navigator.clipboard?.writeText !== undefined && window.isSecureContext !== false) {
    try {
      await navigator.clipboard.writeText(text);
      return;
    } catch {
      // Fall through to the non-secure/permission fallback.
    }
  }

  copyTextWithExecCommand(text);
}

function copyTextWithExecCommand(text: string): void {
  const textArea = document.createElement("textarea");
  textArea.value = text;
  textArea.setAttribute("readonly", "true");
  textArea.style.position = "fixed";
  textArea.style.left = "-9999px";
  textArea.style.top = "0";
  document.body.appendChild(textArea);
  textArea.focus();
  textArea.select();

  try {
    if (typeof document.execCommand !== "function" || !document.execCommand("copy")) {
      throw new Error("Clipboard copy is not available");
    }
  } finally {
    document.body.removeChild(textArea);
  }
}
