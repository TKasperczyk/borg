import type { KeyboardEvent } from "react";

const INTERACTIVE_DESCENDANT_SELECTOR =
  "button,a,input,select,textarea,[role='button'],[contenteditable]";

export function isInteractiveDescendantEvent(
  currentTarget: Element,
  target: EventTarget | null,
): boolean {
  const targetElement =
    target instanceof Element
      ? target
      : typeof Node !== "undefined" && target instanceof Node
        ? target.parentElement
        : null;
  if (targetElement === null) {
    return false;
  }

  const interactive = targetElement.closest(INTERACTIVE_DESCENDANT_SELECTOR);
  return (
    interactive !== null && interactive !== currentTarget && currentTarget.contains(interactive)
  );
}

export function activateOnEnterOrSpace<T extends Element>(
  event: KeyboardEvent<T>,
  action: () => void,
): void {
  if (event.key !== "Enter" && event.key !== " ") {
    return;
  }
  if (isInteractiveDescendantEvent(event.currentTarget, event.target)) {
    return;
  }

  event.preventDefault();
  action();
}
