import { useCallback, useEffect, useState } from "react";

import { routeIdForDigitKey, type RouteId } from "../routes";

type UsePaletteHotkeyOptions = {
  disabled?: boolean;
  onRouteChord?: (route: RouteId) => void;
  onHelpChord?: () => void;
};

export type PaletteHotkeyState = {
  open: boolean;
  setOpen: (open: boolean) => void;
  close: () => void;
};

export function isEditableShortcutTarget(target: EventTarget | null): boolean {
  if (
    target instanceof HTMLInputElement ||
    target instanceof HTMLTextAreaElement ||
    target instanceof HTMLSelectElement
  ) {
    return true;
  }
  return target instanceof HTMLElement && target.isContentEditable;
}

function modalOpen(): boolean {
  return document.querySelector('[role="dialog"]') !== null;
}

export function usePaletteHotkey({
  disabled = false,
  onRouteChord,
  onHelpChord,
}: UsePaletteHotkeyOptions = {}): PaletteHotkeyState {
  const [open, setOpenState] = useState(false);

  const setOpen = useCallback((nextOpen: boolean) => {
    setOpenState(nextOpen);
  }, []);

  const close = useCallback(() => {
    setOpenState(false);
  }, []);

  useEffect(() => {
    if (disabled) {
      return;
    }

    function onKeyDown(event: KeyboardEvent): void {
      if (
        event.code === "Slash" &&
        event.shiftKey &&
        !event.altKey &&
        !event.ctrlKey &&
        !event.metaKey
      ) {
        if (
          onHelpChord !== undefined &&
          !open &&
          !modalOpen() &&
          !isEditableShortcutTarget(event.target)
        ) {
          event.preventDefault();
          onHelpChord();
        }
        return;
      }

      if (event.altKey && !event.ctrlKey && !event.metaKey && !event.shiftKey) {
        const route = routeIdForDigitKey(event.code);
        if (
          route !== null &&
          onRouteChord !== undefined &&
          !open &&
          !modalOpen() &&
          !isEditableShortcutTarget(event.target)
        ) {
          event.preventDefault();
          onRouteChord(route);
        }
        return;
      }

      if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "k") {
        event.preventDefault();
        setOpenState((current) => !current);
      }
    }

    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [disabled, onHelpChord, onRouteChord, open]);

  return { open, setOpen, close };
}
