import { useCallback, useEffect, useState } from "react";

type UsePaletteHotkeyOptions = {
  disabled?: boolean;
};

export type PaletteHotkeyState = {
  open: boolean;
  setOpen: (open: boolean) => void;
  close: () => void;
};

export function usePaletteHotkey({
  disabled = false,
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
      if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "k") {
        event.preventDefault();
        setOpenState((current) => !current);
      }
    }

    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [disabled]);

  return { open, setOpen, close };
}
