import "@fontsource-variable/jetbrains-mono";
import { StrictMode } from "react";
import { createRoot } from "react-dom/client";

import { App } from "./App";
import { LiveProvider } from "./live/useLive";
import { StateProvider } from "./state/app-state";
import { MoodProvider } from "./state/mood";
import "./styles/base.css";
import "./styles/ui.css";

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <LiveProvider>
      <StateProvider>
        <MoodProvider>
          <App />
        </MoodProvider>
      </StateProvider>
    </LiveProvider>
  </StrictMode>,
);
