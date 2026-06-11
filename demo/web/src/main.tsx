import "@fontsource-variable/jetbrains-mono";
import { StrictMode } from "react";
import { createRoot } from "react-dom/client";

import { App } from "./App";
import { LiveProvider } from "./live/useLive";
import "./styles/base.css";
import "./styles/ui.css";

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <LiveProvider>
      <App />
    </LiveProvider>
  </StrictMode>,
);
