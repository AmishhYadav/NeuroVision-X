import { StrictMode, useEffect, useState } from "react";
import { createRoot } from "react-dom/client";
import "./index.css";
import App from "./App.tsx";
import { Landing } from "./pages/landing/Landing.tsx";

// Two real destinations, no router dependency: the landing page at "/" and
// the clinical viewer at "/app". Plain pathname + pushState so both are
// real, bookmarkable, back-button-safe URLs - see Landing.tsx's ViewerLink,
// which navigates the same way.
function Root() {
  const [path, setPath] = useState(window.location.pathname);

  useEffect(() => {
    const onPopState = () => setPath(window.location.pathname);
    window.addEventListener("popstate", onPopState);
    return () => window.removeEventListener("popstate", onPopState);
  }, []);

  return path === "/app" ? <App /> : <Landing />;
}

const container = document.getElementById("root");
if (!container) {
  throw new Error("Root element #root not found.");
}

createRoot(container).render(
  <StrictMode>
    <Root />
  </StrictMode>,
);
