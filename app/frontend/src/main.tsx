import { StrictMode, useEffect, useState } from "react";
import { createRoot } from "react-dom/client";
import "./index.css";
import App from "./App.tsx";
import { Landing } from "./pages/landing/Landing.tsx";
import { ClinicalPage } from "./pages/clinical/ClinicalPage.tsx";

// Three real destinations, no router dependency: the landing page at "/",
// the precomputed-case viewer at "/app" (unchanged), and the live clinical
// upload page at "/clinical". Plain pathname + pushState so all three are
// real, bookmarkable, back-button-safe URLs - see Landing.tsx's ViewerLink,
// which navigates the same way, and ClinicalPage's own back-to-landing link.
function Root() {
  const [path, setPath] = useState(window.location.pathname);

  useEffect(() => {
    const onPopState = () => setPath(window.location.pathname);
    window.addEventListener("popstate", onPopState);
    return () => window.removeEventListener("popstate", onPopState);
  }, []);

  if (path === "/app") return <App />;
  if (path === "/clinical") return <ClinicalPage />;
  return <Landing />;
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
