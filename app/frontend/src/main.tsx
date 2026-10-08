import { lazy, StrictMode, Suspense, useEffect, useState, type ReactNode } from "react";
import { createRoot } from "react-dom/client";
import "./index.css";

// Each page is its own chunk: the landing page never downloads the viewer's
// 3D code and vice versa.
const App = lazy(() => import("./App.tsx"));
const Landing = lazy(() => import("./pages/landing/Landing.tsx").then((m) => ({ default: m.Landing })));
const ClinicalPage = lazy(() =>
  import("./pages/clinical/ClinicalPage.tsx").then((m) => ({ default: m.ClinicalPage })),
);
const ReportPage = lazy(() =>
  import("./pages/report/ReportPage.tsx").then((m) => ({ default: m.ReportPage })),
);

// Four real destinations, no router dependency: the landing page at "/",
// the precomputed-case viewer at "/app" (unchanged), the live clinical
// upload page at "/clinical", and a printable plain-language report at
// "/report/{case_id}". Plain pathname + pushState so all four are real,
// bookmarkable, back-button-safe URLs - see Landing.tsx's ViewerLink, which
// navigates the same way, and ClinicalPage's own back-to-landing link.
function Root() {
  const [path, setPath] = useState(window.location.pathname);

  useEffect(() => {
    const onPopState = () => setPath(window.location.pathname);
    window.addEventListener("popstate", onPopState);
    return () => window.removeEventListener("popstate", onPopState);
  }, []);

  let page: ReactNode;
  if (path === "/app") page = <App />;
  else if (path === "/clinical") page = <ClinicalPage />;
  else if (path.startsWith("/report/"))
    page = <ReportPage caseId={decodeURIComponent(path.slice("/report/".length))} />;
  else page = <Landing />;
  return (
    <Suspense
      fallback={
        <div className="flex h-screen items-center justify-center bg-surface-page text-sm text-text-dim">
          Loading…
        </div>
      }
    >
      {page}
    </Suspense>
  );
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
