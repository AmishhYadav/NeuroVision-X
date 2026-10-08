import { navigateTo } from "../../lib/navigate";
import { Brain } from "lucide-react";

export function Landing() {
  return (
    <div className="min-h-screen bg-surface-page text-text-primary flex flex-col font-sans">
      {/* Navbar */}
      <nav className="flex items-center justify-between p-6 glass-panel rounded-none border-b border-white/5 shadow-none">
        <div className="flex items-center gap-2">
          <Brain className="w-8 h-8 text-brand-primary" />
          <span className="font-heading font-bold text-xl tracking-tight">NeuroVision-X</span>
        </div>
        <div className="flex gap-4">
          <button
            onClick={() => navigateTo("/app")}
            className="btn-secondary"
          >
            Open Viewer
          </button>
          <button
            onClick={() => navigateTo("/clinical")}
            className="btn-primary"
          >
            Clinical Ingest
          </button>
        </div>
      </nav>

      {/* Hero Section */}
      <main className="flex-1 flex flex-col items-center justify-center p-8 relative overflow-hidden">
        {/* Abstract Background Elements */}
        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[800px] h-[800px] bg-brand-primary/20 rounded-full blur-[120px] pointer-events-none" />
        <div className="absolute top-1/4 right-1/4 w-[400px] h-[400px] bg-brand-teal/20 rounded-full blur-[100px] pointer-events-none" />
        <div className="absolute bottom-1/4 left-1/4 w-[500px] h-[500px] bg-brand-pink/10 rounded-full blur-[120px] pointer-events-none" />

        <div className="max-w-4xl text-center space-y-8 relative z-10 glass-panel p-12 hero-rise">
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full border border-brand-teal/30 bg-brand-teal/10 text-brand-teal font-mono text-sm mb-4">
            <span className="w-2 h-2 rounded-full bg-brand-teal animate-pulse" />
            3D Anatomical Brain Modeling Platform
          </div>
          
          <h1 className="text-6xl font-heading font-extrabold tracking-tight leading-tight text-transparent bg-clip-text bg-gradient-to-r from-white via-white to-white/60">
            Surgical Precision
            <br />
            <span className="text-brand-primary">Neuro-Oncology</span> Platform
          </h1>
          
          <p className="text-lg text-text-secondary max-w-2xl mx-auto leading-relaxed">
            Advanced 3D brain tumor segmentation, uncertainty visualization, and automated clinical reporting.
            Empowering neurosurgeons with AI-driven digital twins for pre-operative planning and analysis.
          </p>

          <div className="flex items-center justify-center gap-6 pt-4">
            <button
              onClick={() => navigateTo("/app")}
              className="btn-primary text-lg px-8 py-4 flex items-center gap-2"
            >
              <Brain className="w-5 h-5" />
              Launch Digital Twin Studio
            </button>
          </div>
        </div>

        {/* Feature grid */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 max-w-6xl w-full mt-24 relative z-10">
          <div className="glass-panel p-6 flex flex-col gap-4 transition-transform hover:-translate-y-1">
            <div className="w-12 h-12 rounded-full bg-brand-primary/20 flex items-center justify-center text-brand-primary">
              <Brain />
            </div>
            <h3 className="text-xl font-heading font-bold text-white">3D Digital Twin</h3>
            <p className="text-text-secondary text-sm leading-relaxed">
              Real-time isosurface generation mapped directly to patient MRI scans. Visualize necrotic, edema, and enhancing regions in full 3D context.
            </p>
          </div>
          <div className="glass-panel p-6 flex flex-col gap-4 transition-transform hover:-translate-y-1">
            <div className="w-12 h-12 rounded-full bg-brand-teal/20 flex items-center justify-center text-brand-teal">
              <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" /></svg>
            </div>
            <h3 className="text-xl font-heading font-bold text-white">Uncertainty Calibration</h3>
            <p className="text-text-secondary text-sm leading-relaxed">
              Examine model confidence gradients and entropy maps to identify boundary ambiguities before surgery.
            </p>
          </div>
          <div className="glass-panel p-6 flex flex-col gap-4 transition-transform hover:-translate-y-1">
            <div className="w-12 h-12 rounded-full bg-brand-pink/20 flex items-center justify-center text-brand-pink">
              <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" /></svg>
            </div>
            <h3 className="text-xl font-heading font-bold text-white">Structured Reporting</h3>
            <p className="text-text-secondary text-sm leading-relaxed">
              Automated neuro-anatomical reports highlighting tumor volume metrics, affected tracts, and plain-language interpretations.
            </p>
          </div>
        </div>
      </main>
    </div>
  );
}
