import { useState } from "react";
import { ApiError, ApiUnreachableError, createClinicalJob } from "../../api";

interface ClinicalUploadPanelProps {
  onJobCreated: (jobId: string) => void;
}

/**
 * File picker + upload button for a raw DICOM study `.zip`.
 *
 * A rejected upload (empty file, oversized, not a zip, a zip-slip attempt)
 * never creates a job at all - the backend answers with a plain 400 before
 * anything is queued - so the only thing to show here on failure is the
 * error message itself, not a job state.
 */
export function ClinicalUploadPanel({ onJobCreated }: ClinicalUploadPanelProps) {
  const [file, setFile] = useState<File | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function handleUpload() {
    if (!file || submitting) return;
    setSubmitting(true);
    setError(null);
    try {
      const job = await createClinicalJob(file);
      onJobCreated(job.job_id);
    } catch (err) {
      if (err instanceof ApiUnreachableError) {
        setError(
          "No response from the API. Start it with `uvicorn app.backend.main:app --reload`.",
        );
      } else if (err instanceof ApiError) {
        setError(err.message);
      } else {
        setError(err instanceof Error ? err.message : "Upload failed.");
      }
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <div className="mx-auto flex w-full max-w-lg flex-col gap-4 border border-surface-seam bg-surface-panel p-6">
      <div>
        <p className="eyebrow">Clinical upload</p>
        <h2 className="mt-1 font-condensed text-2xl text-text-primary">Upload a DICOM study</h2>
        <p className="mt-2 font-mono text-xs leading-relaxed text-text-secondary">
          A .zip of one patient's raw DICOM study. It runs through ingest, input QC, clinical
          preprocessing and segmentation - and is declined outright if any gate along the way
          determines it cannot be handled safely.
        </p>
      </div>

      <label className="flex flex-col gap-2">
        <span className="eyebrow">DICOM study (.zip)</span>
        <input
          type="file"
          accept=".zip"
          onChange={(e) => setFile(e.target.files?.[0] ?? null)}
          className="font-mono text-xs text-text-secondary file:mr-3 file:rounded-sm file:border file:border-surface-seam file:bg-surface-raised file:px-3 file:py-1.5 file:font-mono file:text-xs file:text-text-primary hover:file:border-text-dim"
        />
      </label>

      <button
        type="button"
        disabled={!file || submitting}
        onClick={handleUpload}
        className={`self-start rounded-sm border px-4 py-2 font-condensed text-xs tracking-[0.1em] uppercase transition-colors duration-[120ms] ${
          !file || submitting
            ? "cursor-not-allowed border-surface-seam text-text-dim"
            : "border-text-primary text-text-primary hover:bg-surface-raised"
        }`}
      >
        {submitting ? "Uploading…" : "Upload study"}
      </button>

      {error && (
        <p
          role="alert"
          className="border border-surface-seam bg-surface-raised px-3 py-2 font-mono text-xs leading-relaxed text-text-primary"
        >
          {error}
        </p>
      )}
    </div>
  );
}
