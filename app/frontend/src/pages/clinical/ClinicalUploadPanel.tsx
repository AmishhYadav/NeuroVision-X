import { FileArchive, Play } from "lucide-react";
import { useRef, useState, type DragEvent } from "react";
import { ApiError, ApiUnreachableError, createClinicalJob } from "../../api";

interface ClinicalUploadPanelProps {
  onJobCreated: (jobId: string) => void;
}

/** Bytes as B / KB / MB (1 decimal, base 1024). */
function formatBytes(n: number): string {
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  return `${(n / (1024 * 1024)).toFixed(1)} MB`;
}

/**
 * Drop zone + file card for a raw DICOM study `.zip`.
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
  const [dragging, setDragging] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  function handleDrop(e: DragEvent<HTMLDivElement>) {
    e.preventDefault();
    setDragging(false);
    const files = Array.from(e.dataTransfer.files);
    if (files.length !== 1 || !files[0].name.toLowerCase().endsWith(".zip")) {
      setError("Drop a single .zip file.");
      return;
    }
    setError(null);
    setFile(files[0]);
  }

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
    <div className="flex flex-col gap-4">
      <div
        onDragOver={(e) => {
          e.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={handleDrop}
        className={`flex flex-col items-center gap-4 rounded-[10px] border-2 border-dashed px-6 py-14 text-center transition-colors ${
          dragging ? "border-brand-primary bg-brand-primary/5" : "border-brand-primary/40"
        }`}
      >
        <span className="flex h-16 w-16 items-center justify-center rounded-xl border border-brand-primary/40 bg-brand-primary/10">
          <FileArchive className="h-8 w-8 text-brand-primary" aria-hidden="true" />
        </span>
        <div>
          <p className="font-heading text-xl">Drop a DICOM study (.zip)</p>
          <p className="mt-1 text-sm text-text-secondary">One patient's raw DICOM study, zipped.</p>
        </div>
        <button
          type="button"
          onClick={() => inputRef.current?.click()}
          className="chip cursor-pointer hover:border-brand-primary hover:text-text-primary"
        >
          or browse
        </button>
        <input
          ref={inputRef}
          type="file"
          accept=".zip"
          aria-label="DICOM study (.zip)"
          className="hidden"
          onChange={(e) => {
            setFile(e.target.files?.[0] ?? null);
            setError(null);
          }}
        />
      </div>

      {file && (
        <div className="glass-panel flex flex-wrap items-center gap-4 p-4">
          <span className="flex h-11 w-11 shrink-0 items-center justify-center rounded-lg border border-surface-seam bg-surface-raised">
            <FileArchive className="h-5 w-5 text-text-secondary" aria-hidden="true" />
          </span>
          <div className="min-w-0 flex-1">
            <p className="truncate font-mono text-sm text-text-primary" title={file.name}>
              {file.name}
            </p>
            <p className="font-mono text-xs text-text-dim">{formatBytes(file.size)}</p>
          </div>
          <button type="button" className="btn-primary" disabled={submitting} onClick={handleUpload}>
            <Play className="h-4 w-4" aria-hidden="true" />
            {submitting ? "Uploading…" : "Start pipeline"}
          </button>
        </div>
      )}

      {error && (
        <p
          role="alert"
          className="glass-panel border-gate-refuse/40 px-4 py-3 text-xs leading-relaxed text-text-primary"
        >
          {error}
        </p>
      )}
    </div>
  );
}
