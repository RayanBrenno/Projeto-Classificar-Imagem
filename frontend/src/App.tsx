import { useCallback, useRef, useState } from "react";
import { ConfidenceBar } from "./ConfidenceBar";

type Prediction = {
  class: string;
  confidence: number;
  top_k: [string, number][];
};

type Status = "idle" | "loading" | "done" | "error";

const API = import.meta.env.VITE_API_URL ?? "/api";

export function App() {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [status, setStatus] = useState<Status>("idle");
  const [result, setResult] = useState<Prediction | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const selectFile = useCallback((f: File | undefined) => {
    if (!f) return;
    if (!f.type.startsWith("image/")) {
      setError("Arquivo inválido — envie uma imagem.");
      return;
    }
    setFile(f);
    setPreview(URL.createObjectURL(f));
    setResult(null);
    setError(null);
    setStatus("idle");
  }, []);

  const reset = useCallback(() => {
    setFile(null);
    setPreview(null);
    setResult(null);
    setError(null);
    setStatus("idle");
    if (inputRef.current) inputRef.current.value = "";
  }, []);

  const classify = useCallback(async () => {
    if (!file) return;
    setStatus("loading");
    setError(null);
    try {
      const fd = new FormData();
      fd.append("image", file);
      const r = await fetch(`${API}/classify`, { method: "POST", body: fd });
      if (!r.ok) {
        const body = await r.json().catch(() => ({ message: r.statusText }));
        throw new Error(body.message ?? `HTTP ${r.status}`);
      }
      setResult(await r.json());
      setStatus("done");
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      setStatus("error");
    }
  }, [file]);

  const onDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setIsDragging(false);
      selectFile(e.dataTransfer.files?.[0]);
    },
    [selectFile],
  );

  return (
    <main className="min-h-screen bg-slate-950 flex items-start justify-center pt-20 pb-24 px-5">
      {/* dot grid background */}
      <div
        className="fixed inset-0 pointer-events-none opacity-[0.035]"
        style={{
          backgroundImage: "radial-gradient(circle, #94a3b8 1px, transparent 1px)",
          backgroundSize: "28px 28px",
        }}
      />

      <div className="relative w-full max-w-2xl">
        {/* Header */}
        <header className="mb-9">
          <div className="flex items-center gap-2 mb-3">
            <span className="relative flex h-2.5 w-2.5">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-60" />
              <span className="relative inline-flex rounded-full h-2.5 w-2.5 bg-emerald-400" />
            </span>
            <span className="text-xs font-mono text-emerald-400 tracking-[0.2em] uppercase">
              sistema ativo
            </span>
          </div>
          <h1 className="text-[2.6rem] leading-none font-display font-bold text-white tracking-tight">
            Image Classifier
          </h1>
          <p className="mt-2 text-sm font-mono text-slate-500 tracking-widest">
            CIFAR-10 · ResNet18 · 87.76% acc
          </p>
        </header>

        {/* Card */}
        <div className="rounded-2xl bg-slate-900/70 border border-slate-800/60 p-6 backdrop-blur-sm">

          {/* Drop zone */}
          <label
            className={`relative block rounded-xl overflow-hidden cursor-pointer transition-all duration-200 ${
              isDragging
                ? "border border-violet-500/70 bg-violet-500/5 shadow-[0_0_24px_-4px_rgba(139,92,246,0.3)]"
                : preview
                ? "border border-slate-800"
                : "border border-dashed border-slate-700 bg-slate-950/60 hover:border-violet-500/50 hover:bg-violet-500/5 hover:shadow-[0_0_20px_-6px_rgba(139,92,246,0.25)]"
            }`}
            onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
            onDragLeave={() => setIsDragging(false)}
            onDrop={onDrop}
          >
            {preview ? (
              <img
                src={preview}
                alt="preview"
                className="w-full object-contain max-h-80"
              />
            ) : (
              <div className="flex flex-col items-center justify-center gap-4 py-20">
                <div className="w-14 h-14 rounded-xl bg-slate-800/80 border border-slate-700/80 flex items-center justify-center">
                  <svg
                    className="w-6 h-6 text-slate-400"
                    fill="none"
                    stroke="currentColor"
                    strokeWidth={1.5}
                    viewBox="0 0 24 24"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5m-13.5-9L12 3m0 0l4.5 4.5M12 3v13.5"
                    />
                  </svg>
                </div>
                <div className="text-center space-y-1">
                  <p className="text-base font-semibold text-slate-300">
                    Arraste uma imagem
                  </p>
                  <p className="text-sm text-slate-600 font-mono">
                    ou clique para selecionar · max 5MB
                  </p>
                </div>
              </div>
            )}

            {isDragging && (
              <div
                className="absolute left-0 right-0 h-px bg-gradient-to-r from-transparent via-violet-400 to-transparent"
                style={{ animation: "scan 1s linear infinite" }}
              />
            )}

            <input
              ref={inputRef}
              type="file"
              accept="image/*"
              hidden
              onChange={(e) => selectFile(e.target.files?.[0])}
            />
          </label>

          {/* Actions */}
          <div className="flex gap-3 mt-5">
            <button
              type="button"
              onClick={reset}
              className="px-5 py-2.5 text-sm font-mono text-slate-500 rounded-lg border border-slate-800 hover:border-slate-700 hover:text-slate-300 transition-all duration-150"
            >
              limpar
            </button>
            <button
              type="button"
              onClick={classify}
              disabled={!file || status === "loading"}
              className="flex-1 py-2.5 text-base font-display font-semibold text-white rounded-lg bg-violet-600 hover:bg-violet-500 active:scale-[0.98] disabled:opacity-30 disabled:cursor-not-allowed transition-all duration-150 tracking-wide shadow-[0_0_20px_-4px_rgba(139,92,246,0.5)] hover:shadow-[0_0_28px_-4px_rgba(139,92,246,0.7)]"
            >
              {status === "loading" ? (
                <span className="flex items-center justify-center gap-2">
                  <span
                    className="w-4 h-4 rounded-full border-2 border-white/25 border-t-white"
                    style={{ animation: "spin 0.7s linear infinite" }}
                  />
                  analisando...
                </span>
              ) : (
                "classificar →"
              )}
            </button>
          </div>

          {/* Error */}
          {status === "error" && error && (
            <div className="mt-5 px-4 py-3 rounded-lg bg-red-500/8 border border-red-500/25 text-sm font-mono text-red-400">
              ⚠ {error}
            </div>
          )}

          {/* Result */}
          {result && (
            <div className="animate-fade-up mt-7 pt-6 border-t border-slate-800/60">
              <div className="flex items-end justify-between mb-6">
                <div>
                  <p className="text-xs font-mono text-slate-600 tracking-[0.2em] uppercase mb-1.5">
                    resultado
                  </p>
                  <p className="text-3xl font-display font-bold text-white tracking-tight capitalize">
                    {result.class}
                  </p>
                </div>
                <span className="text-[2.75rem] leading-none font-mono font-light text-violet-300 tabular-nums">
                  {Math.round(result.confidence * 100)}%
                </span>
              </div>

              <div className="space-y-3">
                {result.top_k.map(([cls, prob], i) => (
                  <ConfidenceBar key={cls} label={cls} value={prob} rank={i} />
                ))}
              </div>
            </div>
          )}
        </div>

        <p className="text-center text-lg font-mono text-slate-700 mt-6 tracking-wider">
          plane · car · bird · cat · deer · dog · frog · horse · ship · truck
        </p>
      </div>
    </main>
  );
}
