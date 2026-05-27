import { CheckCircle2, Loader2, Sparkles, Clock, Cpu, Shield, Download } from "lucide-react";
import resultImage from "@/assets/tryon-result.jpg";

interface ResultPanelProps {
  status: "idle" | "processing" | "complete";
  model: string;
  elapsed: number;
}

export const ResultPanel = ({ status, model, elapsed }: ResultPanelProps) => {
  return (
    <div className="flex h-full flex-col gap-4">
      <div className="relative overflow-hidden rounded-2xl border border-border bg-card shadow-soft">
        {status === "complete" && (
          <div className="flex items-center justify-between gap-3 bg-gradient-primary px-5 py-3 text-primary-foreground animate-fade-in">
            <div className="flex items-center gap-2">
              <CheckCircle2 className="h-5 w-5" />
              <span className="font-semibold tracking-tight">Try-On Complete</span>
            </div>
            <button className="flex items-center gap-1.5 rounded-lg bg-white/15 px-3 py-1.5 text-xs font-medium backdrop-blur-sm transition-smooth hover:bg-white/25">
              <Download className="h-3.5 w-3.5" />
              Export
            </button>
          </div>
        )}

        <div className="relative aspect-[3/4] w-full bg-secondary/40 grid-pattern">
          {status === "idle" && (
            <div className="absolute inset-0 flex flex-col items-center justify-center gap-4 p-8 text-center">
              <div className="relative">
                <span className="absolute inset-0 animate-pulse-ring rounded-full bg-accent/30" />
                <div className="relative flex h-16 w-16 items-center justify-center rounded-full bg-card text-accent shadow-soft">
                  <Sparkles className="h-7 w-7" />
                </div>
              </div>
              <div className="space-y-1">
                <p className="font-semibold text-foreground">Awaiting inference</p>
                <p className="max-w-xs text-sm text-muted-foreground">
                  Output will appear here after inference. Upload a subject and a garment to begin.
                </p>
              </div>
            </div>
          )}

          {status === "processing" && (
            <div className="absolute inset-0 flex flex-col items-center justify-center gap-5 bg-card/60 backdrop-blur-sm">
              <div
                className="h-2 w-64 max-w-[70%] overflow-hidden rounded-full bg-secondary"
                aria-label="Loading"
              >
                <div className="h-full w-1/3 animate-shimmer rounded-full bg-[linear-gradient(90deg,transparent,hsl(var(--accent)),transparent)] bg-[length:200%_100%]" />
              </div>
              <div className="flex items-center gap-2 text-sm font-medium text-primary">
                <Loader2 className="h-4 w-4 animate-spin" />
                <span>Synthesizing image with {model}…</span>
              </div>
              <p className="text-xs text-muted-foreground">Aligning pose · Warping garment · Blending textures</p>
            </div>
          )}

          {status === "complete" && (
            <img
              src={resultImage}
              alt="Virtual try-on result"
              className="absolute inset-0 h-full w-full object-cover animate-scale-in"
            />
          )}
        </div>
      </div>

      <div className="rounded-2xl border border-border bg-card p-5 shadow-soft">
        <div className="mb-4 flex items-center justify-between">
          <h3 className="text-sm font-semibold text-foreground">Processing Details</h3>
          <span
            className={`rounded-full px-2.5 py-0.5 text-xs font-medium ${
              status === "complete"
                ? "bg-green-100 text-green-700"
                : status === "processing"
                ? "bg-amber-100 text-amber-700"
                : "bg-secondary text-muted-foreground"
            }`}
          >
            {status === "complete" ? "Success" : status === "processing" ? "Running" : "Idle"}
          </span>
        </div>

        <div className="grid grid-cols-3 gap-3">
          <Metric
            icon={<Clock className="h-4 w-4" />}
            label="Processing"
            value={status === "complete" ? `${elapsed.toFixed(1)}s` : status === "processing" ? "…" : "—"}
          />
          <Metric icon={<Cpu className="h-4 w-4" />} label="Model" value={model.split(" ")[0]} />
          <Metric
            icon={<Sparkles className="h-4 w-4" />}
            label="Resolution"
            value={status === "complete" ? "768×1024" : "—"}
          />
        </div>

        <div className="mt-4 flex items-start gap-2 rounded-lg bg-secondary/60 p-3 text-xs text-muted-foreground">
          <Shield className="mt-0.5 h-3.5 w-3.5 flex-shrink-0 text-primary" />
          <span>This is a CS project demo. Realism may vary. No user data stored.</span>
        </div>
      </div>
    </div>
  );
};

const Metric = ({ icon, label, value }: { icon: React.ReactNode; label: string; value: string }) => (
  <div className="rounded-lg border border-border bg-secondary/40 p-3">
    <div className="mb-1 flex items-center gap-1.5 text-muted-foreground">
      {icon}
      <span className="text-[11px] uppercase tracking-wide">{label}</span>
    </div>
    <p className="text-sm font-semibold text-foreground">{value}</p>
  </div>
);
