import { AlertTriangle, CheckCircle2, Info, ShieldCheck } from "lucide-react";
import { Progress } from "@/components/ui/progress";
import { cn } from "@/lib/utils";
import type { TryOnWarning } from "@/types/tryOn";

const getConfidenceState = (confidence?: number) => {
  if (confidence === undefined) {
    return {
      label: "Confidence pending",
      description: "Run generation to inspect placement quality.",
      className: "border-border bg-card text-muted-foreground",
      icon: <Info className="h-4 w-4" />,
    };
  }

  if (confidence >= 0.78) {
    return {
      label: "High confidence",
      description: "Object placement and person preservation look stable.",
      className: "border-emerald-200 bg-emerald-50 text-emerald-800",
      icon: <CheckCircle2 className="h-4 w-4" />,
    };
  }

  if (confidence >= 0.58) {
    return {
      label: "Medium confidence",
      description: "Please inspect object placement before using the result.",
      className: "border-amber-200 bg-amber-50 text-amber-800",
      icon: <AlertTriangle className="h-4 w-4" />,
    };
  }

  return {
    label: "Low confidence",
    description: "The system may not have placed the item correctly. Try adding a more specific prompt.",
    className: "border-destructive/30 bg-destructive/10 text-destructive",
    icon: <AlertTriangle className="h-4 w-4" />,
  };
};

interface ConfidencePanelProps {
  confidence?: number;
  warnings: TryOnWarning[];
}

export const ConfidencePanel = ({ confidence, warnings }: ConfidencePanelProps) => {
  const state = getConfidenceState(confidence);
  const confidencePercent = confidence === undefined ? 0 : Math.round(confidence * 100);

  return (
    <section className="space-y-3 rounded-lg border border-border bg-card p-4 shadow-soft">
      <div className="flex items-center justify-between gap-3">
        <div className="flex items-center gap-2">
          <ShieldCheck className="h-4 w-4 text-primary" />
          <h3 className="text-sm font-semibold text-foreground">Confidence & warnings</h3>
        </div>
        <span className="text-sm font-semibold text-foreground">{confidence === undefined ? "--" : `${confidencePercent}%`}</span>
      </div>

      <Progress value={confidencePercent} className="h-2" />

      <div className={cn("flex items-start gap-2 rounded-lg border p-3 text-xs leading-5", state.className)}>
        <div className="mt-0.5 shrink-0">{state.icon}</div>
        <div>
          <p className="font-semibold">{state.label}</p>
          <p>{state.description}</p>
        </div>
      </div>

      {warnings.length > 0 && (
        <div className="space-y-2">
          {warnings.map((warning, index) => (
            <div
              key={`${warning.code || "warning"}-${index}`}
              className="rounded-lg border border-amber-200 bg-amber-50 px-3 py-2 text-xs leading-5 text-amber-800"
            >
              {warning.message}
            </div>
          ))}
        </div>
      )}
    </section>
  );
};
