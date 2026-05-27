import { useState } from "react";
import { AlertTriangle, CheckCircle2, ChevronDown, Circle, Loader2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";
import { Progress } from "@/components/ui/progress";
import { cn } from "@/lib/utils";
import type { PipelineStep } from "@/types/tryOn";

interface PipelineProgressProps {
  steps: PipelineStep[];
  progress: number;
  isGenerating: boolean;
}

const statusIcon = (status: PipelineStep["status"]) => {
  if (status === "done") return <CheckCircle2 className="h-4 w-4 text-emerald-600" />;
  if (status === "running") return <Loader2 className="h-4 w-4 animate-spin text-primary" />;
  if (status === "error") return <AlertTriangle className="h-4 w-4 text-destructive" />;
  return <Circle className="h-4 w-4 text-muted-foreground/60" />;
};

export const PipelineProgress = ({ steps, progress, isGenerating }: PipelineProgressProps) => {
  const [isOpen, setIsOpen] = useState(false);
  const activeStep = steps.find((step) => step.status === "running") || steps.find((step) => step.status === "error");

  return (
    <Collapsible open={isOpen} onOpenChange={setIsOpen}>
      <section className="rounded-lg border border-border bg-card p-4 shadow-soft" aria-live="polite">
        <div className="flex items-center justify-between gap-3">
          <div className="min-w-0">
            <h2 className="text-sm font-semibold text-foreground">Pipeline status</h2>
            <p className="truncate text-xs text-muted-foreground">
              {activeStep?.label || (isGenerating ? "Mask-free generation in progress" : "Ready for mask-free try-on")}
            </p>
          </div>
          <div className="flex shrink-0 items-center gap-2">
            <span className="rounded-md border border-border bg-secondary px-2.5 py-1 text-xs font-semibold text-foreground">
              {progress}%
            </span>
            <CollapsibleTrigger asChild>
              <Button type="button" variant="outline" size="sm" className="h-8 rounded-md px-2.5">
                {isOpen ? "Hide" : "Show"}
                <ChevronDown className={cn("h-4 w-4 transition-transform", isOpen && "rotate-180")} />
              </Button>
            </CollapsibleTrigger>
          </div>
        </div>

        <Progress value={progress} className="mt-4 h-2" />

        <CollapsibleContent>
          <ol className="mt-4 space-y-3">
            {steps.map((step) => (
              <li key={step.id} className="flex gap-3">
                <div className="mt-0.5">{statusIcon(step.status)}</div>
                <div className="min-w-0 flex-1">
                  <div className="flex items-center justify-between gap-3">
                    <p
                      className={cn(
                        "truncate text-sm font-medium",
                        step.status === "pending" ? "text-muted-foreground" : "text-foreground",
                      )}
                    >
                      {step.label}
                    </p>
                    <span className="shrink-0 text-[11px] uppercase text-muted-foreground">{step.status}</span>
                  </div>
                  <p className="mt-0.5 text-xs leading-5 text-muted-foreground">{step.description}</p>
                </div>
              </li>
            ))}
          </ol>
        </CollapsibleContent>
      </section>
    </Collapsible>
  );
};
