import { Download, ImageIcon, RefreshCw } from "lucide-react";
import { EmptyState } from "@/components/EmptyState";
import { ErrorAlert } from "@/components/ErrorAlert";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import type { TryOnCandidate, TryOnComparisonDelta } from "@/types/tryOn";

const percent = (value?: number) => (value === undefined ? "--" : `${Math.round(value * 100)}%`);

const formatDelta = (value?: number) => {
  if (value === undefined) return "--";
  const sign = value > 0 ? "+" : "";
  return `${sign}${(value * 100).toFixed(1)} pts`;
};

const OutputSlot = ({
  title,
  caption,
  candidate,
  isGenerating,
}: {
  title: string;
  caption: string;
  candidate?: TryOnCandidate | null;
  isGenerating: boolean;
}) => (
  <div className="overflow-hidden rounded-lg border border-border bg-card shadow-soft">
    <div className="flex min-h-14 items-center justify-between gap-3 border-b border-border px-4 py-3">
      <div className="min-w-0">
        <h3 className="text-sm font-semibold text-foreground">{title}</h3>
        <p className="mt-0.5 truncate text-xs text-muted-foreground">{caption}</p>
      </div>
      <span className="text-xs font-semibold text-primary">{percent(candidate?.score)}</span>
    </div>

    <div className="relative flex aspect-[3/4] items-center justify-center bg-secondary/35">
      {isGenerating ? (
        <div className="absolute inset-0 p-4">
          <Skeleton className="h-full w-full rounded-lg" />
          <span className="sr-only">Generating output</span>
        </div>
      ) : candidate?.imageUrl ? (
        <img src={candidate.imageUrl} alt={title} className="h-full w-full object-contain p-2" />
      ) : (
        <EmptyState icon={<ImageIcon className="h-5 w-5" />} title="No output yet" className="h-full" />
      )}
    </div>
  </div>
);

export const SideBySideOutput = ({
  leftCandidate,
  rightCandidate,
  delta,
  isGenerating,
  error,
  canRetry,
  onRetry,
  onDownload,
}: {
  leftCandidate?: TryOnCandidate | null;
  rightCandidate?: TryOnCandidate | null;
  delta?: TryOnComparisonDelta;
  isGenerating: boolean;
  error?: string | null;
  canRetry: boolean;
  onRetry: () => void;
  onDownload: () => void;
}) => (
  <div className="space-y-4">
    {error && <ErrorAlert message={error} onRetry={canRetry ? onRetry : undefined} />}

    <div className="grid gap-3 lg:grid-cols-2">
      <OutputSlot
        title="Pretrained baseline"
        caption=""
        candidate={leftCandidate}
        isGenerating={isGenerating}
      />
      <OutputSlot
        title="Geometry-selected output"
        caption=""
        candidate={rightCandidate}
        isGenerating={isGenerating}
      />
    </div>

    {(delta?.total !== undefined || delta?.reason) && (
      <div className="rounded-lg border border-primary/15 bg-primary/5 px-4 py-3 text-xs leading-5 text-muted-foreground">
        <span className="font-semibold text-foreground">Right panel delta: {formatDelta(delta?.total)}</span>
        {delta?.reason && <span> {delta.reason}</span>}
      </div>
    )}

    <div className="grid grid-cols-2 gap-2">
      <Button type="button" variant="outline" className="rounded-lg" onClick={onDownload} disabled={!rightCandidate?.imageUrl}>
        <Download className="h-4 w-4" />
        Download right
      </Button>
      <Button type="button" variant="outline" className="rounded-lg" onClick={onRetry} disabled={!canRetry}>
        <RefreshCw className="h-4 w-4" />
        Regenerate
      </Button>
    </div>
  </div>
);
