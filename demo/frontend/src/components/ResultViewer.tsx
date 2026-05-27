import { Download, Expand, ImageIcon, RefreshCw, Sparkles } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import { Skeleton } from "@/components/ui/skeleton";
import { EmptyState } from "@/components/EmptyState";
import { ErrorAlert } from "@/components/ErrorAlert";

interface ResultViewerProps {
  imageUrl?: string;
  isGenerating: boolean;
  error?: string | null;
  canRetry: boolean;
  onRetry: () => void;
  onRegenerate: () => void;
  onDownload: () => void;
}

export const ResultViewer = ({
  imageUrl,
  isGenerating,
  error,
  canRetry,
  onRetry,
  onRegenerate,
  onDownload,
}: ResultViewerProps) => (
  <div className="space-y-4">
    {error && <ErrorAlert message={error} onRetry={canRetry ? onRetry : undefined} />}

    <div className="overflow-hidden rounded-lg border border-border bg-card shadow-soft">
      <div className="flex items-center justify-between gap-3 border-b border-border px-4 py-3">
        <div>
          <h2 className="text-sm font-semibold text-foreground">Best try-on result</h2>
          <p className="text-xs text-muted-foreground">QA-ranked output candidate</p>
        </div>
        <div className="flex items-center gap-1.5">
          {imageUrl && (
            <Dialog>
              <DialogTrigger asChild>
                <Button type="button" variant="ghost" size="icon" className="h-8 w-8" aria-label="Open result fullscreen">
                  <Expand className="h-4 w-4" />
                </Button>
              </DialogTrigger>
              <DialogContent className="max-w-5xl border-none bg-background/95 p-4">
                <DialogHeader>
                  <DialogTitle>Try-on result</DialogTitle>
                  <DialogDescription>Fullscreen result preview.</DialogDescription>
                </DialogHeader>
                <div className="flex max-h-[78vh] items-center justify-center overflow-hidden rounded-lg bg-secondary/50">
                  <img src={imageUrl} alt="Fullscreen try-on result" className="max-h-[78vh] max-w-full object-contain" />
                </div>
              </DialogContent>
            </Dialog>
          )}
          <Button
            type="button"
            variant="ghost"
            size="icon"
            className="h-8 w-8"
            onClick={onDownload}
            disabled={!imageUrl}
            aria-label="Download result"
          >
            <Download className="h-4 w-4" />
          </Button>
        </div>
      </div>

      <div className="relative flex aspect-[3/4] items-center justify-center bg-secondary/35">
        {isGenerating ? (
          <div className="absolute inset-0 space-y-3 p-4">
            <Skeleton className="h-full w-full rounded-lg" />
            <span className="sr-only">Generating try-on result</span>
          </div>
        ) : imageUrl ? (
          <img src={imageUrl} alt="Virtual try-on output" className="h-full w-full object-contain p-2" />
        ) : (
          <EmptyState
            icon={<ImageIcon className="h-5 w-5" />}
            title="Your try-on result will appear here"
            description="Upload both images, add an optional prompt, then run mask-free generation."
          />
        )}
      </div>
    </div>

    <div className="grid grid-cols-2 gap-2">
      <Button type="button" variant="outline" className="rounded-lg" onClick={onDownload} disabled={!imageUrl}>
        <Download className="h-4 w-4" />
        Download
      </Button>
      <Button type="button" variant="outline" className="rounded-lg" onClick={onRegenerate} disabled={!canRetry}>
        <RefreshCw className="h-4 w-4" />
        Regenerate
      </Button>
    </div>

    {imageUrl && (
      <div className="flex items-start gap-2 rounded-lg border border-emerald-200 bg-emerald-50 p-3 text-xs leading-5 text-emerald-800">
        <Sparkles className="mt-0.5 h-4 w-4 shrink-0" />
        <span>Best candidate selected after QA reranking.</span>
      </div>
    )}
  </div>
);
