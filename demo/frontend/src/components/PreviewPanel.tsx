import { ImageIcon, ScanFace, Shirt } from "lucide-react";
import type { ReactNode } from "react";
import { Badge } from "@/components/ui/badge";
import { EmptyState } from "@/components/EmptyState";
import { PipelineProgress } from "@/components/PipelineProgress";
import type { PipelineStep } from "@/types/tryOn";

interface PreviewPanelProps {
  personPreviewUrl: string | null;
  itemPreviewUrl: string | null;
  personFile: File | null;
  itemFile: File | null;
  pipelineSteps: PipelineStep[];
  pipelineProgress: number;
  isGenerating: boolean;
}

const PreviewTile = ({
  label,
  file,
  imageUrl,
  emptyTitle,
  icon,
}: {
  label: string;
  file: File | null;
  imageUrl: string | null;
  emptyTitle: string;
  icon: ReactNode;
}) => (
  <div className="overflow-hidden rounded-lg border border-border bg-card shadow-soft">
    <div className="flex items-center justify-between gap-3 border-b border-border px-4 py-3">
      <div className="flex min-w-0 items-center gap-2">
        {icon}
        <span className="text-sm font-semibold text-foreground">{label}</span>
      </div>
      {file && (
        <span className="max-w-[48%] truncate text-xs text-muted-foreground" title={file.name}>
          {file.name}
        </span>
      )}
    </div>
    <div className="flex aspect-[4/5] items-center justify-center bg-secondary/35 p-3">
      {imageUrl ? (
        <img src={imageUrl} alt={`${label} preview`} className="max-h-full max-w-full object-contain" />
      ) : (
        <EmptyState icon={<ImageIcon className="h-5 w-5" />} title={emptyTitle} className="h-full" />
      )}
    </div>
  </div>
);

export const PreviewPanel = ({
  personPreviewUrl,
  itemPreviewUrl,
  personFile,
  itemFile,
  pipelineSteps,
  pipelineProgress,
  isGenerating,
}: PreviewPanelProps) => (
  <section className="space-y-4">
    <div className="rounded-lg border border-border bg-card p-4 shadow-soft">
      <div className="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
        <div>
          <h2 className="text-lg font-semibold tracking-tight text-foreground">Preview workspace</h2>
        </div>
        <Badge variant="secondary" className="w-fit rounded-md border border-primary/15 bg-primary/10 text-primary">
          OmniTry++ mask-free
        </Badge>
      </div>
    </div>

    <div className="grid gap-4 md:grid-cols-2">
      <PreviewTile
        label="Subject"
        file={personFile}
        imageUrl={personPreviewUrl}
        emptyTitle="Upload a person image"
        icon={<ScanFace className="h-4 w-4 text-primary" />}
      />
      <PreviewTile
        label="Item"
        file={itemFile}
        imageUrl={itemPreviewUrl}
        emptyTitle="Upload a garment or accessory"
        icon={<Shirt className="h-4 w-4 text-primary" />}
      />
    </div>

    <PipelineProgress steps={pipelineSteps} progress={pipelineProgress} isGenerating={isGenerating} />
  </section>
);
