import { ImageIcon } from "lucide-react";
import { EmptyState } from "@/components/EmptyState";
import type { TryOnComparison } from "@/types/tryOn";

interface ComparisonViewProps {
  personPreviewUrl: string | null;
  itemPreviewUrl: string | null;
  resultImageUrl?: string;
  comparison?: TryOnComparison | null;
}

const percent = (value?: number) => (value === undefined ? "--" : `${Math.round(value * 100)}%`);

const ComparisonImage = ({
  label,
  imageUrl,
  emptyTitle,
  score,
  caption,
}: {
  label: string;
  imageUrl?: string | null;
  emptyTitle: string;
  score?: number;
  caption?: string;
}) => (
  <div className="overflow-hidden rounded-lg border border-border bg-card">
    <div className="flex min-h-12 items-center justify-between gap-3 border-b border-border px-4 py-3">
      <h3 className="text-sm font-semibold text-foreground">{label}</h3>
      {score !== undefined && <span className="text-xs font-semibold text-primary">{percent(score)}</span>}
    </div>
    <div className="flex aspect-[3/4] items-center justify-center bg-secondary/35 p-3">
      {imageUrl ? (
        <img src={imageUrl} alt={label} className="max-h-full max-w-full object-contain" />
      ) : (
        <EmptyState icon={<ImageIcon className="h-5 w-5" />} title={emptyTitle} className="h-full" />
      )}
    </div>
    {caption && <div className="border-t border-border px-4 py-2 text-xs leading-5 text-muted-foreground">{caption}</div>}
  </div>
);

const formatDelta = (value?: number) => {
  if (value === undefined) return "--";
  const sign = value > 0 ? "+" : "";
  return `${sign}${(value * 100).toFixed(1)} pts`;
};

export const ComparisonView = ({ personPreviewUrl, itemPreviewUrl, resultImageUrl, comparison }: ComparisonViewProps) => {
  const pretrained = comparison?.pretrained;
  const geometry = comparison?.geometry;
  const winner = comparison?.delta?.winner;
  const deltaText = comparison?.delta ? `Geometry delta: ${formatDelta(comparison.delta.total)}` : undefined;
  const personImageUrl = comparison?.personImageUrl || personPreviewUrl;
  const itemImageUrl = comparison?.itemImageUrl || itemPreviewUrl;

  return (
    <div className="space-y-4">
      <div>
        <h2 className="text-sm font-semibold text-foreground">Live pipeline comparison</h2>
        <p className="mt-1 text-xs text-muted-foreground">
          Original inputs, pretrained baseline, and pretrained + geometry affordance output.
        </p>
      </div>

      <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-4">
        <ComparisonImage label="Original Person" imageUrl={personImageUrl} emptyTitle="No person image" />
        <ComparisonImage label="Input Item" imageUrl={itemImageUrl} emptyTitle="No item image" />
        <ComparisonImage
          label="Pretrained"
          imageUrl={pretrained?.imageUrl}
          emptyTitle="No pretrained output"
          score={pretrained?.score}
          caption={pretrained?.confidenceLabel ? `Confidence: ${pretrained.confidenceLabel}` : undefined}
        />
        <ComparisonImage
          label="Pretrained + Geometry"
          imageUrl={geometry?.imageUrl || resultImageUrl}
          emptyTitle="No geometry output"
          score={geometry?.score}
          caption={geometry?.confidenceLabel ? `Confidence: ${geometry.confidenceLabel}` : undefined}
        />
      </div>

      {(deltaText || comparison?.delta?.reason) && (
        <div className="rounded-lg border border-primary/15 bg-primary/5 px-4 py-3 text-xs leading-5 text-muted-foreground">
          <span className="font-semibold text-foreground">
            {winner ? `Winner: ${winner}. ` : ""}
            {deltaText}
          </span>
          {comparison?.delta?.reason && <span> {comparison.delta.reason}</span>}
        </div>
      )}
    </div>
  );
};
