import { ImageIcon } from "lucide-react";
import { EmptyState } from "@/components/EmptyState";

interface ComparisonViewProps {
  personPreviewUrl: string | null;
  itemPreviewUrl: string | null;
  resultImageUrl?: string;
}

const ComparisonImage = ({
  label,
  imageUrl,
  emptyTitle,
}: {
  label: string;
  imageUrl?: string | null;
  emptyTitle: string;
}) => (
  <div className="overflow-hidden rounded-lg border border-border bg-card">
    <div className="border-b border-border px-4 py-3">
      <h3 className="text-sm font-semibold text-foreground">{label}</h3>
    </div>
    <div className="flex aspect-[3/4] items-center justify-center bg-secondary/35 p-3">
      {imageUrl ? (
        <img src={imageUrl} alt={label} className="max-h-full max-w-full object-contain" />
      ) : (
        <EmptyState icon={<ImageIcon className="h-5 w-5" />} title={emptyTitle} className="h-full" />
      )}
    </div>
  </div>
);

export const ComparisonView = ({ personPreviewUrl, itemPreviewUrl, resultImageUrl }: ComparisonViewProps) => (
  <div className="space-y-4">
    <div>
      <h2 className="text-sm font-semibold text-foreground">Comparison view</h2>
      <p className="mt-1 text-xs text-muted-foreground">Original person, input item, and selected try-on result.</p>
    </div>

    <div className="grid gap-3 md:grid-cols-3">
      <ComparisonImage label="Original Person" imageUrl={personPreviewUrl} emptyTitle="No person image" />
      <ComparisonImage label="Input Item" imageUrl={itemPreviewUrl} emptyTitle="No item image" />
      <ComparisonImage label="Try-On Result" imageUrl={resultImageUrl} emptyTitle="No result yet" />
    </div>
  </div>
);
