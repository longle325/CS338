import { FileText } from "lucide-react";
import { MODE_CONFIG } from "@/lib/tryOnConfig";
import type { TryOnCandidate, TryOnComparison, TryOnMetadata, TryOnMode } from "@/types/tryOn";

const formatMs = (value?: number) => {
  if (value === undefined) return "--";
  if (value < 1000) return `${value} ms`;
  return `${(value / 1000).toFixed(1)} s`;
};

const formatConfidence = (value?: number) => (value === undefined ? "--" : `${Math.round(value * 100)}%`);

const DetailRow = ({ label, value }: { label: string; value?: string | number | null }) => (
  <div className="grid grid-cols-[minmax(120px,0.9fr)_minmax(0,1.1fr)] gap-3 border-b border-border py-2 last:border-b-0">
    <dt className="text-xs font-medium uppercase text-muted-foreground">{label}</dt>
    <dd className="min-w-0 break-words text-sm text-foreground">{value ?? "--"}</dd>
  </div>
);

interface DetailsPanelProps {
  personFile: File | null;
  itemFile: File | null;
  prompt: string;
  mode: TryOnMode;
  numCandidates: number;
  metadata: TryOnMetadata | null;
  comparison: TryOnComparison | null;
  selectedCandidate: TryOnCandidate | null;
  confidence?: number;
  error?: string | null;
}

export const DetailsPanel = ({
  personFile,
  itemFile,
  prompt,
  mode,
  numCandidates,
  metadata,
  comparison,
  selectedCandidate,
  confidence,
  error,
}: DetailsPanelProps) => (
  <section className="rounded-lg border border-border bg-card p-4 shadow-soft">
    <div className="mb-3 flex items-center gap-2">
      <FileText className="h-4 w-4 text-primary" />
      <h2 className="text-sm font-semibold text-foreground">Details / debug info</h2>
    </div>

    <dl>
      <DetailRow label="Person file" value={personFile?.name} />
      <DetailRow label="Item file" value={itemFile?.name} />
      <DetailRow label="Prompt" value={prompt.trim() || "No prompt"} />
      <DetailRow label="Mode" value={MODE_CONFIG[mode].label} />
      <DetailRow label="Object class" value={metadata?.objectClass || comparison?.objectClass} />
      <DetailRow label="Candidates K" value={metadata?.numCandidates ?? numCandidates} />
      <DetailRow label="QA reranking" value="Automatic pipeline step" />
      <DetailRow
        label="Local refinement"
        value={metadata?.source === "mock" ? "Mock fallback / planned backend step" : "Automatic pipeline step"}
      />
      <DetailRow label="Generation time" value={formatMs(metadata?.generationTimeMs)} />
      <DetailRow
        label="Selected candidate"
        value={selectedCandidate ? selectedCandidate.label || `Candidate ${selectedCandidate.candidateIndex + 1}` : "--"}
      />
      <DetailRow label="Comparison winner" value={comparison?.delta?.winner} />
      <DetailRow label="Geometry delta" value={comparison?.delta?.total === undefined ? undefined : `${(comparison.delta.total * 100).toFixed(1)} pts`} />
      <DetailRow label="Confidence" value={formatConfidence(confidence)} />
      <DetailRow label="Job id" value={metadata?.jobId} />
      <DetailRow label="Response id" value={metadata?.responseId} />
      <DetailRow label="Status URL" value={metadata?.statusUrl} />
      <DetailRow label="Source" value={metadata?.source === "mock" ? "Mock fallback" : metadata?.source || "API-ready"} />
      <DetailRow label="Error" value={error} />
    </dl>
  </section>
);
