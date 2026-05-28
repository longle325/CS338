import { ConfidencePanel } from "@/components/ConfidencePanel";
import { DetailsPanel } from "@/components/DetailsPanel";
import { SideBySideOutput } from "@/components/SideBySideOutput";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import type { TryOnCandidate, TryOnComparison, TryOnMetadata, TryOnMode, TryOnWarning } from "@/types/tryOn";

interface OutputPanelProps {
  personFile: File | null;
  itemFile: File | null;
  personPreviewUrl: string | null;
  itemPreviewUrl: string | null;
  prompt: string;
  mode: TryOnMode;
  numCandidates: number;
  isGenerating: boolean;
  selectedCandidate: TryOnCandidate | null;
  candidates: TryOnCandidate[];
  confidence?: number;
  warnings: TryOnWarning[];
  error?: string | null;
  metadata: TryOnMetadata | null;
  comparison: TryOnComparison | null;
  canGenerate: boolean;
  onGenerate: () => void;
  onDownload: () => void;
}

export const OutputPanel = ({
  personFile,
  itemFile,
  personPreviewUrl,
  itemPreviewUrl,
  prompt,
  mode,
  numCandidates,
  isGenerating,
  selectedCandidate,
  candidates,
  confidence,
  warnings,
  error,
  metadata,
  comparison,
  canGenerate,
  onGenerate,
  onDownload,
}: OutputPanelProps) => {
  const leftCandidate = candidates.find((candidate) => candidate.branch === "pretrained") || null;
  const rightCandidate =
    candidates.find((candidate) => candidate.branch === "geometry") || selectedCandidate || null;

  return (
    <section className="space-y-4">
      <Tabs defaultValue="result" className="rounded-lg border border-border bg-card p-4 shadow-soft">
        <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
          <div>
            <h2 className="text-lg font-semibold tracking-tight text-foreground">Output</h2>
          </div>
          <TabsList className="grid h-auto w-full grid-cols-2 rounded-lg sm:w-auto">
            <TabsTrigger value="result" className="px-2 text-xs sm:px-3">
              Result
            </TabsTrigger>
            <TabsTrigger value="details" className="px-2 text-xs sm:px-3">
              Details
            </TabsTrigger>
          </TabsList>
        </div>

        <TabsContent value="result" className="mt-4 space-y-4">
          <SideBySideOutput
            leftCandidate={leftCandidate}
            rightCandidate={rightCandidate}
            delta={comparison?.delta}
            isGenerating={isGenerating}
            error={error}
            canRetry={canGenerate}
            onRetry={onGenerate}
            onDownload={onDownload}
          />
          <ConfidencePanel warnings={warnings} />
        </TabsContent>

        <TabsContent value="details" className="mt-4">
          <DetailsPanel
            personFile={personFile}
            itemFile={itemFile}
            prompt={prompt}
            mode={mode}
            numCandidates={numCandidates}
            metadata={metadata}
            comparison={comparison}
            selectedCandidate={selectedCandidate}
            confidence={confidence}
            error={error}
          />
        </TabsContent>
      </Tabs>
    </section>
  );
};
