import { CandidateGallery } from "@/components/CandidateGallery";
import { ComparisonView } from "@/components/ComparisonView";
import { ConfidencePanel } from "@/components/ConfidencePanel";
import { DetailsPanel } from "@/components/DetailsPanel";
import { ResultViewer } from "@/components/ResultViewer";
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
  selectedCandidateId: string | null;
  candidates: TryOnCandidate[];
  confidence?: number;
  warnings: TryOnWarning[];
  error?: string | null;
  metadata: TryOnMetadata | null;
  comparison: TryOnComparison | null;
  canGenerate: boolean;
  onGenerate: () => void;
  onDownload: () => void;
  onSelectCandidate: (candidateId: string) => void;
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
  selectedCandidateId,
  candidates,
  confidence,
  warnings,
  error,
  metadata,
  comparison,
  canGenerate,
  onGenerate,
  onDownload,
  onSelectCandidate,
}: OutputPanelProps) => (
  <section className="space-y-4">
    <Tabs defaultValue="result" className="rounded-lg border border-border bg-card p-4 shadow-soft">
      <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <h2 className="text-lg font-semibold tracking-tight text-foreground">Output</h2>
          <p className="text-sm text-muted-foreground">Best result, candidates, confidence, and debug details.</p>
        </div>
        <TabsList className="grid h-auto w-full grid-cols-4 rounded-lg sm:w-auto">
          <TabsTrigger value="result" className="px-2 text-xs sm:px-3">
            Result
          </TabsTrigger>
          <TabsTrigger value="comparison" className="px-2 text-xs sm:px-3">
            Compare
          </TabsTrigger>
          <TabsTrigger value="candidates" className="px-2 text-xs sm:px-3">
            Candidates
          </TabsTrigger>
          <TabsTrigger value="details" className="px-2 text-xs sm:px-3">
            Details
          </TabsTrigger>
        </TabsList>
      </div>

      <TabsContent value="result" className="mt-4 space-y-4">
        <ResultViewer
          imageUrl={selectedCandidate?.imageUrl}
          isGenerating={isGenerating}
          error={error}
          canRetry={canGenerate}
          onRetry={onGenerate}
          onRegenerate={onGenerate}
          onDownload={onDownload}
        />
        <ConfidencePanel confidence={confidence} warnings={warnings} />
        {candidates.length > 0 && (
          <CandidateGallery
            candidates={candidates}
            selectedCandidateId={selectedCandidateId}
            onSelect={onSelectCandidate}
          />
        )}
      </TabsContent>

      <TabsContent value="comparison" className="mt-4">
        <ComparisonView
          personPreviewUrl={personPreviewUrl}
          itemPreviewUrl={itemPreviewUrl}
          resultImageUrl={selectedCandidate?.imageUrl}
          comparison={comparison}
        />
      </TabsContent>

      <TabsContent value="candidates" className="mt-4">
        <CandidateGallery candidates={candidates} selectedCandidateId={selectedCandidateId} onSelect={onSelectCandidate} />
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
