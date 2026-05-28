import { CheckCircle2, Images } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { EmptyState } from "@/components/EmptyState";
import { cn } from "@/lib/utils";
import type { TryOnCandidate } from "@/types/tryOn";

const percent = (value?: number) => (value === undefined ? "--" : `${Math.round(value * 100)}%`);

interface CandidateGalleryProps {
  candidates: TryOnCandidate[];
  selectedCandidateId: string | null;
  onSelect: (candidateId: string) => void;
}

export const CandidateGallery = ({ candidates, selectedCandidateId, onSelect }: CandidateGalleryProps) => {
  if (candidates.length === 0) {
    return (
      <EmptyState
        icon={<Images className="h-5 w-5" />}
        title="No candidates yet"
        description="Candidates appear after generation."
        className="rounded-lg border border-border bg-card"
      />
    );
  }

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between gap-3">
        <h3 className="text-sm font-semibold text-foreground">Candidate gallery</h3>
        <span className="text-xs text-muted-foreground">
          {candidates.length} candidate{candidates.length > 1 ? "s" : ""} generated
        </span>
      </div>

      <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 xl:grid-cols-2 2xl:grid-cols-3">
        {candidates.map((candidate) => {
          const isSelected = candidate.id === selectedCandidateId;
          const title = candidate.label || `Candidate ${candidate.candidateIndex + 1}`;
          const branchLabel =
            candidate.branch === "pretrained"
              ? "Pretrained"
              : candidate.branch === "geometry"
                ? "Geometry"
                : candidate.branch;

          return (
            <button
              key={candidate.id}
              type="button"
              onClick={() => onSelect(candidate.id)}
              className={cn(
                "group overflow-hidden rounded-lg border bg-card text-left transition-smooth focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring",
                isSelected ? "border-primary shadow-glow" : "border-border hover:border-primary/50",
              )}
            >
              <div className="relative aspect-[3/4] bg-secondary/40">
                <img
                  src={candidate.imageUrl}
                  alt={title}
                  className="h-full w-full object-cover"
                />
                <div className="absolute left-2 top-2 flex flex-wrap gap-1">
                  {branchLabel && <Badge variant="secondary" className="rounded-md bg-background/90">{branchLabel}</Badge>}
                  {candidate.isBest && <Badge className="rounded-md bg-emerald-600 text-white">Best</Badge>}
                  {isSelected && (
                    <Badge variant="secondary" className="rounded-md bg-background/90">
                      <CheckCircle2 className="mr-1 h-3 w-3" />
                      Selected
                    </Badge>
                  )}
                </div>
              </div>
              <div className="space-y-2 p-3">
                <div className="flex items-center justify-between gap-2">
                  <span className="min-w-0 truncate text-sm font-semibold text-foreground">{title}</span>
                  <span className="text-xs font-semibold text-primary">{percent(candidate.score)}</span>
                </div>
                {candidate.scores && (
                  <div className="grid grid-cols-2 gap-x-3 gap-y-1 text-[11px] text-muted-foreground">
                    <span>Object {percent(candidate.scores.objectSimilarity)}</span>
                    <span>Person {percent(candidate.scores.personPreservation)}</span>
                    <span>Localize {percent(candidate.scores.localization)}</span>
                    <span>Artifact {percent(candidate.scores.artifact)}</span>
                  </div>
                )}
              </div>
            </button>
          );
        })}
      </div>
    </div>
  );
};
