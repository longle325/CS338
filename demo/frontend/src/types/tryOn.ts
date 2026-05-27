export type TryOnMode = "fast" | "balanced" | "high_quality";

export type PipelineStepStatus = "pending" | "running" | "done" | "error";

export interface PipelineStep {
  id: string;
  label: string;
  description: string;
  status: PipelineStepStatus;
}

export type TryOnCategory =
  | "auto"
  | "garment"
  | "shoes"
  | "jewelry"
  | "bag"
  | "watch"
  | "glasses"
  | "hat"
  | "holdable";

export interface AdvancedTryOnSettings {
  guidanceScale?: number;
  seed?: number;
  enableReranker: boolean;
  enableRefiner: boolean;
}

export interface TryOnRequest {
  personFile: File;
  itemFile: File;
  prompt?: string;
  category?: TryOnCategory;
  mode: TryOnMode;
  numCandidates: number;
  guidanceScale?: number;
  seed?: number;
  enableReranker?: boolean;
  enableRefiner?: boolean;
}

export interface TryOnCandidateScores {
  objectSimilarity?: number;
  personPreservation?: number;
  localization?: number;
  artifact?: number;
  overall?: number;
}

export interface TryOnCandidate {
  id: string;
  imageUrl: string;
  score?: number;
  confidence?: number;
  scores?: TryOnCandidateScores;
  isBest?: boolean;
  candidateIndex: number;
}

export interface TryOnResult {
  imageUrl: string;
  confidence?: number;
  candidateId?: string;
  candidateIndex?: number;
}

export interface TryOnWarning {
  code?: string;
  message: string;
  severity?: "info" | "warning" | "error";
}

export interface TryOnMetadata {
  jobId?: string;
  responseId?: string;
  generationTimeMs?: number;
  mode?: TryOnMode;
  numCandidates?: number;
  source?: "api" | "mock";
}

export interface NormalizedTryOnResponse {
  result: TryOnResult;
  candidates: TryOnCandidate[];
  metadata: TryOnMetadata;
  warnings: TryOnWarning[];
}
