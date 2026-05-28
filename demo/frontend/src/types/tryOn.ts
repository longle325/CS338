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
  | "top clothes"
  | "bottom clothes"
  | "dress"
  | "shoe"
  | "earrings"
  | "bracelet"
  | "necklace"
  | "ring"
  | "sunglasses"
  | "glasses"
  | "belt"
  | "bag"
  | "hat"
  | "tie"
  | "bow tie"
  | "garment"
  | "shoes"
  | "jewelry"
  | "watch"
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
  label?: string;
  branch?: "pretrained" | "geometry" | string;
  imageUrl: string;
  score?: number;
  confidence?: number;
  scores?: TryOnCandidateScores;
  isBest?: boolean;
  candidateIndex: number;
  rank?: number;
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
  objectClass?: string;
  statusUrl?: string;
  source?: "api" | "mock";
}

export interface TryOnBranchSummary {
  label: string;
  branch: "pretrained" | "geometry" | string;
  imageUrl?: string;
  score?: number;
  confidenceLabel?: string;
  candidateCount?: number;
  diagnosticsUrl?: string;
  scores?: TryOnCandidateScores;
}

export interface TryOnComparisonDelta {
  total?: number;
  object?: number;
  person?: number;
  artifact?: number;
  winner?: "pretrained" | "geometry" | "tie" | string;
  reason?: string;
}

export interface TryOnComparison {
  personImageUrl?: string;
  itemImageUrl?: string;
  objectClass?: string;
  pretrained?: TryOnBranchSummary;
  geometry?: TryOnBranchSummary;
  delta?: TryOnComparisonDelta;
}

export interface NormalizedTryOnResponse {
  result: TryOnResult;
  candidates: TryOnCandidate[];
  metadata: TryOnMetadata;
  comparison?: TryOnComparison;
  warnings: TryOnWarning[];
}
