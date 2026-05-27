import type {
  NormalizedTryOnResponse,
  TryOnCandidate,
  TryOnCandidateScores,
  TryOnMetadata,
  TryOnRequest,
  TryOnWarning,
} from "@/types/tryOn";

type ApiRecord = Record<string, unknown>;

const isRecord = (value: unknown): value is ApiRecord =>
  typeof value === "object" && value !== null && !Array.isArray(value);

const asRecord = (value: unknown): ApiRecord => (isRecord(value) ? value : {});

const asArray = (value: unknown): unknown[] => (Array.isArray(value) ? value : []);

const asString = (value: unknown): string | undefined => (typeof value === "string" ? value : undefined);

const asNumber = (value: unknown): number | undefined => {
  if (typeof value === "number" && Number.isFinite(value)) return value;
  if (typeof value === "string" && value.trim() !== "") {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : undefined;
  }
  return undefined;
};

const normalizeUnitScore = (value: unknown): number | undefined => {
  const score = asNumber(value);
  if (score === undefined) return undefined;
  if (score > 1 && score <= 100) return score / 100;
  return Math.min(1, Math.max(0, score));
};

const pickImageUrl = (value: unknown): string | undefined => {
  if (typeof value === "string") return value;
  const record = asRecord(value);
  return (
    asString(record.image_url) ||
    asString(record.imageUrl) ||
    asString(record.url) ||
    asString(record.output_url) ||
    asString(record.outputUrl) ||
    asString(record.result_url) ||
    asString(record.resultUrl)
  );
};

const normalizeScores = (rawScores: unknown, fallbackScore?: number): TryOnCandidateScores | undefined => {
  const scores = asRecord(rawScores);
  const normalized: TryOnCandidateScores = {
    objectSimilarity: normalizeUnitScore(scores.object_similarity ?? scores.objectSimilarity),
    personPreservation: normalizeUnitScore(scores.person_preservation ?? scores.personPreservation),
    localization: normalizeUnitScore(scores.localization),
    artifact: normalizeUnitScore(scores.artifact),
    overall: normalizeUnitScore(scores.overall ?? fallbackScore),
  };

  return Object.values(normalized).some((value) => value !== undefined) ? normalized : undefined;
};

const normalizeWarnings = (rawWarnings: unknown): TryOnWarning[] =>
  asArray(rawWarnings)
    .map((warning) => {
      if (typeof warning === "string") {
        return { message: warning, severity: "warning" as const };
      }

      const record = asRecord(warning);
      const message = asString(record.message) || asString(record.detail);
      if (!message) return null;

      const severity = asString(record.severity);
      return {
        code: asString(record.code),
        message,
        severity: severity === "error" || severity === "info" || severity === "warning" ? severity : "warning",
      };
    })
    .filter(Boolean) as TryOnWarning[];

const normalizeMetadata = (rawMetadata: unknown, request?: Partial<TryOnRequest>): TryOnMetadata => {
  const metadata = asRecord(rawMetadata);

  return {
    jobId: asString(metadata.job_id) || asString(metadata.jobId),
    responseId: asString(metadata.response_id) || asString(metadata.responseId) || asString(metadata.id),
    generationTimeMs: asNumber(metadata.generation_time_ms ?? metadata.generationTimeMs),
    mode: request?.mode,
    numCandidates: request?.numCandidates,
    source: "api",
  };
};

const candidateFromRaw = (rawCandidate: unknown, index: number): TryOnCandidate | null => {
  const candidate = asRecord(rawCandidate);
  const imageUrl = pickImageUrl(candidate);
  if (!imageUrl) return null;

  const score = normalizeUnitScore(candidate.score ?? candidate.confidence ?? candidate.overall_score);
  return {
    id: asString(candidate.id) || asString(candidate.candidate_id) || `candidate_${index + 1}`,
    imageUrl,
    score,
    confidence: normalizeUnitScore(candidate.confidence ?? candidate.score),
    scores: normalizeScores(candidate.scores, score),
    isBest: Boolean(candidate.is_best ?? candidate.isBest),
    candidateIndex: asNumber(candidate.candidate_index ?? candidate.candidateIndex) ?? index,
  };
};

// Demo-only fallback confidence. Real backends should return confidence or scores.
export const mockConfidenceFromCandidate = (candidate?: TryOnCandidate): number | undefined => {
  if (!candidate) return undefined;
  if (candidate.confidence !== undefined) return candidate.confidence;
  if (candidate.score !== undefined) return candidate.score;
  const scores = candidate.scores;
  if (!scores) return undefined;

  const values = [scores.objectSimilarity, scores.personPreservation, scores.localization, scores.overall].filter(
    (value): value is number => value !== undefined,
  );

  if (values.length === 0) return undefined;
  return values.reduce((sum, value) => sum + value, 0) / values.length;
};

export const normalizeTryOnResponse = (
  rawResponse: unknown,
  request?: Partial<TryOnRequest>,
): NormalizedTryOnResponse => {
  const raw = asRecord(rawResponse);
  const rawResult = asRecord(raw.result ?? raw.output);
  const rawCandidates = asArray(raw.candidates);

  const candidates = rawCandidates
    .map((candidate, index) => candidateFromRaw(candidate, index))
    .filter(Boolean) as TryOnCandidate[];

  const directImageUrl =
    pickImageUrl(rawResult) ||
    pickImageUrl(raw) ||
    pickImageUrl(raw.output) ||
    pickImageUrl(asArray(raw.images)[0]) ||
    candidates[0]?.imageUrl;

  if (candidates.length === 0 && directImageUrl) {
    candidates.push({
      id: "candidate_1",
      imageUrl: directImageUrl,
      score: normalizeUnitScore(rawResult.score ?? raw.score ?? raw.confidence),
      confidence: normalizeUnitScore(rawResult.confidence ?? raw.confidence ?? raw.score),
      scores: normalizeScores(rawResult.scores),
      isBest: true,
      candidateIndex: 0,
    });
  }

  const bestCandidate =
    candidates.find((candidate) => candidate.isBest) ||
    candidates.find((candidate) => candidate.candidateIndex === asNumber(rawResult.candidate_index ?? rawResult.candidateIndex)) ||
    candidates[0];

  if (!directImageUrl || !bestCandidate) {
    throw new Error("Response did not include a result image.");
  }

  candidates.forEach((candidate) => {
    candidate.isBest = candidate.id === bestCandidate.id || candidate.isBest;
  });

  return {
    result: {
      imageUrl: pickImageUrl(rawResult) || bestCandidate.imageUrl,
      confidence: normalizeUnitScore(rawResult.confidence ?? raw.confidence) ?? mockConfidenceFromCandidate(bestCandidate),
      candidateId: asString(rawResult.candidate_id) || asString(rawResult.candidateId) || bestCandidate.id,
      candidateIndex:
        asNumber(rawResult.candidate_index ?? rawResult.candidateIndex) ?? bestCandidate.candidateIndex ?? 0,
    },
    candidates,
    metadata: normalizeMetadata(raw.metadata, request),
    warnings: normalizeWarnings(raw.warnings),
  };
};
