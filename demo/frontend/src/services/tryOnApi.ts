import mockResultImage from "@/assets/tryon-result.jpg";
import { normalizeTryOnResponse } from "@/services/normalizeTryOnResponse";
import type { NormalizedTryOnResponse, TryOnCandidate, TryOnRequest, TryOnWarning } from "@/types/tryOn";

const TRYON_API_URL = import.meta.env.VITE_TRYON_API_URL as string | undefined;

const abortError = () => {
  const error = new Error("Generation was canceled.");
  error.name = "AbortError";
  return error;
};

const sleep = (ms: number, signal?: AbortSignal) =>
  new Promise<void>((resolve, reject) => {
    if (signal?.aborted) {
      reject(abortError());
      return;
    }

    const timeoutId = window.setTimeout(resolve, ms);
    signal?.addEventListener(
      "abort",
      () => {
        window.clearTimeout(timeoutId);
        reject(abortError());
      },
      { once: true },
    );
  });

const appendOptional = (formData: FormData, key: string, value: string | number | boolean | undefined) => {
  if (value !== undefined && value !== "") {
    formData.append(key, String(value));
  }
};

const buildFormData = (request: TryOnRequest) => {
  const formData = new FormData();
  formData.append("person_image", request.personFile);
  formData.append("item_image", request.itemFile);
  appendOptional(formData, "prompt", request.prompt?.trim());
  appendOptional(formData, "category", request.category === "auto" ? undefined : request.category);
  appendOptional(formData, "mode", request.mode);
  appendOptional(formData, "num_candidates", request.numCandidates);
  appendOptional(formData, "guidance_scale", request.guidanceScale);
  appendOptional(formData, "seed", request.seed);
  appendOptional(formData, "enable_reranker", request.enableReranker);
  appendOptional(formData, "enable_refiner", request.enableRefiner);
  return formData;
};

const postToTryOnApi = async (request: TryOnRequest, signal?: AbortSignal): Promise<NormalizedTryOnResponse> => {
  const response = await fetch(TRYON_API_URL!, {
    method: "POST",
    body: buildFormData(request),
    signal,
  });

  if (!response.ok) {
    const message = await response.text().catch(() => "");
    throw new Error(message || `Generation failed with status ${response.status}.`);
  }

  const rawResponse = await response.json();
  return normalizeTryOnResponse(rawResponse, request);
};

const mockScores = (index: number) => {
  const overall = Math.max(0.62, 0.91 - index * 0.055);

  return {
    object_similarity: Math.max(0.58, overall - 0.01),
    person_preservation: Math.max(0.6, overall - 0.025),
    localization: Math.max(0.55, overall - 0.04),
    artifact: Math.min(0.25, 0.08 + index * 0.035),
    overall,
  };
};

const buildMockCandidates = (count: number): TryOnCandidate[] =>
  Array.from({ length: count }, (_, index) => {
    const scores = mockScores(index);
    return {
      id: `candidate_${index + 1}`,
      imageUrl: mockResultImage,
      score: scores.overall,
      confidence: scores.overall,
      scores: {
        objectSimilarity: scores.object_similarity,
        personPreservation: scores.person_preservation,
        localization: scores.localization,
        artifact: scores.artifact,
        overall: scores.overall,
      },
      isBest: index === 0,
      candidateIndex: index,
    };
  });

const generateMockTryOn = async (request: TryOnRequest, signal?: AbortSignal): Promise<NormalizedTryOnResponse> => {
  const candidateCount = Math.min(5, Math.max(1, request.numCandidates || 1));
  await sleep(1800 + candidateCount * 450, signal);

  const candidates = buildMockCandidates(candidateCount);
  const bestCandidate = candidates[0];
  const warnings: TryOnWarning[] = request.prompt?.toLowerCase().includes("left")
    ? []
    : [
        {
          code: "mock_prompt_specificity",
          message: "Demo warning: add a more specific prompt if placement needs to be constrained.",
          severity: "info",
        },
      ];

  // Mock fallback for local demos when VITE_TRYON_API_URL is not configured.
  return {
    result: {
      imageUrl: bestCandidate.imageUrl,
      confidence: bestCandidate.confidence,
      candidateId: bestCandidate.id,
      candidateIndex: bestCandidate.candidateIndex,
    },
    candidates,
    metadata: {
      jobId: `mock_${Date.now()}`,
      generationTimeMs: 1800 + candidateCount * 450,
      mode: request.mode,
      numCandidates: candidateCount,
      source: "mock",
    },
    warnings,
  };
};

export const generateTryOn = async (
  request: TryOnRequest,
  signal?: AbortSignal,
): Promise<NormalizedTryOnResponse> => {
  if (TRYON_API_URL) {
    return postToTryOnApi(request, signal);
  }

  return generateMockTryOn(request, signal);
};
