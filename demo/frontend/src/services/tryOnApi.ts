import mockResultImage from "@/assets/tryon-result.jpg";
import { normalizeTryOnResponse } from "@/services/normalizeTryOnResponse";
import type {
  NormalizedTryOnResponse,
  TryOnBranchSummary,
  TryOnCandidate,
  TryOnCandidateScores,
  TryOnComparison,
  TryOnComparisonDelta,
  TryOnMode,
  TryOnRequest,
  TryOnWarning,
} from "@/types/tryOn";

type ApiRecord = Record<string, unknown>;

const API_BASE_URL = (import.meta.env.VITE_TRYON_API_BASE_URL as string | undefined)?.replace(/\/+$/, "") || "";
const TRYON_API_URL = import.meta.env.VITE_TRYON_API_URL as string | undefined;
const USE_MOCK = String(import.meta.env.VITE_TRYON_USE_MOCK || "").toLowerCase() === "true";
const POLL_INTERVAL_MS = 1500;
const POLL_TIMEOUT_MS = 15 * 60 * 1000;

const BACKEND_OBJECT_CLASSES = [
  "top clothes",
  "bottom clothes",
  "dress",
  "shoe",
  "earrings",
  "bracelet",
  "necklace",
  "ring",
  "sunglasses",
  "glasses",
  "belt",
  "bag",
  "hat",
  "tie",
  "bow tie",
] as const;

const CATEGORY_ALIASES: Record<string, string> = {
  garment: "top clothes",
  shoes: "shoe",
  jewelry: "ring",
  eyeglasses: "glasses",
  earring: "earrings",
  bracelets: "bracelet",
  rings: "ring",
  necklaces: "necklace",
  shirt: "top clothes",
  top: "top clothes",
  pants: "bottom clothes",
  bottom: "bottom clothes",
  bowtie: "bow tie",
};

const CLASS_KEYWORDS: Array<[string, string]> = [
  ["bottom clothes", "bottom clothes"],
  ["top clothes", "top clothes"],
  ["bow tie", "bow tie"],
  ["sunglasses", "sunglasses"],
  ["eyeglasses", "glasses"],
  ["glasses", "glasses"],
  ["earrings", "earrings"],
  ["earring", "earrings"],
  ["bracelet", "bracelet"],
  ["necklace", "necklace"],
  ["rings", "ring"],
  ["ring", "ring"],
  ["shoes", "shoe"],
  ["shoe", "shoe"],
  ["dress", "dress"],
  ["shirt", "top clothes"],
  ["top", "top clothes"],
  ["pants", "bottom clothes"],
  ["bottom", "bottom clothes"],
  ["belt", "belt"],
  ["bag", "bag"],
  ["hat", "hat"],
  ["bowtie", "bow tie"],
  ["tie", "tie"],
];

const STEPS_BY_MODE: Record<TryOnMode, number> = {
  fast: 12,
  balanced: 20,
  high_quality: 30,
};

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

const hasKeys = (record: ApiRecord) => Object.keys(record).length > 0;

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

const isAbsoluteUrl = (url: string) => /^https?:\/\//i.test(url);

const apiUrl = (path: string) => {
  if (isAbsoluteUrl(path)) return path;
  const normalizedPath = path.startsWith("/") ? path : `/${path}`;
  return `${API_BASE_URL}${normalizedPath}`;
};

const backendOriginFrom = (url: string) => {
  try {
    const parsed = new URL(url, window.location.origin);
    const isRelativeUrl = !isAbsoluteUrl(url);
    if (isRelativeUrl && !API_BASE_URL) return "";
    return parsed.origin;
  } catch {
    return API_BASE_URL;
  }
};

const resolveBackendUrl = (urlOrPath: string | undefined, backendOrigin: string, fallbackPath: string) => {
  const value = urlOrPath || fallbackPath;
  if (isAbsoluteUrl(value)) return value;
  const normalizedPath = value.startsWith("/") ? value : `/${value}`;
  return `${backendOrigin}${normalizedPath}`;
};

const normalizeCategory = (raw: string | undefined) => {
  const key = (raw || "").trim().toLowerCase().replace(/_/g, " ");
  const normalized = key.split(/\s+/).filter(Boolean).join(" ");
  if (!normalized || normalized === "auto") return undefined;
  if (BACKEND_OBJECT_CLASSES.includes(normalized as (typeof BACKEND_OBJECT_CLASSES)[number])) return normalized;
  return CATEGORY_ALIASES[normalized];
};

const containsKeyword = (text: string, keyword: string) => {
  const escaped = keyword.replace(/[.*+?^${}()|[\]\\]/g, "\\$&").replace(/\s+/g, "\\s+");
  return new RegExp(`(^|\\W)${escaped}(\\W|$)`, "i").test(text);
};

const inferObjectClass = (request: TryOnRequest) => {
  const selected = normalizeCategory(request.category);
  if (selected) return selected;

  const text = [request.prompt, request.itemFile.name].filter(Boolean).join(" ").toLowerCase();
  for (const [keyword, objectClass] of CLASS_KEYWORDS) {
    if (containsKeyword(text, keyword)) return objectClass;
  }

  throw new Error("Please choose the item category before running the live backend.");
};

const buildBackendFormData = (request: TryOnRequest) => {
  const formData = new FormData();
  formData.append("person_image", request.personFile);
  formData.append("object_image", request.itemFile);
  formData.append("object_class", inferObjectClass(request));
  appendOptional(formData, "optional_prompt", request.prompt?.trim());
  appendOptional(formData, "steps", STEPS_BY_MODE[request.mode]);
  appendOptional(formData, "guidance_scale", request.guidanceScale);
  appendOptional(formData, "seed", request.seed ?? -1);
  appendOptional(formData, "geometry_candidate_count", request.numCandidates);
  formData.append("run_pretrained", "true");
  formData.append("run_geometry", "true");
  return formData;
};

const readErrorMessage = async (response: Response) => {
  const text = await response.text().catch(() => "");
  if (!text) return `Generation failed with status ${response.status}.`;

  try {
    const payload = JSON.parse(text);
    const detail = payload.detail;
    if (typeof detail === "string") return detail;
    if (isRecord(detail) && typeof detail.message === "string") return detail.message;
    if (typeof payload.message === "string") return payload.message;
  } catch {
    return text;
  }

  return text;
};

const fetchJson = async (url: string, signal?: AbortSignal) => {
  const response = await fetch(url, { signal });
  if (!response.ok) throw new Error(await readErrorMessage(response));
  return response.json();
};

const normalizeBackendScores = (rawScores: unknown): TryOnCandidateScores | undefined => {
  const scores = asRecord(rawScores);
  const normalized: TryOnCandidateScores = {
    objectSimilarity: normalizeUnitScore(scores.object ?? scores.object_similarity ?? scores.objectSimilarity),
    personPreservation: normalizeUnitScore(scores.person ?? scores.person_preservation ?? scores.personPreservation),
    localization: normalizeUnitScore(scores.localization),
    artifact: normalizeUnitScore(scores.artifact),
    overall: normalizeUnitScore(scores.total ?? scores.overall ?? scores.score),
  };

  return Object.values(normalized).some((value) => value !== undefined) ? normalized : undefined;
};

const branchSummary = (
  branch: ApiRecord,
  branchName: "pretrained" | "geometry",
  backendOrigin: string,
): TryOnBranchSummary | undefined => {
  if (!hasKeys(branch)) return undefined;
  const scores = normalizeBackendScores(branch.scores);

  return {
    label: asString(branch.label) || (branchName === "geometry" ? "Pretrained + Geometry" : "Pretrained"),
    branch: branchName,
    imageUrl: asString(branch.image_url) ? resolveBackendUrl(asString(branch.image_url), backendOrigin, "") : undefined,
    score: scores?.overall,
    confidenceLabel: asString(asRecord(branch.scores).confidence),
    candidateCount: asNumber(branch.candidate_count),
    diagnosticsUrl: asString(branch.diagnostics_url)
      ? resolveBackendUrl(asString(branch.diagnostics_url), backendOrigin, "")
      : undefined,
    scores,
  };
};

const candidateFromBackend = (
  rawCandidate: unknown,
  index: number,
  branchName: "pretrained" | "geometry",
  backendOrigin: string,
): TryOnCandidate | null => {
  const candidate = asRecord(rawCandidate);
  const imageUrl = asString(candidate.image_url);
  if (!imageUrl) return null;

  const scores = normalizeBackendScores(candidate.scores);
  const rank = asNumber(candidate.rank);
  const seed = asNumber(candidate.seed);
  const label =
    branchName === "pretrained"
      ? "Pretrained baseline"
      : rank !== undefined
        ? `Geometry candidate ${rank}`
        : `Geometry candidate ${index + 1}`;

  return {
    id: `${branchName}_${rank ?? index + 1}_${seed ?? "output"}`,
    label,
    branch: branchName,
    imageUrl: resolveBackendUrl(imageUrl, backendOrigin, ""),
    score: scores?.overall,
    confidence: scores?.overall,
    scores,
    isBest: Boolean(candidate.selected),
    candidateIndex: index,
    rank,
  };
};

const outputCandidateFromBranch = (
  branch: ApiRecord,
  branchName: "pretrained" | "geometry",
  index: number,
  backendOrigin: string,
): TryOnCandidate | null => {
  const imageUrl = asString(branch.image_url);
  if (!imageUrl) return null;

  const scores = normalizeBackendScores(branch.scores);
  return {
    id: `${branchName}_output`,
    label: branchName === "geometry" ? "Geometry output" : "Pretrained baseline",
    branch: branchName,
    imageUrl: resolveBackendUrl(imageUrl, backendOrigin, ""),
    score: scores?.overall,
    confidence: scores?.overall,
    scores,
    isBest: false,
    candidateIndex: index,
  };
};

const normalizeDelta = (rawDelta: unknown): TryOnComparisonDelta | undefined => {
  const delta = asRecord(rawDelta);
  if (!hasKeys(delta)) return undefined;

  return {
    total: asNumber(delta.total),
    object: asNumber(delta.object),
    person: asNumber(delta.person),
    artifact: asNumber(delta.artifact),
    winner: asString(delta.winner),
    reason: asString(delta.reason),
  };
};

const buildWarnings = (delta?: TryOnComparisonDelta): TryOnWarning[] => {
  if (!delta?.reason) return [];
  return [
    {
      code: "geometry_delta",
      message: delta.reason,
      severity: delta.winner === "geometry" ? "info" : "warning",
    },
  ];
};

const adaptBackendResult = (
  rawResult: unknown,
  status: ApiRecord,
  request: TryOnRequest,
  backendOrigin: string,
): NormalizedTryOnResponse => {
  const result = asRecord(rawResult);
  const pretrained = asRecord(result.pretrained);
  const geometry = asRecord(result.geometry);
  const inputs = asRecord(result.inputs);
  const delta = normalizeDelta(result.delta);

  const candidates: TryOnCandidate[] = [];
  const pretrainedCandidate = outputCandidateFromBranch(pretrained, "pretrained", candidates.length, backendOrigin);
  if (pretrainedCandidate) candidates.push(pretrainedCandidate);

  const geometryCandidates = asArray(geometry.candidates)
    .map((candidate, index) => candidateFromBackend(candidate, candidates.length + index, "geometry", backendOrigin))
    .filter(Boolean) as TryOnCandidate[];

  if (geometryCandidates.length > 0) {
    candidates.push(...geometryCandidates);
  } else {
    const geometryCandidate = outputCandidateFromBranch(geometry, "geometry", candidates.length, backendOrigin);
    if (geometryCandidate) candidates.push(geometryCandidate);
  }

  const selectedGeometry =
    candidates.find((candidate) => candidate.branch === "geometry" && candidate.isBest) ||
    candidates.find((candidate) => candidate.branch === "geometry");
  const selectedCandidate = selectedGeometry || pretrainedCandidate || candidates[0];

  if (!selectedCandidate) {
    throw new Error("Backend result did not include a generated image.");
  }

  candidates.forEach((candidate) => {
    candidate.isBest = candidate.id === selectedCandidate.id;
  });

  const comparison: TryOnComparison = {
    personImageUrl: asString(inputs.person_url)
      ? resolveBackendUrl(asString(inputs.person_url), backendOrigin, "")
      : undefined,
    itemImageUrl: asString(inputs.object_url)
      ? resolveBackendUrl(asString(inputs.object_url), backendOrigin, "")
      : undefined,
    objectClass: asString(inputs.object_class) || inferObjectClass(request),
    pretrained: branchSummary(pretrained, "pretrained", backendOrigin),
    geometry: branchSummary(geometry, "geometry", backendOrigin),
    delta,
  };

  return {
    result: {
      imageUrl: selectedCandidate.imageUrl,
      confidence: selectedCandidate.confidence ?? selectedCandidate.score,
      candidateId: selectedCandidate.id,
      candidateIndex: selectedCandidate.candidateIndex,
    },
    candidates,
    metadata: {
      jobId: asString(result.run_id) || asString(status.run_id),
      responseId: asString(result.run_id) || asString(status.run_id),
      generationTimeMs:
        asNumber(result.elapsed_seconds) !== undefined ? Math.round(asNumber(result.elapsed_seconds)! * 1000) : undefined,
      mode: request.mode,
      numCandidates: request.numCandidates,
      objectClass: comparison.objectClass,
      statusUrl: asString(status.status_url)
        ? resolveBackendUrl(asString(status.status_url), backendOrigin, "")
        : undefined,
      source: "api",
    },
    comparison,
    warnings: buildWarnings(delta),
  };
};

const pollCompareRun = async (
  initialStatus: unknown,
  request: TryOnRequest,
  compareUrl: string,
  signal?: AbortSignal,
): Promise<NormalizedTryOnResponse> => {
  let status = asRecord(initialStatus);
  const runId = asString(status.run_id);
  if (!runId) return normalizeTryOnResponse(initialStatus, request);

  const backendOrigin = backendOriginFrom(compareUrl);
  const startedAt = performance.now();

  while (performance.now() - startedAt < POLL_TIMEOUT_MS) {
    const state = asString(status.status);

    if (state === "complete") {
      const embeddedResult = status.result;
      const result =
        embeddedResult ||
        (await fetchJson(
          resolveBackendUrl(asString(status.result_url), backendOrigin, `/api/v1/runs/${runId}/result`),
          signal,
        ));
      return adaptBackendResult(result, status, request, backendOrigin);
    }

    if (state === "failed") {
      throw new Error(asString(status.message) || "Backend generation failed.");
    }

    await sleep(POLL_INTERVAL_MS, signal);
    status = asRecord(
      await fetchJson(resolveBackendUrl(asString(status.status_url), backendOrigin, `/api/v1/runs/${runId}`), signal),
    );
  }

  throw new Error("Backend generation timed out while waiting for the compare run.");
};

const postToTryOnApi = async (request: TryOnRequest, signal?: AbortSignal): Promise<NormalizedTryOnResponse> => {
  const compareUrl = TRYON_API_URL || apiUrl("/api/v1/runs/compare");
  const response = await fetch(compareUrl, {
    method: "POST",
    body: buildBackendFormData(request),
    signal,
  });

  if (!response.ok) {
    throw new Error(await readErrorMessage(response));
  }

  const rawResponse = await response.json();
  return pollCompareRun(rawResponse, request, compareUrl, signal);
};

const mockScores = (index: number) => {
  const overall = Math.max(0.62, 0.91 - index * 0.055);

  return {
    object_similarity: Math.max(0.58, overall - 0.01),
    person_preservation: Math.max(0.6, overall - 0.025),
    localization: Math.max(0.55, overall - 0.04),
    artifact: Math.min(0.92, overall - 0.02),
    overall,
  };
};

const buildMockCandidates = (count: number): TryOnCandidate[] =>
  Array.from({ length: count + 1 }, (_, index) => {
    const isPretrained = index === 0;
    const scores = mockScores(index);
    return {
      id: isPretrained ? "pretrained_mock" : `geometry_mock_${index}`,
      label: isPretrained ? "Pretrained baseline" : `Geometry candidate ${index}`,
      branch: isPretrained ? "pretrained" : "geometry",
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
      isBest: index === 1 || (count === 0 && index === 0),
      candidateIndex: index,
      rank: index,
    };
  });

const generateMockTryOn = async (request: TryOnRequest, signal?: AbortSignal): Promise<NormalizedTryOnResponse> => {
  const candidateCount = Math.min(5, Math.max(1, request.numCandidates || 1));
  await sleep(1800 + candidateCount * 450, signal);

  const candidates = buildMockCandidates(candidateCount);
  const bestCandidate = candidates.find((candidate) => candidate.isBest) || candidates[0];
  const objectClass = normalizeCategory(request.category) || "ring";
  const warnings: TryOnWarning[] = request.prompt?.toLowerCase().includes("left")
    ? []
    : [
        {
          code: "mock_prompt_specificity",
          message: "Demo warning: add a more specific prompt if placement needs to be constrained.",
          severity: "info",
        },
      ];

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
      objectClass,
      source: "mock",
    },
    comparison: {
      objectClass,
      pretrained: {
        label: "Pretrained",
        branch: "pretrained",
        imageUrl: mockResultImage,
        score: candidates[0]?.score,
        scores: candidates[0]?.scores,
        confidenceLabel: "medium",
        candidateCount: 1,
      },
      geometry: {
        label: "Pretrained + Geometry",
        branch: "geometry",
        imageUrl: bestCandidate.imageUrl,
        score: bestCandidate.score,
        scores: bestCandidate.scores,
        confidenceLabel: "high",
        candidateCount,
      },
      delta: {
        total: 0.07,
        object: 0.09,
        person: -0.01,
        artifact: 0.04,
        winner: "geometry",
        reason: "Geometry selected the candidate with stronger target-object evidence.",
      },
    },
    warnings,
  };
};

export const generateTryOn = async (
  request: TryOnRequest,
  signal?: AbortSignal,
): Promise<NormalizedTryOnResponse> => {
  if (USE_MOCK) {
    return generateMockTryOn(request, signal);
  }

  return postToTryOnApi(request, signal);
};
