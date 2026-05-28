import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { MODE_CONFIG } from "@/lib/tryOnConfig";
import { useObjectUrlPreview } from "@/hooks/useObjectUrlPreview";
import { generateTryOn } from "@/services/tryOnApi";
import {
  completePipelineSteps,
  createPipelineSteps,
  failPipelineSteps,
  getPipelineProgress,
  setPipelineStepStatus,
} from "@/services/pipelineProgress";
import type {
  AdvancedTryOnSettings,
  NormalizedTryOnResponse,
  PipelineStep,
  TryOnCandidate,
  TryOnCategory,
  TryOnComparison,
  TryOnMetadata,
  TryOnMode,
  TryOnResult,
  TryOnWarning,
} from "@/types/tryOn";

const DEFAULT_MODE: TryOnMode = "fast";
const GENERATION_TIMEOUT_MS = 15 * 60 * 1000;

const defaultSettings: AdvancedTryOnSettings = {
  guidanceScale: 30,
  seed: undefined,
  enableReranker: true,
  enableRefiner: true,
};

const getFriendlyError = (error: unknown, timedOut: boolean) => {
  if (timedOut) return "API timeout. Please try again or switch to Fast mode.";
  if (error instanceof Error && error.name === "AbortError") return "Generation was canceled.";
  if (error instanceof Error && error.message) return error.message;
  return "Generation failed. Please try again.";
};

export const useTryOnDemo = () => {
  const [personFile, setPersonFile] = useState<File | null>(null);
  const [itemFile, setItemFile] = useState<File | null>(null);
  const [prompt, setPrompt] = useState("");
  const [category, setCategory] = useState<TryOnCategory>("auto");
  const [mode, setModeState] = useState<TryOnMode>(DEFAULT_MODE);
  const [numCandidates, setNumCandidates] = useState(MODE_CONFIG[DEFAULT_MODE].candidates);
  const [settings, setSettings] = useState<AdvancedTryOnSettings>(defaultSettings);
  const [isGenerating, setIsGenerating] = useState(false);
  const [pipelineSteps, setPipelineSteps] = useState<PipelineStep[]>(createPipelineSteps);
  const [result, setResult] = useState<TryOnResult | null>(null);
  const [candidates, setCandidates] = useState<TryOnCandidate[]>([]);
  const [selectedCandidateId, setSelectedCandidateId] = useState<string | null>(null);
  const [warnings, setWarnings] = useState<TryOnWarning[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [metadata, setMetadata] = useState<TryOnMetadata | null>(null);
  const [comparison, setComparison] = useState<TryOnComparison | null>(null);

  const pipelineTimersRef = useRef<number[]>([]);
  const abortControllerRef = useRef<AbortController | null>(null);

  const personPreviewUrl = useObjectUrlPreview(personFile);
  const itemPreviewUrl = useObjectUrlPreview(itemFile);

  const selectedCandidate = useMemo(
    () => candidates.find((candidate) => candidate.id === selectedCandidateId) || candidates[0] || null,
    [candidates, selectedCandidateId],
  );

  const confidence = selectedCandidate?.confidence ?? selectedCandidate?.score ?? result?.confidence;
  const canGenerate = Boolean(personFile && itemFile && !isGenerating);
  const hasResult = Boolean(result?.imageUrl || selectedCandidate?.imageUrl);
  const pipelineProgress = getPipelineProgress(pipelineSteps);

  const clearPipelineTimers = useCallback(() => {
    pipelineTimersRef.current.forEach((timerId) => window.clearTimeout(timerId));
    pipelineTimersRef.current = [];
  }, []);

  const startPipelineSimulation = useCallback(() => {
    clearPipelineTimers();
    setPipelineSteps(setPipelineStepStatus(createPipelineSteps(), 0, "running"));

    const stepDelays = [650, 1200, 1850, 2550, 3300, 4050];
    pipelineTimersRef.current = stepDelays.map((delay, index) =>
      window.setTimeout(() => {
        setPipelineSteps((steps) => setPipelineStepStatus(steps, Math.min(index + 1, steps.length - 2), "running"));
      }, delay),
    );
  }, [clearPipelineTimers]);

  const completePipeline = useCallback(() => {
    clearPipelineTimers();
    setPipelineSteps((steps) => completePipelineSteps(steps));
  }, [clearPipelineTimers]);

  const failPipeline = useCallback(() => {
    clearPipelineTimers();
    setPipelineSteps((steps) => failPipelineSteps(steps));
  }, [clearPipelineTimers]);

  const setMode = useCallback((nextMode: TryOnMode) => {
    setModeState(nextMode);
    setNumCandidates(MODE_CONFIG[nextMode].candidates);
  }, []);

  const setCandidateCount = useCallback((nextCount: number) => {
    const boundedCount = Math.min(1, Math.max(1, Number.isFinite(nextCount) ? nextCount : 1));
    setNumCandidates(boundedCount);
  }, []);

  const cancelGenerate = useCallback(() => {
    abortControllerRef.current?.abort();
  }, []);

  const clearOutputs = useCallback(() => {
    setResult(null);
    setCandidates([]);
    setSelectedCandidateId(null);
    setWarnings([]);
    setMetadata(null);
    setComparison(null);
    setError(null);
    setPipelineSteps(createPipelineSteps());
  }, []);

  const reset = useCallback(() => {
    abortControllerRef.current?.abort();
    clearPipelineTimers();
    setPersonFile(null);
    setItemFile(null);
    setPrompt("");
    setCategory("auto");
    setModeState(DEFAULT_MODE);
    setNumCandidates(MODE_CONFIG[DEFAULT_MODE].candidates);
    setSettings(defaultSettings);
    setIsGenerating(false);
    clearOutputs();
  }, [clearOutputs, clearPipelineTimers]);

  const applyResponse = useCallback(
    (response: NormalizedTryOnResponse, measuredGenerationTimeMs: number) => {
      const bestCandidate =
        response.candidates.find((candidate) => candidate.id === response.result.candidateId) ||
        response.candidates.find((candidate) => candidate.isBest) ||
        response.candidates[0];

      setResult(response.result);
      setCandidates(response.candidates);
      setSelectedCandidateId(bestCandidate?.id || null);
      setWarnings(response.warnings);
      setComparison(response.comparison || null);
      setMetadata({
        ...response.metadata,
        generationTimeMs: response.metadata.generationTimeMs ?? measuredGenerationTimeMs,
        mode: response.metadata.mode ?? mode,
        numCandidates: response.metadata.numCandidates ?? numCandidates,
      });
    },
    [mode, numCandidates],
  );

  const generate = useCallback(async () => {
    if (!personFile) {
      setError("Please upload a person image.");
      return;
    }

    if (!itemFile) {
      setError("Please upload a garment or accessory image.");
      return;
    }

    abortControllerRef.current?.abort();
    const controller = new AbortController();
    abortControllerRef.current = controller;
    let didTimeout = false;
    const timeoutId = window.setTimeout(() => {
      didTimeout = true;
      controller.abort();
    }, GENERATION_TIMEOUT_MS);

    setIsGenerating(true);
    setError(null);
    setWarnings([]);
    setResult(null);
    setCandidates([]);
    setSelectedCandidateId(null);
    setMetadata(null);
    setComparison(null);
    startPipelineSimulation();

    const start = performance.now();
    try {
      const response = await generateTryOn(
        {
          personFile,
          itemFile,
          prompt: prompt.trim() || undefined,
          category,
          mode,
          numCandidates,
          guidanceScale: settings.guidanceScale,
          seed: settings.seed,
          enableReranker: settings.enableReranker,
          enableRefiner: settings.enableRefiner,
        },
        controller.signal,
      );

      applyResponse(response, Math.round(performance.now() - start));
      completePipeline();
    } catch (caughtError) {
      const message = getFriendlyError(caughtError, didTimeout);
      setError(message);
      failPipeline();
    } finally {
      window.clearTimeout(timeoutId);
      setIsGenerating(false);
      abortControllerRef.current = null;
    }
  }, [
    applyResponse,
    category,
    completePipeline,
    failPipeline,
    itemFile,
    mode,
    numCandidates,
    personFile,
    prompt,
    settings.enableRefiner,
    settings.enableReranker,
    settings.guidanceScale,
    settings.seed,
    startPipelineSimulation,
  ]);

  const downloadSelectedResult = useCallback(() => {
    const imageUrl = selectedCandidate?.imageUrl || result?.imageUrl;
    if (!imageUrl) return;

    const link = document.createElement("a");
    link.href = imageUrl;
    link.download = `omnitry-result-${selectedCandidate?.candidateIndex ?? 0}.jpg`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  }, [result?.imageUrl, selectedCandidate]);

  useEffect(() => {
    return () => {
      abortControllerRef.current?.abort();
      clearPipelineTimers();
    };
  }, [clearPipelineTimers]);

  return {
    personFile,
    personPreviewUrl,
    itemFile,
    itemPreviewUrl,
    prompt,
    category,
    mode,
    numCandidates,
    settings,
    isGenerating,
    pipelineSteps,
    pipelineProgress,
    result,
    candidates,
    selectedCandidate,
    selectedCandidateId,
    confidence,
    warnings,
    error,
    metadata,
    comparison,
    canGenerate,
    hasResult,
    setPersonFile,
    setItemFile,
    setPrompt,
    setCategory,
    setMode,
    setNumCandidates: setCandidateCount,
    setSettings,
    setSelectedCandidateId,
    generate,
    cancelGenerate,
    reset,
    clearOutputs,
    downloadSelectedResult,
  };
};
