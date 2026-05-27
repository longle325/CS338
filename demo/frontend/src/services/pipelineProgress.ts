import type { PipelineStep, PipelineStepStatus } from "@/types/tryOn";

export const PIPELINE_STEP_DEFINITIONS: Omit<PipelineStep, "status">[] = [
  {
    id: "parse_person",
    label: "Parsing person image",
    description: "Detecting pose, face, hands, and visible regions.",
  },
  {
    id: "parse_item",
    label: "Parsing item image",
    description: "Estimating item category, silhouette, and texture cues.",
  },
  {
    id: "plan_geometry",
    label: "Planning affordance / geometry",
    description: "Choosing placement without manual masks or boxes.",
  },
  {
    id: "generate_candidates",
    label: "Generating try-on candidates",
    description: "Creating K mask-free try-on hypotheses.",
  },
  {
    id: "qa_reranking",
    label: "QA reranking",
    description: "Scoring identity preservation and object localization.",
  },
  {
    id: "refine_output",
    label: "Refining final output",
    description: "Polishing artifacts and selecting the best result.",
  },
  {
    id: "completed",
    label: "Completed",
    description: "Best candidate, confidence, and warnings are ready.",
  },
];

export const createPipelineSteps = (): PipelineStep[] =>
  PIPELINE_STEP_DEFINITIONS.map((step) => ({
    ...step,
    status: "pending",
  }));

export const setPipelineStepStatus = (
  steps: PipelineStep[],
  activeIndex: number,
  activeStatus: PipelineStepStatus,
): PipelineStep[] =>
  steps.map((step, index) => {
    if (index < activeIndex) return { ...step, status: "done" };
    if (index === activeIndex) return { ...step, status: activeStatus };
    return { ...step, status: "pending" };
  });

export const completePipelineSteps = (steps: PipelineStep[]): PipelineStep[] =>
  steps.map((step) => ({ ...step, status: "done" }));

export const failPipelineSteps = (steps: PipelineStep[]): PipelineStep[] => {
  const runningIndex = steps.findIndex((step) => step.status === "running");
  const errorIndex = runningIndex >= 0 ? runningIndex : Math.max(0, steps.findIndex((step) => step.status === "pending"));

  return steps.map((step, index) => {
    if (index < errorIndex) return { ...step, status: "done" };
    if (index === errorIndex) return { ...step, status: "error" };
    return { ...step, status: "pending" };
  });
};

export const getPipelineProgress = (steps: PipelineStep[]): number => {
  if (steps.length === 0) return 0;
  const completed = steps.filter((step) => step.status === "done").length;
  const running = steps.some((step) => step.status === "running") ? 0.5 : 0;
  return Math.min(100, Math.round(((completed + running) / steps.length) * 100));
};
