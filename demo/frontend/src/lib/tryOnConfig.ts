import type { TryOnCategory, TryOnMode } from "@/types/tryOn";

export const MAX_PROMPT_LENGTH = 300;
export const MAX_IMAGE_SIZE_BYTES = 10 * 1024 * 1024;
export const ACCEPTED_IMAGE_TYPES = ["image/png", "image/jpeg", "image/jpg", "image/webp"];

export const MODE_CONFIG: Record<TryOnMode, { label: string; description: string; candidates: number }> = {
  fast: {
    label: "Fast",
    description: "6 steps, K = 1",
    candidates: 1,
  },
  balanced: {
    label: "Balanced",
    description: "10 steps, K = 1",
    candidates: 1,
  },
  high_quality: {
    label: "High quality",
    description: "16 steps, K = 1",
    candidates: 1,
  },
};

export const CATEGORY_OPTIONS: { value: TryOnCategory; label: string }[] = [
  { value: "auto", label: "Auto infer" },
  { value: "top clothes", label: "Top clothes" },
  { value: "bottom clothes", label: "Bottom clothes" },
  { value: "dress", label: "Dress" },
  { value: "shoe", label: "Shoe" },
  { value: "earrings", label: "Earrings" },
  { value: "bracelet", label: "Bracelet" },
  { value: "necklace", label: "Necklace" },
  { value: "ring", label: "Ring" },
  { value: "sunglasses", label: "Sunglasses" },
  { value: "glasses", label: "Glasses" },
  { value: "belt", label: "Belt" },
  { value: "bag", label: "Bag" },
  { value: "hat", label: "Hat" },
  { value: "tie", label: "Tie" },
  { value: "bow tie", label: "Bow tie" },
];
