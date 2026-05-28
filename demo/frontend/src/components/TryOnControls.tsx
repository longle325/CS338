import { ChevronDown, Gauge, Play, RotateCcw, Settings2, SlidersHorizontal, Square } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { CATEGORY_OPTIONS, MODE_CONFIG } from "@/lib/tryOnConfig";
import { cn } from "@/lib/utils";
import type { AdvancedTryOnSettings, TryOnCategory, TryOnMode } from "@/types/tryOn";

interface TryOnControlsProps {
  personReady: boolean;
  itemReady: boolean;
  canGenerate: boolean;
  isGenerating: boolean;
  hasResult: boolean;
  mode: TryOnMode;
  numCandidates: number;
  category: TryOnCategory;
  settings: AdvancedTryOnSettings;
  onModeChange: (mode: TryOnMode) => void;
  onNumCandidatesChange: (count: number) => void;
  onCategoryChange: (category: TryOnCategory) => void;
  onSettingsChange: (settings: AdvancedTryOnSettings) => void;
  onGenerate: () => void;
  onCancel: () => void;
  onReset: () => void;
}

export const TryOnControls = ({
  personReady,
  itemReady,
  canGenerate,
  isGenerating,
  hasResult,
  mode,
  numCandidates,
  category,
  settings,
  onModeChange,
  onNumCandidatesChange,
  onCategoryChange,
  onSettingsChange,
  onGenerate,
  onCancel,
  onReset,
}: TryOnControlsProps) => {
  const missingText = !personReady
    ? "Please upload a person image."
    : !itemReady
      ? "Please upload a garment or accessory image."
      : "";

  return (
    <div className="space-y-4">
      <div className="space-y-3">
        <div className="flex items-center gap-2 text-sm font-semibold text-foreground">
          <Gauge className="h-4 w-4 text-primary" />
          Generate mode
        </div>
        <div className="grid grid-cols-3 gap-2">
          {(Object.keys(MODE_CONFIG) as TryOnMode[]).map((modeKey) => {
            const isActive = mode === modeKey;
            const config = MODE_CONFIG[modeKey];

            return (
              <button
                key={modeKey}
                type="button"
                aria-pressed={isActive}
                onClick={() => onModeChange(modeKey)}
                className={cn(
                  "min-h-[78px] rounded-lg border bg-card p-3 text-left transition-smooth focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring",
                  isActive ? "border-primary shadow-glow" : "border-border hover:border-primary/50 hover:bg-secondary/50",
                )}
              >
                <span className="block text-sm font-semibold text-foreground">{config.label}</span>
                <span className="mt-1 block text-xs leading-4 text-muted-foreground">{config.description}</span>
              </button>
            );
          })}
        </div>
      </div>

      <div className="space-y-2">
        <Label htmlFor="item-category" className="text-sm font-semibold">
          Item category
        </Label>
        <Select value={category} onValueChange={(value) => onCategoryChange(value as TryOnCategory)}>
          <SelectTrigger id="item-category" className="h-11 rounded-lg bg-card">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            {CATEGORY_OPTIONS.map((option) => (
              <SelectItem key={option.value} value={option.value}>
                {option.label}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>

      <Collapsible>
        <CollapsibleTrigger asChild>
          <Button type="button" variant="outline" className="h-10 w-full justify-between rounded-lg">
            <span className="inline-flex items-center gap-2">
              <SlidersHorizontal className="h-4 w-4" />
              Advanced settings
            </span>
            <ChevronDown className="h-4 w-4" />
          </Button>
        </CollapsibleTrigger>
        <CollapsibleContent className="mt-3 space-y-4 rounded-lg border border-border bg-secondary/40 p-4">
          <div className="grid grid-cols-2 gap-3">
            <div className="space-y-2">
              <Label htmlFor="candidate-count">Candidates K</Label>
              <Input
                id="candidate-count"
                type="number"
                min={1}
                max={5}
                value={numCandidates}
                onChange={(event) => onNumCandidatesChange(Number(event.target.value))}
                className="rounded-lg bg-card"
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="guidance-scale">Guidance scale</Label>
              <Input
                id="guidance-scale"
                type="number"
                step={1}
                min={1}
                max={50}
                value={settings.guidanceScale ?? ""}
                onChange={(event) =>
                  onSettingsChange({
                    ...settings,
                    guidanceScale: event.target.value === "" ? undefined : Number(event.target.value),
                  })
                }
                className="rounded-lg bg-card"
              />
            </div>
          </div>

          <div className="space-y-2">
            <Label htmlFor="seed">Seed</Label>
            <Input
              id="seed"
              type="number"
              placeholder="Random"
              value={settings.seed ?? ""}
              onChange={(event) =>
                onSettingsChange({
                  ...settings,
                  seed: event.target.value === "" ? undefined : Number(event.target.value),
                })
              }
              className="rounded-lg bg-card"
            />
          </div>
        </CollapsibleContent>
      </Collapsible>

      <div className="space-y-2">
        {isGenerating ? (
          <Button type="button" variant="outline" className="h-12 w-full rounded-lg" onClick={onCancel}>
            <Square className="h-4 w-4" />
            Cancel generation
          </Button>
        ) : (
          <Button
            type="button"
            className="h-12 w-full rounded-lg bg-primary text-base font-semibold text-primary-foreground shadow-elegant hover:bg-primary/90"
            disabled={!canGenerate}
            onClick={onGenerate}
          >
            <Play className="h-4 w-4 fill-current" />
            {hasResult ? "Regenerate" : "Try On"}
          </Button>
        )}

        <Button type="button" variant="ghost" className="h-10 w-full rounded-lg" onClick={onReset}>
          <RotateCcw className="h-4 w-4" />
          Reset
        </Button>

        {missingText && <p className="text-center text-xs text-muted-foreground">{missingText}</p>}
      </div>

      <div className="flex items-start gap-2 rounded-lg border border-primary/15 bg-primary/5 p-3 text-xs leading-5 text-muted-foreground">
        <Settings2 className="mt-0.5 h-4 w-4 shrink-0 text-primary" />
        <span>Mask-free try-on: no manual mask or bounding box is required.</span>
      </div>
    </div>
  );
};
