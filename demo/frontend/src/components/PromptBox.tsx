import { Plus, WandSparkles } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { MAX_PROMPT_LENGTH } from "@/lib/tryOnConfig";

const SUGGESTIONS = [
  "wear naturally",
  "keep original pose",
  "preserve face and hands",
  "high quality try-on",
  "front view",
];

interface PromptBoxProps {
  value: string;
  onChange: (value: string) => void;
}

export const PromptBox = ({ value, onChange }: PromptBoxProps) => {
  const appendSuggestion = (suggestion: string) => {
    const trimmedValue = value.trim();
    const nextValue = trimmedValue ? `${trimmedValue}, ${suggestion}` : suggestion;
    onChange(nextValue.slice(0, MAX_PROMPT_LENGTH));
  };

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between gap-3">
        <Label htmlFor="tryon-prompt" className="flex items-center gap-2 text-sm font-semibold">
          <WandSparkles className="h-4 w-4 text-primary" />
          Optional prompt
        </Label>
        <span className="text-xs text-muted-foreground">
          {value.length}/{MAX_PROMPT_LENGTH}
        </span>
      </div>

      <Textarea
        id="tryon-prompt"
        value={value}
        maxLength={MAX_PROMPT_LENGTH}
        onChange={(event) => onChange(event.target.value)}
        placeholder="Optional prompt: wear this watch on the left wrist; hold this bag naturally; try these glasses on the face"
        className="min-h-[104px] resize-none rounded-lg bg-card leading-6"
      />

      <div className="flex flex-wrap gap-2">
        {SUGGESTIONS.map((suggestion) => (
          <Button
            key={suggestion}
            type="button"
            variant="secondary"
            size="sm"
            className="h-8 rounded-md px-2.5 text-xs"
            onClick={() => appendSuggestion(suggestion)}
          >
            <Plus className="h-3.5 w-3.5" />
            {suggestion}
          </Button>
        ))}
      </div>
    </div>
  );
};
