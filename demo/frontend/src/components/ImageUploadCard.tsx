import { useCallback, useId, useRef, useState } from "react";
import type { DragEvent, KeyboardEvent, MouseEvent } from "react";
import { FileImage, ImagePlus, RefreshCw, Trash2, UploadCloud, XCircle } from "lucide-react";
import { Button } from "@/components/ui/button";
import { ACCEPTED_IMAGE_TYPES, MAX_IMAGE_SIZE_BYTES } from "@/lib/tryOnConfig";
import { cn } from "@/lib/utils";

const VALID_EXTENSIONS = ["png", "jpg", "jpeg", "webp"];

const formatFileSize = (bytes: number) => {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
};

const getExtension = (fileName: string) => fileName.split(".").pop()?.toLowerCase();

const validateFile = (file: File, maxSizeBytes: number) => {
  const isSupportedType =
    ACCEPTED_IMAGE_TYPES.includes(file.type) || VALID_EXTENSIONS.includes(getExtension(file.name) || "");

  if (!isSupportedType) {
    return "Unsupported file type. Please use PNG, JPG, JPEG, or WEBP.";
  }

  if (file.size > maxSizeBytes) {
    return `File is too large. Please use an image up to ${formatFileSize(maxSizeBytes)}.`;
  }

  return null;
};

interface ImageUploadCardProps {
  label: string;
  emptyTitle: string;
  hint: string;
  file: File | null;
  previewUrl: string | null;
  onFileChange: (file: File) => void;
  onRemove: () => void;
  step?: number;
  size?: "compact" | "large";
  categoryHint?: string;
}

export const ImageUploadCard = ({
  label,
  emptyTitle,
  hint,
  file,
  previewUrl,
  onFileChange,
  onRemove,
  step,
  size = "large",
  categoryHint,
}: ImageUploadCardProps) => {
  const inputId = useId();
  const inputRef = useRef<HTMLInputElement>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const openFileDialog = () => {
    if (inputRef.current) inputRef.current.value = "";
    inputRef.current?.click();
  };

  const handleFile = useCallback(
    (nextFile?: File) => {
      if (!nextFile) return;

      const validationError = validateFile(nextFile, MAX_IMAGE_SIZE_BYTES);
      if (validationError) {
        setError(validationError);
        return;
      }

      setError(null);
      onFileChange(nextFile);
    },
    [onFileChange],
  );

  const handleRemove = (event: MouseEvent<HTMLButtonElement>) => {
    event.stopPropagation();
    setError(null);
    onRemove();
    if (inputRef.current) inputRef.current.value = "";
  };

  const onDrop = (event: DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    setIsDragging(false);
    handleFile(event.dataTransfer.files?.[0]);
  };

  const onKeyDown = (event: KeyboardEvent<HTMLDivElement>) => {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      openFileDialog();
    }
  };

  const heightClass = size === "large" ? "min-h-[300px]" : "min-h-[240px]";

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between gap-3">
        <label htmlFor={inputId} className="flex items-center gap-2 text-sm font-semibold text-foreground">
          {step && (
            <span className="flex h-6 w-6 items-center justify-center rounded-md bg-primary/10 text-xs font-semibold text-primary">
              {step}
            </span>
          )}
          {label}
        </label>
        {file && (
          <span className="max-w-[46%] truncate text-xs text-muted-foreground" title={file.name}>
            {formatFileSize(file.size)}
          </span>
        )}
      </div>

      <div
        role="button"
        tabIndex={0}
        onClick={openFileDialog}
        onKeyDown={onKeyDown}
        onDragOver={(event) => {
          event.preventDefault();
          setIsDragging(true);
        }}
        onDragLeave={() => setIsDragging(false)}
        onDrop={onDrop}
        className={cn(
          "group relative flex cursor-pointer flex-col overflow-hidden rounded-lg border border-dashed border-border bg-card transition-smooth hover:border-primary/60 hover:bg-secondary/50 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring",
          heightClass,
          isDragging && "border-primary bg-primary/5 shadow-glow",
          error && "border-destructive/70",
        )}
        aria-label={file ? `Replace ${label}` : emptyTitle}
      >
        {previewUrl ? (
          <>
            <div className="flex min-h-0 flex-1 items-center justify-center bg-[radial-gradient(circle_at_top,hsl(var(--secondary)),transparent_55%)] p-3">
              <img src={previewUrl} alt={`${label} preview`} className="max-h-full max-w-full object-contain" />
            </div>
            <div className="border-t border-border bg-background/95 p-3">
              <div className="flex min-w-0 items-center justify-between gap-3">
                <div className="flex min-w-0 items-center gap-2">
                  <FileImage className="h-4 w-4 shrink-0 text-primary" />
                  <span className="truncate text-sm font-medium text-foreground" title={file?.name}>
                    {file?.name}
                  </span>
                </div>
                <div className="flex shrink-0 items-center gap-1">
                  <Button
                    type="button"
                    variant="ghost"
                    size="icon"
                    className="h-8 w-8"
                    onClick={(event) => {
                      event.stopPropagation();
                      openFileDialog();
                    }}
                    aria-label={`Replace ${label}`}
                  >
                    <RefreshCw className="h-4 w-4" />
                  </Button>
                  <Button
                    type="button"
                    variant="ghost"
                    size="icon"
                    className="h-8 w-8 text-destructive hover:text-destructive"
                    onClick={handleRemove}
                    aria-label={`Remove ${label}`}
                  >
                    <Trash2 className="h-4 w-4" />
                  </Button>
                </div>
              </div>
            </div>
          </>
        ) : (
          <div className="flex flex-1 flex-col items-center justify-center gap-4 px-6 py-8 text-center">
            <div className="flex h-14 w-14 items-center justify-center rounded-lg bg-primary text-primary-foreground shadow-elegant transition-smooth group-hover:scale-105">
              <UploadCloud className="h-6 w-6" />
            </div>
            <div className="space-y-1">
              <p className="text-sm font-semibold text-foreground">{emptyTitle}</p>
              <p className="text-xs leading-5 text-muted-foreground">{hint}</p>
            </div>
            <div className="inline-flex items-center gap-2 rounded-md border border-border bg-secondary/70 px-3 py-1.5 text-xs text-muted-foreground">
              <ImagePlus className="h-3.5 w-3.5" />
              PNG, JPG, JPEG, WEBP up to 10MB
            </div>
            {categoryHint && <p className="max-w-sm text-xs leading-5 text-muted-foreground">{categoryHint}</p>}
          </div>
        )}
      </div>

      {error && (
        <div className="flex items-start gap-2 rounded-md border border-destructive/30 bg-destructive/10 px-3 py-2 text-xs text-destructive">
          <XCircle className="mt-0.5 h-3.5 w-3.5 shrink-0" />
          <span>{error}</span>
        </div>
      )}

      <input
        ref={inputRef}
        id={inputId}
        type="file"
        accept="image/png,image/jpeg,image/webp"
        className="sr-only"
        onChange={(event) => handleFile(event.target.files?.[0])}
      />
    </div>
  );
};
