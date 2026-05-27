import { AlertCircle, RefreshCw } from "lucide-react";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";

interface ErrorAlertProps {
  message: string;
  onRetry?: () => void;
}

export const ErrorAlert = ({ message, onRetry }: ErrorAlertProps) => (
  <Alert variant="destructive" className="rounded-lg">
    <AlertCircle className="h-4 w-4" />
    <AlertTitle>Generation failed</AlertTitle>
    <AlertDescription className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
      <span>{message || "Generation failed. Please try again."}</span>
      {onRetry && (
        <Button type="button" size="sm" variant="outline" className="shrink-0" onClick={onRetry}>
          <RefreshCw className="h-4 w-4" />
          Retry
        </Button>
      )}
    </AlertDescription>
  </Alert>
);
