import type { ReactNode } from "react";
import { cn } from "@/lib/utils";

interface EmptyStateProps {
  icon: ReactNode;
  title: string;
  description?: string;
  className?: string;
}

export const EmptyState = ({ icon, title, description, className }: EmptyStateProps) => (
  <div className={cn("flex flex-col items-center justify-center gap-3 p-6 text-center", className)}>
    <div className="flex h-12 w-12 items-center justify-center rounded-lg bg-secondary text-primary">{icon}</div>
    <div className="space-y-1">
      <p className="text-sm font-semibold text-foreground">{title}</p>
      {description && <p className="mx-auto max-w-xs text-xs leading-5 text-muted-foreground">{description}</p>}
    </div>
  </div>
);
