import type { TryOnWarning } from "@/types/tryOn";

interface ConfidencePanelProps {
  warnings: TryOnWarning[];
}

export const ConfidencePanel = ({ warnings }: ConfidencePanelProps) => {
  if (warnings.length === 0) return null;

  return (
    <section className="space-y-3 rounded-lg border border-border bg-card p-4 shadow-soft">
      <div className="space-y-2">
        {warnings.map((warning, index) => (
          <div
            key={`${warning.code || "warning"}-${index}`}
            className="rounded-lg border border-amber-200 bg-amber-50 px-3 py-2 text-xs leading-5 text-amber-800"
          >
            {warning.message}
          </div>
        ))}
      </div>
    </section>
  );
};
