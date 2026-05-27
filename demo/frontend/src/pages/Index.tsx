import { Cpu, Layers3, Shirt, Sparkles } from "lucide-react";
import { ImageUploadCard } from "@/components/ImageUploadCard";
import { OutputPanel } from "@/components/OutputPanel";
import { PreviewPanel } from "@/components/PreviewPanel";
import { PromptBox } from "@/components/PromptBox";
import { TryOnControls } from "@/components/TryOnControls";
import { Badge } from "@/components/ui/badge";
import { useTryOnDemo } from "@/hooks/useTryOnDemo";

const Index = () => {
  const demo = useTryOnDemo();

  return (
    <div className="min-h-screen">
      <header className="border-b border-border bg-background/90 backdrop-blur">
        <div className="container flex flex-col gap-4 py-5 md:flex-row md:items-center md:justify-between">
          <div className="flex items-center gap-4">
            <div className="flex h-11 w-11 items-center justify-center rounded-lg bg-primary text-primary-foreground shadow-elegant">
              <Shirt className="h-5 w-5" />
            </div>
            <div>
              <div className="flex flex-wrap items-center gap-2">
                <h1 className="text-2xl font-bold tracking-tight text-foreground md:text-3xl">
                  OmniTry++ Virtual Try-On
                </h1>
                <Badge variant="secondary" className="rounded-md bg-emerald-50 text-emerald-700">
                  Mask-free
                </Badge>
              </div>
              <p className="mt-1 text-sm text-muted-foreground">
                Person Image + Item Image + Optional Prompt to K Candidates to QA Reranking to Best Result
              </p>
            </div>
          </div>

          <div className="flex flex-wrap items-center gap-2">
            <Badge variant="outline" className="rounded-md border-primary/20 bg-primary/5 text-primary">
              <Cpu className="mr-1 h-3.5 w-3.5" />
              API-ready
            </Badge>
            <Badge variant="outline" className="rounded-md border-emerald-200 bg-emerald-50 text-emerald-700">
              <Sparkles className="mr-1 h-3.5 w-3.5" />
              Demo pipeline
            </Badge>
          </div>
        </div>
      </header>

      <main className="container py-6">
        <div className="grid gap-5 lg:grid-cols-[minmax(300px,380px)_minmax(0,1fr)]">
          <aside className="space-y-5 rounded-lg border border-border bg-card p-4 shadow-soft lg:sticky lg:top-5 lg:self-start">
            <div className="flex items-start justify-between gap-3">
              <div>
                <h2 className="text-lg font-semibold tracking-tight text-foreground">Input panel</h2>
                <p className="mt-1 text-sm text-muted-foreground">Upload inputs and tune generation.</p>
              </div>
              <Layers3 className="h-5 w-5 text-primary" />
            </div>

            <ImageUploadCard
              step={1}
              label="Person image"
              emptyTitle="Upload a person image"
              hint="Drag and drop or click to select a full-body or half-body subject image."
              file={demo.personFile}
              previewUrl={demo.personPreviewUrl}
              onFileChange={demo.setPersonFile}
              onRemove={() => demo.setPersonFile(null)}
            />

            <ImageUploadCard
              step={2}
              label="Item image"
              emptyTitle="Upload a garment or accessory"
              hint="Drag and drop or click to select the garment, accessory, shoes, bag, watch, glasses, hat, or holdable item."
              categoryHint="Category can be inferred automatically, or set manually below."
              file={demo.itemFile}
              previewUrl={demo.itemPreviewUrl}
              onFileChange={demo.setItemFile}
              onRemove={() => demo.setItemFile(null)}
              size="compact"
            />

            <PromptBox value={demo.prompt} onChange={demo.setPrompt} />

            <TryOnControls
              personReady={Boolean(demo.personFile)}
              itemReady={Boolean(demo.itemFile)}
              canGenerate={demo.canGenerate}
              isGenerating={demo.isGenerating}
              hasResult={demo.hasResult}
              mode={demo.mode}
              numCandidates={demo.numCandidates}
              category={demo.category}
              settings={demo.settings}
              onModeChange={demo.setMode}
              onNumCandidatesChange={demo.setNumCandidates}
              onCategoryChange={demo.setCategory}
              onSettingsChange={demo.setSettings}
              onGenerate={demo.generate}
              onCancel={demo.cancelGenerate}
              onReset={demo.reset}
            />
          </aside>

          <div className="min-w-0 space-y-5">
            <PreviewPanel
              personPreviewUrl={demo.personPreviewUrl}
              itemPreviewUrl={demo.itemPreviewUrl}
              personFile={demo.personFile}
              itemFile={demo.itemFile}
              pipelineSteps={demo.pipelineSteps}
              pipelineProgress={demo.pipelineProgress}
              isGenerating={demo.isGenerating}
            />

            <OutputPanel
              personFile={demo.personFile}
              itemFile={demo.itemFile}
              personPreviewUrl={demo.personPreviewUrl}
              itemPreviewUrl={demo.itemPreviewUrl}
              prompt={demo.prompt}
              mode={demo.mode}
              numCandidates={demo.numCandidates}
              isGenerating={demo.isGenerating}
              selectedCandidate={demo.selectedCandidate}
              selectedCandidateId={demo.selectedCandidateId}
              candidates={demo.candidates}
              confidence={demo.confidence}
              warnings={demo.warnings}
              error={demo.error}
              metadata={demo.metadata}
              canGenerate={demo.canGenerate}
              onGenerate={demo.generate}
              onDownload={demo.downloadSelectedResult}
              onSelectCandidate={demo.setSelectedCandidateId}
            />
          </div>
        </div>
      </main>
    </div>
  );
};

export default Index;
