# Live Inference Demo Components

This demo should focus on a direct, classroom-friendly comparison:

> Same person image, same object image, same pretrained OmniTry weights. One branch runs the plain pretrained pipeline. The other branch adds Geo-Affordance Candidate Selection during inference.

The goal is not to make a benchmark dashboard first. The goal is to make the improvement easy to see during a live presentation.

## Main Screen

### 1. Input Panel

Purpose: collect exactly the inputs needed for one try-on comparison.

Components:

- Person image uploader
- Object image uploader
- Object class selector
- Optional prompt input
- Run Compare button
- Small advanced settings drawer

Object classes:

- ring
- bracelet
- necklace
- sunglasses
- eyeglasses
- earring
- shoe
- hat
- bag

Advanced settings:

- seed
- diffusion steps
- guidance scale
- geometry candidate count
- output resolution / max area

Default behavior:

- baseline candidate count: 1
- geometry candidate count: 2 or 3
- same seed family for both branches
- same pretrained checkpoint for both branches

## Inference Branches

### 2. Pretrained Branch

Purpose: show what the original model does without geometry assistance.

Pipeline:

1. Build the standard OmniTry prompt.
2. Run the pretrained model once.
3. Return the generated image.
4. Score the result using the same lightweight scoring function as the geometry branch.

Displayed label:

`Pretrained`

What to show:

- output image
- total score
- object preservation score
- person preservation score
- artifact score

### 3. Pretrained + Geometry Branch

Purpose: show the same pretrained model with affordance-aware inference.

Pipeline:

1. Build the geometry-aware prompt.
2. Generate multiple candidates.
3. Score each candidate using object, person, and artifact terms.
4. Select the best candidate.
5. Return the selected image and candidate diagnostics.

Displayed label:

`Pretrained + Geometry`

What to show:

- selected output image
- selected candidate id
- total score
- object preservation score
- person preservation score
- artifact score
- candidate strip
- affordance/zoom evidence

## Output Comparison

### 4. Side-By-Side Result Viewer

Purpose: make the comparison obvious in one glance.

Layout:

- Column 1: person input
- Column 2: object input
- Column 3: pretrained output
- Column 4: pretrained + geometry output

Design notes:

- Keep all images aligned with the same aspect ratio.
- Keep labels short.
- Use a visual highlight around the better total score.
- Do not hide the baseline output, even when it looks bad.

### 5. Score Comparison Strip

Purpose: turn visual quality into a reproducible explanation.

Metrics:

- total
- object
- person
- artifact

Rows:

- Pretrained
- Pretrained + Geometry
- Delta

Interpretation:

- Object score explains whether the target item survives.
- Person score explains whether the identity/body is preserved.
- Artifact score explains image health and obvious generation damage.
- Total score is the weighted selection signal.

### 6. Zoom Evidence Panel

Purpose: show why the geometry branch helps small objects.

For each object class, crop the most relevant region:

- ring: hand/finger region
- bracelet: wrist region
- necklace: neck/chest region
- sunglasses/eyeglasses: face/eye region
- earring: ear/side-face region
- shoe: foot/lower-leg region
- hat: head region
- bag: hand/shoulder/torso region

Panel content:

- pretrained zoom crop
- geometry zoom crop
- optional object reference crop

This is the strongest visual evidence for the presentation because many small-object failures are hard to see in the full-body image.

### 7. Geometry Candidate Panel

Purpose: explain that the method is inference-time candidate selection, not hidden retraining.

Content:

- candidate thumbnails
- candidate scores
- selected candidate badge
- short reason string

Example reason strings:

- `Selected because object score improved while person preservation stayed stable.`
- `Selected because the object appears in the expected affordance region.`
- `Rejected candidates damaged the face or erased the target item.`

## Presentation Mode

### 8. One-Click Demo Examples

Purpose: reduce risk during the live demo.

Add a small example gallery with preselected inputs:

- ring on different women
- sunglasses on different men
- shoes on standing person
- bracelet on visible wrist
- necklace on front-facing person

Each example should load person image, object image, and object class into the input panel. The presenter can then click `Run Compare`.

### 9. Curated Output Gallery

Purpose: provide backup evidence if live inference is slow.

Content:

- selected successful cases from the paper-360 run
- diverse people
- diverse object classes
- baseline and geometry outputs side by side
- zoom crops for each case

This section is backup material, not the main demo.

## Backend State

### 10. Run Status Panel

Purpose: make long inference feel controlled.

States:

- idle
- loading model
- running pretrained
- running geometry candidates
- scoring candidates
- complete
- failed

Display:

- current stage
- elapsed time
- GPU id
- seed
- error message when failed

## Export

### 11. Result Export

Purpose: make each demo result reusable in the report.

Export files:

- person input
- object input
- pretrained output
- geometry output
- candidate thumbnails
- scores JSON
- diagnostics markdown

Suggested output structure:

```text
outputs/live_demo/<run_id>/
  person.jpg
  object.jpg
  pretrained.jpg
  geometry.jpg
  candidates/
    candidate_0.jpg
    candidate_1.jpg
  scores.json
  diagnostics.md
```

## Suggested First Implementation

Build the first version with these minimum components:

1. Input Panel
2. Run Compare button
3. Side-By-Side Result Viewer
4. Score Comparison Strip
5. Geometry Candidate Panel
6. One-Click Demo Examples

Add the zoom evidence panel after the comparison pipeline is stable.

## Core Message

The demo should repeatedly communicate one idea:

> The geometry method does not claim a new trained model. It keeps the pretrained OmniTry model fixed and improves small-object try-on by adding affordance-aware prompting plus candidate selection at inference time.
