# OmniTry Live Inference Backend

FastAPI backend for the live demo. The frontend can upload one person image and one object image, then compare:

- `Pretrained`: original OmniTry prompt, one candidate.
- `Pretrained + Geometry`: geometry-aware prompt, multiple candidates, score-and-select.

The model is lazy-loaded on the first inference request. Starting the server does not load the FLUX checkpoint immediately.

## Run

```bash
cd /data0/long/CS338
bash demo/backend/run.sh
```

Default URL:

```text
http://localhost:8010
```

Useful environment variables:

```bash
OMNITRY_BACKEND_PORT=8010
OMNITRY_CPU_OFFLOAD=1
CUDA_VISIBLE_DEVICES=0
OMNITRY_BACKEND_CORS_ORIGINS='*'
OMNITRY_BACKEND_OUTPUT_ROOT=outputs/live_demo_backend
```

Use `OMNITRY_CPU_OFFLOAD=0` only when the selected GPU has enough free VRAM for a full resident pipeline.

## API

### Health

```http
GET /api/v1/health
```

### Object Classes

```http
GET /api/v1/classes
```

### Demo Examples

```http
GET /api/v1/examples
```

Returns URLs under `/demo-examples/...` for the built-in example assets.

### Create Compare Run

```http
POST /api/v1/runs/compare
Content-Type: multipart/form-data
```

Fields:

| name | type | required | default |
|---|---|---:|---:|
| `person_image` | file | yes | |
| `object_image` | file | yes | |
| `object_class` | string | yes | |
| `optional_prompt` | string | no | `""` |
| `steps` | int | no | `20` |
| `guidance_scale` | float | no | `30.0` |
| `seed` | int | no | `-1` |
| `geometry_candidate_count` | int | no | `2` |
| `run_pretrained` | bool | no | `true` |
| `run_geometry` | bool | no | `true` |

Example:

```bash
curl -X POST http://localhost:8010/api/v1/runs/compare \
  -F person_image=@demo_example/person_ring.jpg \
  -F object_image=@demo_example/object_ring.jpg \
  -F object_class=ring \
  -F geometry_candidate_count=2
```

Response:

```json
{
  "run_id": "abc123def456",
  "status": "queued",
  "status_url": "/api/v1/runs/abc123def456",
  "artifact_base_url": "/artifacts/abc123def456"
}
```

### Poll Run Status

```http
GET /api/v1/runs/{run_id}
```

Statuses:

- `queued`
- `running`
- `complete`
- `failed`

When complete, the response includes:

- `result.pretrained.image_url`
- `result.geometry.image_url`
- `result.geometry.candidates`
- `result.delta`
- `result_url`

### Result JSON

```http
GET /api/v1/runs/{run_id}/result
```

### Diagnostics Markdown

```http
GET /api/v1/runs/{run_id}/diagnostics/pretrained
GET /api/v1/runs/{run_id}/diagnostics/geometry
```

## Output Layout

Each run writes:

```text
outputs/live_demo_backend/<run_id>/
  person.jpg
  object.jpg
  pretrained.jpg
  geometry.jpg
  pretrained_diagnostics.md
  geometry_diagnostics.md
  candidates/
    pretrained/
      candidate_0.jpg
    geometry/
      candidate_0.jpg
      candidate_1.jpg
  result.json
  status.json
```
