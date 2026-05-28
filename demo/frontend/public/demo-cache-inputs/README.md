# GPU-Free Cached Demo Inputs

Use these files when recording on a machine without GPU.

Run:

```bash
bash scripts/run_cached_video_demo.sh
```

Then upload the matching `*_person.jpg` and `*_object.jpg` files.
After about 10 seconds, the UI displays the cached left/right outputs.

Recommended cases:

| Case | Category | Person | Item |
|---|---|---|---|
| `01_ring_strong` | `ring` | `01_ring_strong_person.jpg` | `01_ring_strong_object.jpg` |
| `02_bracelet_strong` | `bracelet` | `02_bracelet_strong_person.jpg` | `02_bracelet_strong_object.jpg` |
| `04_glasses` | `glasses` | `04_glasses_person.jpg` | `04_glasses_object.jpg` |
| `05_earrings` | `earrings` | `05_earrings_person.jpg` | `05_earrings_object.jpg` |

`contact_sheet.jpg` shows person, item, pretrained output, and geometry output for all cached cases.
