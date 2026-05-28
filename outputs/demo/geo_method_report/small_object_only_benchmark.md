# Small-Object-Only Benchmark Note

This presentation should use the small-object benchmark as the main quantitative evidence. The 360-case OmniTry-Bench run is a broad robustness check only, because it includes clothing and larger accessories.

## Quantitative Small-Object Hard Set

Classes: `ring`, `bracelet`, `earrings`.

| Protocol | Items | Total | Object | Person | Artifact |
|---|---:|---:|---:|---:|---:|
| Pretrained checkpoint, K=1 | 32 | 0.623209 | 0.255470 | 0.976477 | 0.640091 |
| Pretrained checkpoint + GACS, K=2 | 32 | 0.623760 | 0.255866 | 0.977028 | 0.640823 |
| Delta | 32 | +0.000551 | +0.000396 | +0.000551 | +0.000732 |

Wins/ties/losses: `17 / 14 / 1`.

## Class Breakdown

| Class | Count | K=1 total | GACS K=2 total | Delta |
|---|---:|---:|---:|---:|
| ring | 16 | 0.602326 | 0.603051 | +0.000725 |
| bracelet | 15 | 0.651278 | 0.651681 | +0.000403 |
| earrings | 1 | 0.536287 | 0.536287 | +0.000000 |


## Diverse Small-Object Visual Check

Classes: `ring`, `earrings`, `glasses`, `necklace`, `bracelet`.

| Set | Items | Pretrained total | GACS total | Delta |
|---|---:|---:|---:|---:|
| Diverse visual examples | 5 | 0.586336 | 0.588101 | +0.001765 |

These diverse examples are mainly for presentation visuals. They broaden the story beyond the hard-set classes, but the main quantitative benchmark remains the 32-case hard small-object set above.

## Important Clarification

The 360-case run should not be described as a small-object-only benchmark. It contains `bag`, `belt`, `bottom clothes`, `dress`, `hat`, `shoe`, `top clothes`, and other categories. Use it only as a robustness/generalization appendix.
