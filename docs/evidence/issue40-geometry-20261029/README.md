---
orphan: true
---

# #40 geometry evidence, 29 October 2026

[Full report](../../lidar_geometry_issue40.md). Verdict: CARLA COMPLETE — sensor
geometry validated; #40 offline integration remains. No positive clearance or
20-run stop/HOLD/RESUME acceptance is claimed.

All attempted sweeps are retained in five independently decompressible tar.xz
archives, including the initial close-target ego-overlap evaluation failures and
original clipped-last-bin screening results. Member manifests record each file's
size/SHA256 and archive identity. Each archive was reopened and every member
verified. Extract all archives into one directory for replay/audit; no live
connection is required. Raw-sensor buffers, transformed arrays, capture metadata,
cleanup records, scripts and per-candidate results are inside. `full-bin-audit.json`
recomputes uniform 0.5 m bins through 6.5 m. It does not overwrite original data.

The five batch counts are 64,36,8,90,90 (288 total). Source/config admission review,
19 negative probes, production-loader check and verification logs accompany them.
Previous #103 roof evidence remains under `../E2-01-04/live-20261029/`.
