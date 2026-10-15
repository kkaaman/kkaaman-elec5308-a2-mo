# CARLA machine inventory — PROC-02

<!-- PROC-02: Abdullah -->
## Abdullah — verified 14 October 2026

| Item | Recorded result |
|---|---|
| OS | Windows 11 Enterprise, 10.0.26200, 64-bit |
| GPU | NVIDIA RTX 4000 SFF Ada Generation |
| VRAM | 20,475 MiB |
| NVIDIA driver | 596.71 |
| Project Python | 3.12.10 (.venv) |
| CARLA client | 0.9.16 |
| CARLA server | 0.9.16 |
| Map tested | Carla/Maps/Town10HD_Opt |
| Connection | PASS |
| Vehicle spawn | PASS — vehicle.tesla.model3 |
| Vehicle cleanup | PASS |
| PyTorch | 2.14.1+cpu; CUDA unavailable in current environment |

Evidence: terminal outputs reviewed during setup. Connection and vehicle
spawn/cleanup passed. Full driving performance has not been benchmarked.

## Team machines and demo arrangement

Abdullah reports that Mohammad, Alan, Zhaolin and Abdullah use university PCs
with the same nominal hardware specification. Only Abdullah's machine has an
independently recorded PROC-02 smoke test so far. The recorded hardware above
represents the shared specification, based on that confirmation.

| Member | Hardware information | CARLA smoke-test evidence |
|---|---|---|
| Abdullah | Independently recorded | PASS — connection, Town10HD_Opt, spawn and cleanup |
| Mohammad | Reported same university-PC specification | Pending independent verification |
| Alan | Reported same university-PC specification | Pending independent verification |
| Zhaolin | Reported same university-PC specification | Pending independent verification |

## Demo arrangement

Abdullah's verified university PC is currently a demonstrated CARLA-capable
candidate. The team still needs to nominate a primary and backup demo machine
after at least one additional machine completes the same smoke test.

Individual Python, CARLA client/server and GPU-runtime configurations for
Mohammad, Alan and Zhaolin remain to be independently recorded.

Evidence: `docs/evidence/PROC-02-abdullah-smoke.json`.