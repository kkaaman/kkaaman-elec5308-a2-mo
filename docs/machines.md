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

Abdullah confirms that Mohammad, Alan, Zhaolin and Abdullah use university
PCs with the same hardware specifications. The recorded hardware above
represents the shared specification, based on that confirmation.

| Member | Hardware evidence | CARLA smoke-test evidence |
|---|---|---|
| Abdullah | Measured locally | Connection, Town10HD_Opt, spawn and cleanup passed |
| Mohammad | Same specifications, confirmed by Abdullah | Not independently recorded |
| Alan | Same specifications, confirmed by Abdullah | Not independently recorded |
| Zhaolin | Same specifications, confirmed by Abdullah | Not independently recorded |

Demo arrangement: university PC setup verified by Abdullah.
A separate primary/backup ranking is omitted at Abdullah's direction
because the team uses university PCs with the same specifications.

Individual Python/client/server versions on the other PCs have not been
independently recorded. Review requested from Zhaolin.
