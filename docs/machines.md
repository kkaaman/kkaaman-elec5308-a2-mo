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

## Remaining PROC-02 work

- Collect OS, GPU, VRAM, Python and CARLA versions for Mohammad, Alan and Zhaolin.
- Record each member's Town10HD smoke-test results.
- Confirm availability for the demo.
- Select primary and backup machines after comparing results.
- Obtain Zhaolin's review.

Demo selection: pending. Abdullah's machine is a verified candidate.
