# Live release validation — 31 October 2026

BLOCKED — release candidate has a concrete runtime defect that still prevents a working presentation

Runtime source: `13ff182`. CARLA client/server 0.9.16, Town10HD_Opt, RTX 4070 Laptop GPU, Python 3.12.11. Full environment and sensor/config snapshots remain in the raw run summaries. No motion admission was granted.

The final parked target run `runs/release/live-freeze-target-final` acknowledged 72 full-brake packets, then latched `host_deadline_expired` on frame 275177. Its capture took 64.206 ms. Current local detection was accepted on 60 cycles. Successful acknowledged samples: capture-to-decision median 61.638 ms, p95 81.592 ms, max 94.393 ms; capture-to-ack median 62.889 ms, p95 83.892 ms, p99 91.362 ms, max 96.017 ms. These statistics exclude the failed cycle and do not establish a 100 ms PASS. Physical stop verification was false in the fault log; do not claim physical fault-stop acceptance.

The preceding `live-freeze-target` run completed 120/120 cycles, with 113 current local detections and four current LiDAR-ready reports. Capture-to-ack median 63.802 ms, p95 82.258 ms, max 97.132 ms. Most LiDAR cycles remained UNKNOWN/HOLD. Its negative target evaluation gap was invalid pre-snapshot metadata; never use it for accuracy scoring. The final correction measured a positive 15.208 m after warm-up; target truth is evaluation-only and never feeds control.

Earlier navigation component runs completed the frozen route (one trial each, not five): latest `live-freeze-nav-02` route completion 1.0, zero collisions/lane events, mean absolute CTE 0.02045 m, cruise RMS speed error 0.11126 m/s, receipt-to-apply p95 6.661 ms. This excludes obstacle perception and is not full-stack S0 acceptance.

Live HUD attempts all retained deadline failures. Live HUD is disabled; offline/recorded diagnostics only. Parking remains synthetic-only. Weather matrices are offline evidence. No autonomous RESUME, normal target-induced #41 stop, full-workload timing acceptance, lane-quality acceptance, remote failover or live parking PASS is asserted.

Final offline checks: 1639 passed, 1 skipped, 3 subtests; Ruff clean; AST simulator boundary PASS; generated module inventory current; strict Sphinx build PASS; production admission plus all 19 negative probes PASS. Docker/service implementation was unaffected; previous smoke evidence is retained rather than relabelled as a fresh run.

`runs.json` preserves per-run successes, failures, readiness and actual timings. `raw-evidence-sha256.json` indexes 2256 raw files retained locally under `runs/release/`; raw sensor files are not all committed. `commits.txt` lists all 16 new runtime commits, each Conventional Commit PASS. No history rewrite or merge occurred.

All new live fixes are AI-assisted, human review PENDING. The user's 31 October review applies only to the earlier #99 proposal. PR #99 remains Draft/incomplete (normal stop); #101 remains incomplete (lane/S3-S5/full workload); #116 remains Draft/incomplete (clearance/restart); #117 remains Draft/incomplete (G4, actual review/rehearsals/release evidence). No linked issue is closed. The complete issue register remains `release_remaining.md`; CARLA availability no longer explains the selected workload blocker: an observed runtime deadline failure does.

Reproduce the failed profile (choose a fresh output directory):

```bash
cd /home/alan/release-candidate-20261031
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/alan/issue40-runtime312/bin/python scripts/run_release.py observe --execute --target --host 172.28.176.1 --output runs/release/target-review-next
```

Report handoff: use the route metrics as navigation-component evidence, the actual parked detection frames as parked perception evidence, and the failure log for the deadline limitation. Do not write a release-ready conclusion or present forced brake as a target-induced stop. Further deadline diagnosis is required before a freeze can be approved.
