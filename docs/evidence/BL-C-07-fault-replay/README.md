# BL-C-07: detection flicker and tracker-ID reset on recorded RGB-D

Date: 2026-10-18. Town10HD_Opt validation split, 3260 frames, 163 sequences of 20 frames at 0.05 s.
Detector `yolo26n.pt` at the frozen settings, 1280 px and confidence 0.25, with the production
association, box depth and track filter (2-of-3 confirmation, 0.6 s expiry).

| Replay | Folder (under Downloads) | Range gate (car / person) |
|---|---|---|
| Clean reference | `Group7_batch35_Perception_20261018_172528_836/detector` | 17.5 m / 17.5 m |
| Flicker: detections withheld on frames 0, 1 of every 5 | `Group7_batch35_ReplayFaults_20261018_184358_657/flicker` | 0 / 0 (see below) |
| ID reset: tracker reset at frame 10 of every sequence | `Group7_batch35_ReplayFaults_20261018_184358_657/id_reset` | 17.5 m / 17.5 m |

The range gate counts every frame and needs 90% recall per pose. Flicker withholds 40% of
frames, so it reads 0 by construction. That is not a regression. Fault evidence is scored by
`fault_summary.py` (in this folder) on the placed in-path target at each pose, on the frames
the detector was allowed to see. Full per-bin output: `fault_summary_1280.json`.

Reproduce:

```
python docs/evidence/BL-C-07-fault-replay/fault_summary.py clean=<clean>/replay.jsonl flicker=<flicker>/replay.jsonl id_reset=<id_reset>/replay.jsonl --json out.json
```

## Results within the measured range (gate bins 5-20 m, targets 2.5-22.5 m)

There are 73 in-path car and person poses in this range.

Clean reference: three poses are never confirmed. These are the misses the range gate's
one-miss-per-bin allowance accepts:
- a car at 10.2 m (`val-road8-round001-pose010020`), detected in 4 of 20 frames;
- a person at 15.3 m (`val-road13-round000-pose002240`), never detected;
- a person at 20.6 m (`val-road10-round000-pose003440`), never detected.

### Flicker

- Losses: the same three poses, and no others, go unconfirmed. Flicker loses no target.
- Detection and valid depth on the frames shown: 100% for every confirmed pose.
- Confirmation: at most 0.15 s after the sequence start, which is the second frame the
  detector saw (clean: also at most 0.15 s).
- Longest wait between confirmed observations: at most 0.20 s. That is exactly the
  injected pattern (confirmed on frames 3 and 4 of each 5) and never longer. The planner
  keeps a required stop latched through it (unit test `test_baseline_safety`, "a flickering
  detection cannot release it"). Unconfirmed observations still limit speed.
- ID changes: none introduced.

### ID reset

- Gate: unchanged, 17.5 m / 17.5 m.
- Re-confirmation: 71 in-path targets are detected on the reset frame. All but one are
  confirmed again within 1 frame (0.05 s). The exception is the 10.2 m car above, which is
  unconfirmed in the clean replay too. The same three poses as in the clean replay go
  unconfirmed.
- Limitation: the backend restarts its tracker IDs at 1 within the same episode namespace.
  So 82 of 89 in-path targets tracked across the reset (all ranges) kept the same numeric
  ID, and the filter treated them as the same track. Only 7 real ID changes were
  exercised, and all re-confirmed within 1 frame. In a drive the detector is reset only at
  episode start (`autonomy/stack.py`) with a new episode ID, so a live ID collision cannot
  occur. Where it does occur, an inherited confirmation can only stop earlier, and a mixed
  history only enlarges the filtered footprint (the latest surface is always kept). A full
  test needs a tracker generation in `TrackIdentity`. That is a source change and is left
  until after the frozen validation.
- Timing: `warm_p95_s` for this replay (111 ms) includes the reset and a second inference
  on frame 10 (median 282 ms on that frame). It is not a runtime latency. The clean p95 is
  22 ms; flicker is 35 ms.

Beyond 22.5 m (the 25 m and 30 m bins) the clean replay already fails the gate. The faults
change nothing there that matters to the frozen 4.8 m/s cruise, which is derived from the
17.5 m range.
