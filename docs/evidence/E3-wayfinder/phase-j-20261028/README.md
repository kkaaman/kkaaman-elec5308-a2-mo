# PR102 wayfinder continuation

Original contributor history and AI disclosure (Claude) are retained.
Continuation assistance: OpenAI Codex. Canonical main was merged normally.
All production modules and auxiliary config changes were reviewed. Optional
A-star routes, pinned POIs, canonical route stitching, turn refusal rules and
observer-only guidance/rendering leave the frozen baseline unchanged.
Supported loader-only rebinding and unchanged identities are recorded adjacent.
Admission and all 19 negative probes pass.

Added explicit recorded/synthetic pose-input replay, strict time ordering and
input/route hashes. It never invents a stop or promotes declared origin into
verified provenance. Existing script-generated images remain synthetic.
Ruff, boundary smoke (102 modules), strict Sphinx and diff checks pass.
Focused/full results are adjacent.

CARLA BLOCKED for actual recorded route/current-pose/turn HUD images including
stopped ETA, and documented RV-06 dependency remains pending. CARLA stayed OFF.
No recorded pose dataset was found. Issue54 and PR102 remain open.

After a later authorized capture, run:
python scripts/wayfinder_replay.py --origin baseline_start --destination baseline_goal --poses runs/wayfinder-poses.json --output runs/wayfinder-recorded-v1
Use a new output directory. Required schema: schema_version1, origin recorded,
run_id, steps(time_s,x_m,y_m,yaw_rad,speed_mps). Retain capture/source/config
provenance and real stopped samples. A declared origin alone is not acceptance.
