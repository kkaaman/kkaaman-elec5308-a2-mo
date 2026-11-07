"""BL-B-06/07: live scenario placement, evaluation-truth and event-sensor checks.

Usage (repository root, CARLA 0.9.16 running Town10HD_Opt, no other clients):
    python scripts/verify_scenarios.py --output <new folder>

See docs/scenarios.md. Writes scenario_verification.json; exit code 0 only on MEASURED_PASS.
"""

from carla_adapter.scenario_verification import main

if __name__ == "__main__":
    raise SystemExit(main())
