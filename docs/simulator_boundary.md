# Simulator boundary in CI (RV-05)

Core packages consume plain contracts and sensor arrays. They must not import
`carla`, `carla_adapter`, or the historical CARLA demo modules. Map topology and
ego pose can enter through the contracts described in [interfaces](interfaces.md);
this guard does not prove that arbitrary input data is free of evaluation truth.

## Three checks

1. `ruff check .` enables `TID251`, banning `carla` and `carla_adapter` imports.
2. `python scripts/check_simulator_boundary.py` parses every core source file
   with Python's AST, including local imports, aliases, relative imports,
   `TYPE_CHECKING` blocks and literal calls to `importlib.import_module` or
   `__import__`. Scanning every core file catches leaks through intermediate
   compatibility modules. Unknown targets in recognized dynamic import calls
   fail closed: use a statically named import or inject an implementation.
3. `python scripts/check_simulator_boundary.py --smoke` also imports every core
   source module in a fresh process with simulator/demo imports blocked. The
   blocker fails even when code catches `ImportError` or `Exception`. Fake CARLA
   modules installed by other tests cannot affect this process. Imports do not
   construct detectors, load weights, run inference or connect to a server.

The test job explicitly checks that CARLA is not installed, runs the boundary
command and failure fixtures, then runs the complete regression suite. Ordinary
ML dependencies remain installed; boundary checks do not require model assets
or network services. Missing required dependencies or a failed import is a
failure, never a skip. CARLA acceptance remains a separate opt-in command.

## Exceptions

`src/carla_adapter/*.py` may use CARLA. The current historical simulator entry
points are `src/demo_combined_perception.py`, `src/demo_fused_lane_following.py`
and `examples/run_carla_demo.py`. Core cannot import either source demo module.
The Ruff allowlist also names `scripts/export_lane_graph.py` and the four
existing adapter unit-test files explicitly. It does not exempt all tests,
examples, scripts or future source packages.

New runtime entry points need a reviewed allowlist change. Keep simulator
conversion logic in the adapter and business logic in core. Ruff suppression
comments do not disable the AST check. Namespace packages and new core source
files are discovered automatically.

## Failure evidence and local verification

`tests/unit/test_boundary.py` constructs small temporary failure fixtures for
direct/aliased/local/type-only imports, adapter submodules, demo dependencies,
dynamic loaders and intermediate shims. It also checks permitted adapters and
avoids flagging comments, docstrings or similarly named unrelated modules. Ruff
fixtures verify that the configured rule fails and that named exceptions pass.
The CLI fixture verifies a nonzero exit and a file/line diagnostic.

```bash
python scripts/check_simulator_boundary.py --smoke
python -m pytest tests/unit/test_boundary.py
ruff check .
python -m pytest
sphinx-build -W docs docs/_build/html
```

The guard supports ordinary imports and recognized dynamic loaders. It is not a
sandbox or a proof about arbitrary `eval`, custom import machinery, constructed
loader aliases, or runtime data flow. Those need code review. See
[Ruff TID251](https://docs.astral.sh/ruff/rules/banned-api/) for the lint rule and
[interfaces](interfaces.md) for the data boundary.

## Validation on 15 October 2026

Verified locally on Windows with Python 3.12.14, Ruff 0.17.0, pytest 9.1.1
and Sphinx 9.1.0. CARLA was not installed.

| Check | Result |
| --- | --- |
| Boundary command with `--smoke` | AST passed; 52 core modules imported with simulator imports blocked |
| Boundary failure/allowlist fixtures | 48 passed |
| Ruff | All checks passed |
| Full pytest suite | 356 passed, 1 skipped, 3 subtests passed |
| Strict Sphinx build | Passed |

The skipped test is the opt-in live Town10HD export test. No live CARLA
acceptance is claimed. GitHub Actions has been configured but was not run
remotely in this task. The [regression log](evidence/RV-05-regression.txt)
records the local suite result. A representative rejected fixture is
`from carla_adapter.vehicle import FixedVehicleAdapter` in a core file; the
checker returns a nonzero exit and identifies the source file, line and target.
