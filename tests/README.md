# tests/

`python -m pytest` runs everything. The CARLA tests are reported as skipped on
machines without CARLA.

| Folder | What goes here | Needs CARLA |
|---|---|---|
| `unit/` | Small, fast tests of one module (geometry, filters, metrics, boundary) | No |
| `integration/` | Stack or harness runs across several packages (stop/hold/resume, faults, cloud fallback) | No |
| `fixtures/` | Small checked-in test data (recorded frames, fake detections). Keep files small and never commit raw runs | No |
| `carla/` | Tests against a live CARLA 0.9.16 server. Run them with `python -m pytest tests/carla` on a CARLA machine | Yes |
| `navigation_control/` and the top-level `test_*.py` | A1 tests, kept as the regression baseline | No |

**Rules:**
- Tests import packages by name. `tests/conftest.py` puts `src/` on the path.
- Each module in `tests/carla/` must start with
  `carla = pytest.importorskip("carla")`. Without CARLA, it then shows as
  skipped rather than failing or vanishing. A skipped test is not a simulator
  pass.
- Test files in different folders can share a base name, because pytest runs
  with `--import-mode=importlib`.
