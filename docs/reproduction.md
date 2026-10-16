# Reproduction and prerequisites — DOC-03

Run commands from the repository root in the active virtual environment. Paths
below work in PowerShell or a shell when adjusted to your local directories.
Do not reuse an existing run-output filename. See the
[evidence index](evidence_index.md) for missing final evaluation and reviews.

## Recorded versions and scope

The [machine record](machines.md) records Python 3.12.10 and CARLA client/server
0.9.16. The detector [sample record](evidence/baseline-batch/detector.json) records
ultralytics 8.4.177, torch 2.14.1 and numpy 2.5.4. The CPU torch build elsewhere
in the evidence does not establish CUDA support. These are historical records,
not a full validated package lock or a claim that every combination was tested.
The project permits Python >=3.10; the recorded live profile uses Python 3.12.
Keep a Python 3.12 environment for reproducing that profile. A Python 3.13 core
test run does not validate the Python 3.12 CARLA wheel.

`configs/reproduction.json` pins the three model identities and the recorded
package subset. `configs/models.yaml` remains the detector's operational
manifest. The additional lane identity was computed from the supplied baseline
binary; this does not upgrade its pending inference/review status. Hash checks
never load model code or initiate downloads.

## Connected preparation on the target platform

1. Clone the reviewed revision and prepare a separate Python 3.12 virtual
   environment (`py -3.12 -m venv .venv` on Windows, or
   `python3.12 -m venv .venv` on Linux); activate it. Record `git rev-parse HEAD`
   and `git status --short` with the evidence.
2. Install the project/test/docs dependencies and the selected CPU or GPU torch
   build. The existing connected installer remains available:
   `python setup_environment.py --skip-carla --skip-torch --skip-lfs` installs
   project dependencies without choosing a GPU build or fetching weights.
   These switches do not prevent pip from resolving dependencies online.
3. For live work install the explicit matching CARLA wheel, for example
   `python setup_environment.py --carla-wheel C:/CARLA/PythonAPI/carla/dist/carla-0.9.16-cp312-cp312-win_amd64.whl --skip-torch --skip-lfs`.
   Select a wheel matching OS, architecture and Python. An explicitly supplied
   mismatched/failing wheel now fails rather than falling back to another package.
   Alternatively install the verified wheel directly with pip. Record its hash.
4. Restore `yolo26n.pt`, `segmentation_model.pth`, `lane_unet_best.pth` from verified
   local backups, or fetch tracked LFS objects while still connected. Retain
   offline copies. Run the asset check below before loading anything. A small
   LFS pointer is not a model.
5. Validate the chosen environment, then prepare a **complete transitive pinned
   requirements lock with hashes**, including test/docs/build tools and the
   selected torch/torchvision builds and CARLA wheel if needed. Download every
   locked wheel for this exact Python/OS/architecture to a local wheelhouse.
   Include `setuptools` and `wheel` for editable installation without build
   isolation. `pip freeze` is useful inventory but is not a hashed lock. No full
   team-validated lock is committed yet; do not call this preparation complete
   until a fresh disconnected installation is recorded.

## Disconnected installation

Copy the source, wheelhouse, hashed lock, models and CARLA server installation to
the destination. Create and activate a fresh matching virtual environment, then:

```text
python setup_environment.py --offline --wheelhouse C:/offline/wheels --lock-file C:/offline/runtime-lock.txt
python setup_environment.py --offline --check-only
```

The offline branch uses isolated pip, `--no-index`, local `--find-links`,
`--only-binary=:all:`, `--no-deps` and `--require-hashes`; project installation uses `--no-deps --no-build-isolation`.
Use only locally prepared wheels and lock entries (no URL/VCS entries). It never
pulls LFS or chooses a torch/CARLA package from an online index. Missing wheels,
hashes or build dependencies cause failure; `pip check` then verifies installed
dependency consistency. A disconnected rerun is the final
proof that the prepared set is sufficient; that evidence is still pending.

## Read-only checks and useful errors

```text
python setup_environment.py --offline --check-only --assets-only
python setup_environment.py --offline --check-only
```

The first verifies size and SHA-256 of all three local models, without importing
ML libraries. Missing files, pointer stubs and corrupt binaries fail with the
asset name. The second additionally checks the three historically recorded
package versions and Python 3.12 (patch version is reported for comparison).
A torch `+cpu` or CUDA build suffix is retained in output but compared by base
version; passing this subset check does **not** validate GPU availability,
transitive dependencies, model inference or driving performance. A mismatch
means a different environment: record and validate it rather than silently
rewriting the reference manifest.

For actual model loading/sample inference, use the existing commands (installed
project and local binaries required):

```text
python scripts/check_detector_offline.py --output runs/doc03-detector.json
python scripts/verify_assets.py --models segmentation lane --backup-dir C:/offline/model-backup --output runs/doc03-assets.json
```

The asset script requires Git metadata and the tracked LFS pointer at HEAD. Keep
backups separate from the source files. These checks establish identity/inference
contracts only, not perception accuracy. Read [assets](assets.md) for their scope.

## Live CARLA and GPU prerequisites

Start the matching CARLA 0.9.16 server before the live check. Keep port/host
explicit where not localhost:2000. This read-only adapter command checks actual
client and server strings with a bounded connection timeout:

```text
python -m carla_adapter.environment_check --expected-version 0.9.16
python -m carla_adapter.environment_check --expected-version 0.9.16 --require-cuda
```

The optional CUDA flag checks torch GPU availability and records the device;
CPU/offline tests do not require it. A driver shown in `nvidia-smi` alone does not
prove that the installed torch build can use CUDA. A missing client, unreachable
server, wrong version or unavailable required GPU returns failure with a remedy.
This check does not tick, spawn actors or validate sensor/driving behaviour.
Follow [getting started](getting_started.rst) and the relevant task's smoke-run
instructions after this preflight. Preserve logs and actual machine details.

## Offline repository verification and documentation

With test/docs dependencies already installed:

```text
python scripts/check_simulator_boundary.py --smoke
python -m pytest
python -m ruff check .
python scripts/generate_module_notes.py --check
python -m sphinx -W docs docs/_build/html
```

Core CI excludes the CARLA client; marked live tests require a separate server.
The docs build needs installed ML libraries to import the public modules, but
must not require a server or download models. CI uploads `documentation-html`
only after a successful build. Keep build/test output and the actual CI run URL
with the tested revision. Final fresh-environment evidence and the full lock
remain open, alongside BL-F-07, E4-07, RV-07 and RV-08.
