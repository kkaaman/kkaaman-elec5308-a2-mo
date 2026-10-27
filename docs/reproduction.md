# Reproduction - DOC-03

The [reference manifest](../configs/reproduction.json) records asset identities
and selected runtime versions from retained RV-01, RV-02 and baseline detector
records. It is not a full dependency lock or proof of a fresh installation.
Do not silently replace its hashes with hashes of unverified local files.

## Prepare while connected

Use a dedicated environment matching the target Python version, OS and CPU
architecture. Install and validate the project following
[getting started](getting_started.rst) and [assets](assets.md).
Recover model binaries through the authorized LFS clone and retain separate
verified offline copies at the paths listed in the manifest.

In that validated environment, retain `python -m pip freeze --all` as
`requirements-validated.txt`, the exact project source commit, the matching
CARLA wheel/server package and model backups. Download every pinned dependency
into a separate wheelhouse using:

```text
python -m pip download --only-binary=:all: --dest wheelhouse -r requirements-validated.txt
python -m pip download --only-binary=:all: --dest wheelhouse setuptools wheel
```

Record hashes for the wheelhouse files. If a package is unavailable as a wheel,
prepare and validate its wheel on the target platform before disconnecting.
Inspect freeze output for editable/local paths and retain those exact source
artifacts separately. GPU wheels may require the appropriate PyTorch index
while connected. Never assume CPU and CUDA builds are interchangeable.

## Install disconnected

In a fresh virtual environment with the prepared artifacts available:

```text
python -m pip install --no-index --find-links wheelhouse -r requirements-validated.txt
python -m pip install --no-index --find-links wheelhouse --no-build-isolation --no-deps -e .
python -m pip check
```

Retain full command output and exit codes. Missing packages, build dependencies
or incompatible wheels must be resolved in the preparation environment.
Fresh-machine disconnected installation: **NOT RUN** for this documentation change.

## Read-only reference checks

```text
python setup_environment.py --offline --check-only --assets-only
python setup_environment.py --offline --check-only
```

These commands never install packages, invoke Git LFS or load model weights.
Missing files, wrong sizes (including LFS pointer stubs), SHA-256 mismatches and
selected package/version mismatches produce a nonzero exit status with a remedy.
Model paths are relative to the repository root. The second command checks the
reference Python major/minor and selected packages; it does not check every
transitive dependency. Full installed build suffixes are printed, but comparison
uses the base version. Passing does not establish CUDA support, inference
accuracy, driving acceptance or a successful disconnected installation.

## Optional live CARLA preflight

Start the matching server and use the target project's installed client:

```text
python scripts/check_reproduction.py --live --host localhost --port 2000 --timeout 5
python scripts/check_reproduction.py --live --require-cuda
```

Client and server version strings must both equal the reference version.
Connection queries have a bounded timeout. The CUDA option checks torch device
availability, not available VRAM or demonstrated CARLA performance. The check
never ticks the world or creates actors. Live server and GPU verification:
**NOT RUN** here. An unavailable server, wrong wheel, version mismatch or required
CUDA failure produces a nonzero exit code.

## Repository checks and retained evidence

With test/docs dependencies installed, run from the repository root:

```text
python scripts/check_simulator_boundary.py --smoke
python -m pytest -q --tb=short
python -m ruff check . --extend-exclude build
python scripts/generate_module_notes.py --check
python -m sphinx -W --keep-going docs docs/_build/html
```

Retain logs, exit codes, environment details and the tested commit. Preserve
historical evidence bytes: Windows CRLF conversion can invalidate raw-byte
hashes. Diagnose against committed LF content without editing recorded hashes.
Admission inventory failures require review of source changes, not exclusion of
new files or fabricated admission records.

Link actual results in the [evidence index](evidence_index.md). A configured CI
job is not a successful build record. Keep #62 open while dependency reviews,
full environment lock and disconnected reproduction evidence remain pending.
