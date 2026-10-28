# G1 admission identity and reproduction

## Root cause and production fix

The old loader trusted `FROZEN_GATED_OPERATIONAL` and checked only that response
covered age + delivery + vehicle lag. The runner bound source/SCOPE but excluded
`mission_baseline.json`. It therefore accepted unreviewed changes to cruise,
deceleration, response, margin and scope wording.

`autonomy.admission.load_admitted_baseline()` now validates a trusted versioned
record at `docs/gates/G1-admission.json` before returning configuration. Both
`frozen_governor_config()` and `baseline_operational_stack()` use this production
path. The latter uses one validated in-memory document for its governor and
remaining profile checks, avoiding a second inconsistent config read. No CARLA
import, simulator truth, threshold or perception/control policy is added.

The root is derived from the installed source module, not from an untrusted
config's parent directory. Only the record's registered
`configs/mission_baseline.json` path loads; copies at other paths are not admitted.
Candidate status, absent/non-PASS record, duplicate JSON keys, identity drift,
wrong source revision, wrong fingerprint and unsupported profile fail closed.

## Configuration identity

The complete parsed mission JSON participates: frozen/candidate values, speed and
freeze status, **all scope/limitation fields**, detector/model/tracker, depth and
track policy, timing/expiry, controller/route/vehicle, entry point and publication
metadata. There are **no field omissions** and no self-hash stored inside that
config. No runtime-generated timestamp is added. Authored date/provenance text is
intentionally pinned too, so edits require an explicit record refresh/review.

Serialization is exactly:

```python
json.dumps(document, sort_keys=True, separators=(",", ":"),
           ensure_ascii=True, allow_nan=False).encode("utf-8")
```

SHA256 of these bytes is `config_identity_sha256`. Whitespace, object order and
file line endings are not semantic; numeric/string/boolean/null values are not
rounded or dropped. `config_file_sha256` additionally records the current literal
file bytes for provenance, not a second runtime formatting restriction.
The record itself is outside the hashed config and source surfaces, avoiding a
circular dependency. It is a trusted reviewed repository artifact, not a digital
signature against someone changing the validator and approval record together.

## Exact source binding

The source inventory deliberately matches the #41 algorithm: LF-normalized bytes
of every `src/**/*.py`, every file recursively under `configs/` except files named
`mission_baseline.json`, plus the reviewed depth-calibration JSON. Extra, missing
or changed files reject loading. Source and auxiliary config file formatting is
pinned apart from CRLF normalization, unlike mission JSON object formatting.

The original runner source commit is
`897bd126cb098ccc29a9a16f9b06c1052ca13d73`, fingerprint
`e49901d1a096434e90b7b2dfb1ec5539df42cb0c906d1138a2f53a2d822e77f4`.
The admission hardening changes bound source, so that hash is historical only.
See [the record](G1-admission.json) for the new combined fingerprint and every
per-file SHA256. The source roots are explicitly:

- **Loader-only:** canonical source at admission base `d1edba6` plus exactly the
  two admission source changes. No missing runner is fabricated on main.
- **Issue-41 runner:** original PR #99 source plus exactly the same admission
  hardening. Its complete source manifest/SCOPE must match the new runner binding.

The record validates both variant hashes and config identity. The actual #41
runner also independently computes its own source/SCOPE hash. An old PR #99
checkout cannot use the new config to skip hardening; it must receive the exact
reviewed changes. This was tested using an isolated, non-merged integration tree.
No production #41 source was copied to this admission branch.

The manifest pins all demos/cloud/evaluation modules because the existing
algorithm hashes all source. This does **not** admit their entry points, a
pedestrian profile, general obstacle coverage, empty-road clearance/resume or
S1–S5 operation. The permitted profile remains the documented stationary-vehicle
G2 experiment. Existing all-track and invalidity guards remain unchanged.

## Reproduction

From the repository root with project/test dependencies installed:

```bash
PYTHONPATH=src .venv/bin/python -m scripts.check_g1_admission
PYTHONPATH=src .venv/bin/python -m pytest tests/unit/test_g1_admission.py
PYTHONPATH=src .venv/bin/python -m scripts.review_baseline_freeze --output /tmp/g1-evidence-review.json
```

Use a new output file for the historical evidence reviewer. Its saved result
remains a technical candidate assessment, separate from the admission record;
none of the raw/candidate history is rewritten to manufacture PASS. The first
command is read-only and never imports/connects CARLA. It reports exact current
config/source identities, loaded cruise and pending publication status.

In a checkout containing the exact integrated #41 source:

```python
from pathlib import Path
from carla_adapter.first_stop_runner import admission, source_fingerprint
root = Path.cwd()
print(source_fingerprint(root)[0])
print(admission(root, root / "configs/mission_baseline.json")[1]["status"])
```

This checks admission without `--execute`. The published final source revision
and exact manifest/config hashes must agree before any live run. No future Git
commit ID is invented in this uncommitted proposal; base commits plus exact
content manifests identify the current source. Record the actual publication
commit once it exists.

## Review/publication and AI disclosure

The revised process is recorded in [issue #38](https://github.com/USYD-ELEC5308/26-S2-Assignment2-Group7/issues/38#issuecomment-6395116981):
publication for team review, designated reviewer/owner review, no technical
blockers and documented scope/limitations. It supersedes mandatory individual
acknowledgements from every member. Existing at0941 scope/access acknowledgement
is linked in the gate document; no claim is made that it pre-reviewed this later
patch. Final publication/review of these bytes remains pending, and later
objections are tracked explicitly.

25 October 2026: OpenAI Codex reproduced the old Git-object fingerprint, retained
the prior external failures, implemented production identity enforcement, and
added tests for exact/altered config, source, checksum and scope. Tests use
isolated registered trees and the same production loader, never a harness-only
rejection. A non-merged PR #99 tree verifies combined source compatibility.
No CARLA experiment, model/policy threshold change, commit, push, merge or GitHub
mutation was performed. Candidate measurements and provenance remain intact.

## Validation of this proposal

[Machine-readable validation](G1-admission-validation.json) records 52 focused
G1/freeze/freshness tests, all 8 external admission requirement probes, 854 main
branch tests and 984 tests in the exact PR #99-plus-hardening snapshot (each full
suite: 1 skipped, 3 subtests passed). Ruff, AST boundary smoke, strict Sphinx,
normal whitespace checks, old/new fingerprints, config identity, saved evidence
reproduction and archive/member hashes pass. The isolated integration runner
returns `ADMISSION_CHECK_PASSED` without CARLA. No old/current PR #99 checkout is
silently granted the new binding; code, config and record must be synchronized.


## PR98 canonical-loader inventory advancement

The prepared normal merge combines canonical `ce801258ddbadf3911916c4ed32e0e0e3a8b4981`
with PR98 `c0feb8a0431b4db4ba4ca3bd00c392632191b804`; it is stopped before commit.
No future merge SHA is invented. Exact source identity is the complete per-file
inventory plus its combined fingerprint in `G1-admission.json`.

G1 operational values and scope are unchanged; only the reviewed source inventory
has advanced to include the accepted #39 deterministic scenario/evaluation implementation.
The original PR104 record is preserved byte-for-byte in
`history/G1-admission-pr104.json`. The new `canonical_loader_only` variant covers
117 exact inputs, including all ten PR98 additions and removal of the empty
scenario placeholder. Old fingerprint:
`f9d9e804b8a580d6a325afe73c0f1003aa1968cb55f9df0dde8ab2244c7c3024`.
New fingerprint:
`bb8f19fedc92a330b90ee090a928bd836612390019104d823a44482ff242203d`.

The existing `issue41_runner` variant/fingerprint `cac8bc9c4f131b16b871191f69562d87517d3f1b956849b9950543e582ff400a`
is unchanged and applies only to its exact historical reviewed inventory. It does
not match this loader-only tree. A future runner containing the scenario additions
requires its own explicit reviewed binding; no future source is automatically admitted.
Mission semantic identity/file SHA256 and every operational value remain unchanged.
The checker prints the configured runner fingerprint; the actual loader variant's
fingerprint is separately reproduced in the PR98 rebind provenance.

Preparation is not a claim of new human sign-off. The rebind and final bytes still
require designated review/publication; no unanimous team approval is claimed.
No production validator, source inclusion rule, frozen mission value, guard or
scope is changed. Negative probes continue to reject candidate/config/scope/source drift.
