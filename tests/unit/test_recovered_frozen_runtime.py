"""The frozen runtime rebuilt from the acceptance manifests is the one that was driven."""

import json

from autonomy.configuration import read_config
from autonomy.paths import REPO_ROOT

EVIDENCE = REPO_ROOT / "docs/evidence/TEST-08-acceptance"


def test_recovered_runtime_matches_every_acceptance_run():
    _, document, digest = read_config(EVIDENCE / "frozen_runtime.json")
    runs = json.loads((EVIDENCE / "acceptance_amended.json").read_text(encoding="utf-8"))["runs"]
    assert len(runs) == 30
    assert {r["manifest"]["config_sha256"] for r in runs} == {digest}
    assert document["status"] == "PROVISIONAL"
    assert document["evidence"]["frozen_config_sha256"] == digest
