"""Production source identity and fault coverage, with fixture backends."""
import pytest

from scripts.hybrid_fault_suite import HybridRig
from scripts.offline_fault_suite import BEHAVIOURS, FAULTS, run_case


@pytest.mark.parametrize("profile", ("remote", "hybrid"))
@pytest.mark.parametrize("behaviour", BEHAVIOURS)
@pytest.mark.parametrize("fault", FAULTS)
def test_production_worker_faults_stop(profile, behaviour, fault):
    result = run_case(profile, behaviour, True, fault, rig_factory=HybridRig)
    assert result["status"] == "PASS", result
