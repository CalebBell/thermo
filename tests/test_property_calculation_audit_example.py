"""Keep the documented audit example executable and its labels accurate."""
import json
import runpy
from pathlib import Path

import pytest

from thermo import VaporPressure


def test_property_calculation_audit_example():
    example = Path(__file__).resolve().parents[1]/"docs"/"Examples"/"property_calculation_audit.py"
    records = runpy.run_path(str(example))["example_records"]()
    assert len(records) == 4
    assert [r["evaluation"] for r in records] == [
        "correlation", "unavailable", "correlation", "extrapolation"]
    assert [r["within_limits"] for r in records] == [True, False, True, False]
    assert records[1]["value"] is None
    assert records[0]["value"] == records[2]["value"]
    psat = VaporPressure(CASRN="64-17-5", method="WAGNER_MCGARRY")
    assert records[0]["value"] == pytest.approx(psat.calculate(300.0, "WAGNER_MCGARRY"))
    assert records[3]["value"] > 0.0
    psat.extrapolation = "Arrhenius"
    assert records[3]["value"] == pytest.approx(psat.extrapolate(100.0, "WAGNER_MCGARRY"))
    for record in records:
        assert record["method"] == record["constructor"]["method"] == "WAGNER_MCGARRY"
        assert record["constructor"]["CASRN"] == "64-17-5"
        assert record["units"] == "Pa"
        assert all(record["versions"].values())
    assert json.loads(json.dumps(records, allow_nan=False)) == records
