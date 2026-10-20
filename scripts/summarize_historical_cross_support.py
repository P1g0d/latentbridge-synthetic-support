"""Read-only summary of complete post-hoc synthetic cross-support records."""
from __future__ import annotations

import hashlib
import json
import math
import statistics as stats
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / "diagnostics/historical_cross_support"
COUNTS = {"id2": 48, "xr1": 24, "cmr1": 48}
ARMS = ("nat", "pos")


def describe(values: list[float]) -> dict:
    return {"n": len(values), "mean": stats.mean(values),
            "sample_sd": stats.stdev(values), "min": min(values), "max": max(values),
            "positive_count": sum(value > 0 for value in values)}


def summarize() -> dict:
    summary = {"schema": "latentbridge.historical_cross_support.summary.v1",
               "scope": "POST_HOC_EXPLORATORY_SYNTHETIC_DEV_ONLY",
               "comparison": "within each support column, pos-trained minus nat-trained NMSE",
               "campaigns": {}}
    for campaign, count in COUNTS.items():
        rows, identities = [], {}
        for replicate in range(1, count + 1):
            path = ROOT / campaign / f"replicate_{replicate:02d}.json"
            raw = path.read_bytes()
            row = json.loads(raw)
            assert row["campaign"] == campaign and row["replicate"] == replicate
            assert row["support_split"] == "historical_independent_dev_256"
            assert row["same_column_inputs_labels_denominator"] is True
            assert row["student_parameter_updates"] == 0 and row["target_data_read"] is False
            assert set(row["nmse"]) == set(ARMS)
            assert all(set(row["nmse"][arm]) == set(ARMS) for arm in ARMS)
            assert all(math.isfinite(row["nmse"][arm][support]) and
                       row["nmse"][arm][support] >= 0
                       for arm in ARMS for support in ARMS)
            assert all(math.isfinite(row["label_energy_denominators"][support]) and
                       row["label_energy_denominators"][support] > 0 for support in ARMS)
            assert all(len(row["checkpoints"][arm]["sha256"]) == 64 for arm in ARMS)
            identities[path.relative_to(REPO).as_posix()] = hashlib.sha256(raw).hexdigest()
            rows.append(row)
        matrix = {arm: {support: describe([row["nmse"][arm][support] for row in rows])
                        for support in ARMS} for arm in ARMS}
        differences = {support: describe([row["nmse"]["pos"][support] -
                                          row["nmse"]["nat"][support] for row in rows])
                       for support in ARMS}
        summary["campaigns"][campaign] = {"paired_replicates": count,
                                           "four_cell_matrix": matrix,
                                           "paired_arm_differences": differences,
                                           "record_sha256": identities}
    return summary


if __name__ == "__main__":
    path = ROOT / "summary.json"
    assert not path.exists(), "SUMMARY_ALREADY_EXISTS"
    path.write_text(json.dumps(summarize(), sort_keys=True, indent=2,
                               allow_nan=False) + "\n", encoding="utf-8")
    print(path.relative_to(REPO).as_posix())
