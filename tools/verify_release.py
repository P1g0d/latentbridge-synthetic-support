"""Verify archived bytes and common-support arithmetic without model imports."""
from pathlib import Path
import ast
import hashlib
import json
import math
import statistics

ROOT=Path(__file__).resolve().parents[1]


def load(p):
    return json.loads((ROOT / p).read_text(encoding="utf-8"))


def main():
    inventory=load("provenance/file_inventory.json")["files"]
    for name,entry in inventory.items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest()==entry["sha256"],name
    expected={"id2":"1ce0085a2eb82b16e15ed1cbbee21fd6ea2944ac6911c0ba120b1c3287decd7f",
              "xr1":"529b1768933d910974319e2fa7719362e9ee933146d2af8f031c695156d612c5",
              "cmr1":"0fcd19b492d95df1a5010019df3b80319e520940f7c12c6ba305b6ec541552cc"}
    summary=load("results/cross_support/summary.json")
    for campaign,count in {"id2":48,"xr1":24,"cmr1":48}.items():
        p=ROOT / f"results/formal/{campaign}/results.json"
        assert hashlib.sha256(p.read_bytes()).hexdigest()==expected[campaign]
        rows=[load(f"results/cross_support/{campaign}/replicate_{r:02d}.json") for r in range(1,count+1)]
        assert summary["campaigns"][campaign]["paired_replicates"]==count
        for arm in ["nat","pos"]:
            for support in ["nat","pos"]:
                values=[row["nmse"][arm][support] for row in rows]
                archived=summary["campaigns"][campaign]["four_cell_matrix"][arm][support]
                assert math.isclose(statistics.mean(values),archived["mean"],rel_tol=0,abs_tol=1e-14)
                assert math.isclose(statistics.stdev(values),archived["sample_sd"],rel_tol=0,abs_tol=1e-14)
        for support in ["nat","pos"]:
            values=[row["nmse"]["pos"][support]-row["nmse"]["nat"][support] for row in rows]
            assert math.isclose(statistics.mean(values),summary["campaigns"][campaign]["paired_arm_differences"][support]["mean"],abs_tol=1e-14)
        for r,row in enumerate(rows,1):
            assert row["replicate"]==r and row["campaign"]==campaign
            assert row["same_column_inputs_labels_denominator"] is True
            assert row["student_parameter_updates"]==0 and row["target_data_read"] is False
    for p in list((ROOT / "src").rglob("*.py"))+list((ROOT / "scripts").glob("*.py"))+list((ROOT / "tools").glob("*.py")):
        ast.parse(p.read_text(encoding="utf-8"),filename=p.name)
    print(f"Verified {len(inventory)} file hashes, 3 original formal identities, 120 paired records, cross-support means/SDs, and Python syntax.")


if __name__=="__main__":
    main()
