"""Post-hoc synthetic cross-support scoring of historical terminal Students.

Rebuilds the ORIGINAL independent dev bank from recorded frozen role seeds, and
refuses scoring unless its content hashes match the training receipt. No target
array, target evaluator, optimizer, or new diagnostic RNG is used here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))
from silent_teacher import cmr1_bootstrap as bootstrap  # noqa: E402

bootstrap.assert_bootstrap_ready()
bootstrap.assert_complete_lock(REPO / "requirements-lock.txt")

import torch  # noqa: E402
from silent_teacher import cmr1_campaign, cmr1_runtime, cmr1_student, h9_id2  # noqa: E402
from silent_teacher.full_block import FullBlockStudent  # noqa: E402
from silent_teacher.records import file_hash  # noqa: E402
from silent_teacher.train import setup  # noqa: E402

COUNTS = {"id2": 48, "xr1": 24, "cmr1": 48}
ROOTS = {"id2": REPO / "artifacts/h9_id2/replicates",
         "xr1": REPO / "artifacts/h9_xr1/replicates",
         "cmr1": REPO / "runs/cmr1/formal-v1"}
OUTPUT = REPO / "diagnostics/historical_cross_support"
ARMS = ("nat", "pos")
STAGE_B_SHA = cmr1_campaign.STAGE_B_SHA
STAGE_C_SHA = cmr1_campaign.STAGE_C_SHA
SCHEMA = "latentbridge.historical_cross_support.exploratory.v1"


def sha(path: Path) -> str:
    return file_hash(path)


def tensor_sha(value: torch.Tensor) -> str:
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def require(ok: bool, message: str) -> None:
    if not ok:
        raise RuntimeError(message)


def load_record(campaign: str, replicate: int) -> tuple[Path, dict]:
    folder = ROOTS[campaign] / f"replicate_{replicate:02d}"
    record = json.loads((folder / "replicate.json").read_text(encoding="utf-8"))
    require(record.get("replicate_id", record.get("replicate")) == replicate,
            "REPLICATE_ID_MISMATCH")
    require(record.get("status", record.get("state")) == "PAIR_COMPLETE",
            "PAIR_NOT_COMPLETE")
    require(set(record["arms"]) == set(ARMS), "ARMS_INCOMPLETE")
    return folder, record


def gpt_teacher(campaign: str):
    from scripts.run_h9_id2_replicate import build_teacher as id2_teacher
    from scripts.run_h9_xr1_replicate import build_teacher as xr1_teacher
    from scripts.run_h9_id2_replicate import WEIGHTS_SHA
    require(sha(REPO / "artifacts/gpt2/model.safetensors") == WEIGHTS_SHA,
            "GPT2_TEACHER_WEIGHT_IDENTITY_MISMATCH")
    return (id2_teacher if campaign == "id2" else xr1_teacher)("cuda")


def cmr1_roles() -> dict[str, int]:
    path = REPO / cmr1_campaign.STAGE_B_PATH
    require(sha(path) == STAGE_B_SHA, "STAGE_B_IDENTITY_MISMATCH")
    stage_b = json.loads(path.read_text(encoding="utf-8"))
    require(stage_b["state"] == "FROZEN_STAGE_B_EXECUTION_CONTROL" and
            stage_b["root_seed"] == 2742875683296071956 and
            stage_b["root_commitment"]["namespace"] == "latentbridge/cmr1/v1/formal-v1",
            "CMR1_SCIENTIFIC_IDENTITY_MISMATCH")
    roles = {row["role_key"]: row["integer_seed"] for row in stage_b["campaign_records"]}
    require(len(roles) == 432, "CMR1_ROLE_COUNT_MISMATCH")
    return roles


def rebuild_dev(campaign: str, replicate: int, record: dict, teacher,
                roles: dict[str, int] | None = None) -> tuple[dict, dict]:
    """Return the two historical dev supports, direct labels, and verified identities."""
    if campaign == "cmr1":
        require(record["stage_c_sha256"] == STAGE_C_SHA, "STAGE_C_BINDING_MISMATCH")
        seeds = cmr1_campaign.role_seeds(roles, replicate)
        require(record["seeds"] == seeds, "CMR1_ROLE_BINDING_MISMATCH")
        with torch.no_grad():
            banks, evidence = cmr1_campaign.build_banks(teacher, seeds, "dev", "cuda",
                                                        count=256)
        require(evidence == record["construction"]["dev"], "CMR1_DEV_BANK_IDENTITY_MISMATCH")
        labels = {}
        with torch.no_grad():
            for arm in ARMS:
                labels[arm] = torch.cat([
                    teacher.residual_update(chunk.to("cuda")).cpu()
                    for chunk in banks[arm].split(8)]).contiguous()
        identities = {arm: {"hidden_sha256": tensor_sha(banks[arm]),
                            "labels_sha256": tensor_sha(labels[arm])} for arm in ARMS}
        identities["construction"] = evidence
        return {arm: (banks[arm], labels[arm]) for arm in ARMS}, identities

    from scripts.run_h9_id2_replicate import FORMAL_RNG_MANIFEST, load_formal_manifest
    from scripts.run_h9_xr1_replicate import RNG_MANIFEST, load_formal_manifest as load_xr1
    manifest = load_formal_manifest(FORMAL_RNG_MANIFEST) if campaign == "id2" else load_xr1(RNG_MANIFEST)
    entry = next(row for row in manifest["replicates"] if row["replicate_id"] == replicate)
    require(record["seeds"] == {role: entry[role] for role in h9_id2.ROLES},
            "GPT2_ROLE_BINDING_MISMATCH")
    streams = h9_id2.Id2Streams(record["seeds"], label=f"diagnostic-{replicate:02d}")
    donor = streams.donor_pool("dev")
    permutation = streams.permutations("dev")
    block_index = 5 if campaign == "id2" else 9
    model, block = teacher
    with torch.no_grad():
        built = h9_id2.build_arms(model, block, donor, permutation, "cuda",
                                  block_index=block_index)
    old = record["construction"]["dev"]
    mapping = {"donor_pool_sha256": donor, "permutation_sha256": permutation,
               "H_nat_sha256": built["H_nat"], "H_pos_sha256": built["H_pos"],
               "Y_nat_sha256": built["Y_nat"], "Y_pos_sha256": built["Y_pos"]}
    for name, tensor in mapping.items():
        require(tensor_sha(tensor) == old[name], f"{campaign.upper()}_DEV_BANK_IDENTITY_MISMATCH:{name}")
    identities = {arm: {"hidden_sha256": tensor_sha(built[f"H_{arm}"]),
                        "labels_sha256": tensor_sha(built[f"Y_{arm}"])} for arm in ARMS}
    identities["construction"] = {name: old[name] for name in mapping}
    return {arm: (built[f"H_{arm}"], built[f"Y_{arm}"]) for arm in ARMS}, identities


def terminal_student(campaign: str, folder: Path, record: dict, arm: str):
    if campaign == "cmr1":
        checkpoint = folder / arm / "terminal.pt"
        terminal = json.loads((folder / arm / "terminal.json").read_text(encoding="utf-8"))
        require(terminal == record["arms"][arm] and terminal["state"] == "COMPLETE" and
                terminal["optimizer_steps"] == 10000 and not terminal["test_only"],
                "CMR1_TERMINAL_RECEIPT_MISMATCH")
        require(sha(checkpoint) == terminal["checkpoint_sha256"],
                "CMR1_CHECKPOINT_HASH_MISMATCH")
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        require(payload["step"] == 10000 and payload["stage_c_sha256"] == STAGE_C_SHA,
                "CMR1_CHECKPOINT_CONTENT_MISMATCH")
        student = cmr1_student.build_student_architecture()
        student.load_state_dict(payload["student"], strict=True)
        student.assert_frozen_geometry()
    else:
        checkpoint = folder / f"student_{arm}.pt"
        terminal = record["arms"][arm]
        require(terminal["status"] == "COMPLETE" and terminal["terminal_step"] == 10000,
                "GPT2_TERMINAL_RECEIPT_MISMATCH")
        require(sha(checkpoint) == terminal["checkpoint_sha256"],
                "GPT2_CHECKPOINT_HASH_MISMATCH")
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        require(payload["steps"] == 10000 and payload["replicate_id"] == record["replicate_id"]
                and payload["arm"] == arm and
                (campaign != "xr1" or payload["block_index"] == 9),
                "GPT2_CHECKPOINT_CONTENT_MISMATCH")
        student = FullBlockStudent()
        student.load_state_dict(payload["model_state"], strict=True)
    return student.eval().to("cuda"), {"path": checkpoint.relative_to(REPO).as_posix(),
                                       "sha256": sha(checkpoint)}


@torch.no_grad()
def nmse(student, hidden: torch.Tensor, labels: torch.Tensor, denominator: float) -> float:
    numerator = 0.0
    for h, y in zip(hidden.split(8), labels.split(8)):
        difference = student(h.to("cuda")).double() - y.to("cuda").double()
        numerator += float(difference.square().sum())
    result = numerator / denominator
    require(math.isfinite(result), "NONFINITE_NMSE")
    return result


def score_pair(campaign: str, replicate: int, teacher, roles=None) -> dict:
    folder, record = load_record(campaign, replicate)
    banks, identities = rebuild_dev(campaign, replicate, record, teacher, roles)
    denominators = {}
    for support in ARMS:
        _, labels = banks[support]
        denom = sum(float(y.double().square().sum()) for y in labels.split(8))
        require(math.isfinite(denom) and denom > 0, "LABEL_ENERGY_INVALID")
        denominators[support] = denom
    values = {}; checkpoints = {}
    for arm in ARMS:
        student, checkpoints[arm] = terminal_student(campaign, folder, record, arm)
        with torch.no_grad():
            values[arm] = {support: nmse(student, *banks[support], denominators[support])
                           for support in ARMS}
        del student
        torch.cuda.empty_cache()
    # The diagonal uses exactly the original dev bank and the historical metric.
    for arm in ARMS:
        old = (record["own_q"][arm]["raw_error"] if campaign != "cmr1" else
               next(json.loads(line)["synthetic_dev_nmse"] for line in
                    (folder / arm / "curve.jsonl").open(encoding="utf-8")
                    if '"synthetic_dev_nmse"' in line and '"step": 10000' in line))
        require(abs(values[arm][arm] - old) <= 1e-9,
                f"HISTORICAL_DIAGONAL_MISMATCH:{campaign}/{replicate}/{arm}")
    return {"schema": SCHEMA, "campaign": campaign, "replicate": replicate,
            "support_split": "historical_independent_dev_256",
            "bank_identities": identities, "label_energy_denominators": denominators,
            "checkpoints": checkpoints, "nmse": values,
            "same_column_inputs_labels_denominator": True,
            "post_hoc_exploratory": True, "target_data_read": False,
            "student_parameter_updates": 0}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", choices=COUNTS, required=True)
    parser.add_argument("--start", type=int, default=1,
                        help="first replicate for an infrastructure continuation; never overwrites results")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--smoke", action="store_true")
    mode.add_argument("--all", action="store_true")
    args = parser.parse_args()
    require(1 <= args.start <= COUNTS[args.campaign] and (not args.smoke or args.start == 1),
            "REPLICATE_RANGE_INVALID")
    setup()
    if args.campaign == "cmr1":
        cmr1_runtime.enforce_governed_numerical_mode()
        roles = cmr1_roles()
        teacher = cmr1_campaign.load_pinned_teacher_wrapper(
            REPO / "artifacts/cmr1/teacher", "cuda")
    else:
        roles = None
        teacher = gpt_teacher(args.campaign)
    count = 1 if args.smoke else COUNTS[args.campaign]
    output = OUTPUT / args.campaign
    if args.all:
        output.mkdir(parents=True, exist_ok=True)
    for replicate in range(args.start, count + 1):
        result = score_pair(args.campaign, replicate, teacher, roles)
        if args.smoke:
            print(json.dumps({"campaign": args.campaign, "replicate": replicate,
                              "bank_and_checkpoint_identity": "PASS",
                              "shared_column_and_historical_diagonal": "PASS"}), flush=True)
        else:
            path = output / f"replicate_{replicate:02d}.json"
            require(not path.exists(), "DIAGNOSTIC_RESULT_ALREADY_EXISTS")
            temporary = output / f"replicate_{replicate:02d}.json.tmp"
            require(not temporary.exists(), "INCOMPLETE_DIAGNOSTIC_RESULT_EXISTS")
            temporary.write_text(json.dumps(result, sort_keys=True, allow_nan=False) + "\n",
                                 encoding="utf-8")
            temporary.rename(path)
            print(json.dumps({"campaign": args.campaign, "replicate": replicate,
                              "state": "COMPLETE"}), flush=True)


if __name__ == "__main__":
    main()
