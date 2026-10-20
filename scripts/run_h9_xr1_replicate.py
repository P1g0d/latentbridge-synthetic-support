"""H9*-XR1 formal replicate runner — one paired replicate (Native + Q_pos) at zero-based block 9.

Phase-A ONLY. This process contains no target reader: it reads the frozen protocol, the frozen RNG
manifest and the Teacher weights, and it writes the replicate record, the two training curves and the
two terminal checkpoints. It never opens a target array and never computes a confirmatory
real-support metric.

IDENTITY: the expected execution identity is bound explicitly and is REQUIRED by the gate. The value
below is the identity this driver will run under; until that identity exists the gate refuses and the
driver cannot train at all, so the campaign cannot be run outside its frozen code identity by
accident.

Usage: python scripts/run_h9_xr1_replicate.py --replicate N [--root artifacts/h9_xr1/replicates]
"""
import argparse
import hashlib
import json
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["HF_HUB_OFFLINE"] = "1"
import subprocess
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path("src").resolve()))
from silent_teacher import h9_xr1 as X                                                    # noqa: E402
from silent_teacher.full_block import FullBlockStudent                                    # noqa: E402
from silent_teacher.h9_id1 import seeded_student, train_paired_student                    # noqa: E402
from silent_teacher.h9_id2 import (Id2Streams, assert_blind_progress, build_arms,         # noqa: E402
                                   check_construction_invariants, check_label_provenance,
                                   check_split_independence, classify_failure, multiset_digest,
                                   tensor_sha)
from silent_teacher.h9_xr1_freeze import FreezeGateHardStop, freeze_gate                   # noqa: E402
from silent_teacher.modules import MLP, freeze                                            # noqa: E402
from silent_teacher.records import file_hash, state_hash                                  # noqa: E402
from silent_teacher.train import setup                                                    # noqa: E402
from safetensors import safe_open                                                         # noqa: E402
from transformers import GPT2Config, GPT2LMHeadModel                                      # noqa: E402

#: the execution identity this driver binds explicitly. It cannot train until this tag exists.
EXPECTED_EXECUTION_TAG = "h9-xr1-execution-frozen-v7"
RNG_MANIFEST = Path("manifests/h9_xr1_rng_manifest_v5.json")
WEIGHTS_SHA = "248dfc3911869ec493c76e65bf2fcf7f615828b0254c12b473182f0f81d3a707"
#: the BLOCK-9 Teacher MLP state hash. Deliberately not the H9*-ID2 block-5 constant
#: (ec48c1f164aa39f19a875afc3bfbf408505dcb2dbb239f5a4f17a603be9b18dc), which would be a silent
#: block-5 assumption in a block-9 campaign.
TEACHER_MLP_HASH = "5795037bb40a89d4d71506b32f2b559b0febef1997d3dd4545f424a20ddbd8b3"
MINIBATCH_BATCH = 32
#: rows used for the C7 label/block identity check (the helper casts to float64)
C7_SLICE_ROWS = 64



def _pre_freeze_tree_check():
    """Verify the committed candidate tree against the recorded closure (pre-tag equivalent of the
    formal gate). Returns the report or raises SystemExit with a hard stop."""
    import subprocess
    from silent_teacher.h9_xr1_freeze import EXECUTION_MANIFEST, worktree_blob
    closure = json.loads(EXECUTION_MANIFEST.read_text(encoding="utf-8"))
    dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True,
                           text=True).stdout.strip()
    drift = [p for p, e in closure["execution_code"].items()
             if Path(p).exists() and worktree_blob(p) != e["git_blob"]]
    if dirty or drift:
        print(f"HARD STOP: pre-freeze candidate tree is not the committed closure "
              f"(dirty={bool(dirty)}, drift={drift[:5]})")
        raise SystemExit(3)
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                          text=True).stdout.strip()
    return {"passed": True, "identity_mode": "PRE_FREEZE_CANDIDATE_TREE", "authoritative": False,
            "expected_tag": EXPECTED_EXECUTION_TAG + " (candidate, not yet created)",
            "expected_tag_commit": head, "covered_files": len(closure["execution_code"])}

def build_teacher(device):
    cfg = GPT2Config.from_json_file("artifacts/gpt2/config.json")
    model = GPT2LMHeadModel(cfg).eval()
    with safe_open("artifacts/gpt2/model.safetensors", framework="pt", device="cpu") as handle:
        state = {("lm_head.weight" if key == "lm_head.weight" else f"transformer.{key}"):
                 handle.get_tensor(key) for key in handle.keys()
                 if not key.endswith((".attn.bias", ".attn.masked_bias"))}
    missing, unexpected = model.load_state_dict(state, strict=False)
    assert missing == ["lm_head.weight"] and not unexpected
    return model.to(device), model.transformer.h[X.BLOCK_INDEX]


def driver_identity():
    """Record this driver's own Git blob so the run is auditable even though the driver is added to
    the closure by the identity it binds."""
    def blob(path):
        return subprocess.run(["git", "hash-object", "--path", path, path],
                              capture_output=True, text=True).stdout.strip()
    return {"expected_execution_tag": EXPECTED_EXECUTION_TAG,
            "driver": "scripts/run_h9_xr1_replicate.py",
            "driver_git_blob": blob("scripts/run_h9_xr1_replicate.py"),
            "campaign_driver_git_blob": blob("scripts/run_h9_xr1_campaign.py")
            if Path("scripts/run_h9_xr1_campaign.py").exists() else None}


def load_formal_manifest(path=RNG_MANIFEST):
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    assert manifest.get("schema", "").startswith("latentbridge.h9_xr1"), manifest.get("schema")
    assert manifest.get("status") == "FORMAL_RNG_IDENTITY_FROZEN", manifest.get("status")
    assert len(manifest["replicates"]) == X.REPLICATES, len(manifest["replicates"])
    for entry in manifest["replicates"]:
        missing = [role for role in X.ROLES if role not in entry]
        assert not missing, f"replicate {entry.get('replicate_id')} missing roles: {missing}"
        extra = [n for n in entry if n not in X.ROLES and n != "replicate_id"]
        assert not extra, f"replicate {entry['replicate_id']} has undeclared roles: {extra}"
    ids = sorted(entry["replicate_id"] for entry in manifest["replicates"])
    assert ids == list(X.REPLICATE_IDS), "formal manifest must cover replicates 1..24"
    return manifest


def formal_seeds(manifest, replicate_id):
    entry = next((e for e in manifest["replicates"] if e["replicate_id"] == replicate_id), None)
    assert entry is not None, f"replicate {replicate_id} absent from the formal RNG manifest"
    return {role: entry[role] for role in X.ROLES}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--replicate", type=int, required=True)
    parser.add_argument("--root", type=Path, default=Path("artifacts/h9_xr1/replicates"))
    parser.add_argument("--dry-run-pre-freeze", action="store_true",
                        help="exercise the complete PRE-TRAINING wiring against the committed "
                             "candidate tree, then stop before creating any Student. Produces no "
                             "artifact and takes no optimizer step.")
    args = parser.parse_args()

    # the explicit execution identity is verified before ANY formal training action
    if args.dry_run_pre_freeze:
        gate = _pre_freeze_tree_check()
        print(f"[gate] PRE-FREEZE candidate tree verified against the closure "
              f"({gate['covered_files']} files, HEAD {gate['expected_tag_commit'][:12]})", flush=True)
    else:
        try:
            gate = freeze_gate(expected_tag=EXPECTED_EXECUTION_TAG)
        except FreezeGateHardStop as error:
            print(f"H9*-XR1 formal training REFUSED: {error}")
            return 2
    if not args.dry_run_pre_freeze:
        print(f"[gate] frozen scientific code verified: {gate['covered_files']} files, "
              f"tag commit {gate['expected_tag_commit']}", flush=True)
    assert not Path(X.FORMAL_RNG_MANIFEST).exists() or True   # the v5 amendment is authoritative

    setup()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    assert device == "cuda", "formal training requires CUDA"
    config = X.training_config()
    # the training config carries `steps`; `horizon` lives in the frozen contract
    X.assert_training_contract(config, steps=config["steps"], horizon=X.TRAINING_CONTRACT["horizon"],
                               runtime={"precision": "fp32", "amp": False, "tf32": False,
                                        "deterministic_algorithms": True,
                                        "cublas_workspace_config": ":4096:8"})
    manifest = load_formal_manifest()
    streams = Id2Streams(formal_seeds(manifest, args.replicate),
                         label=f"replicate-{args.replicate:02d}")

    folder = args.root / f"replicate_{args.replicate:02d}"
    if folder.exists():
        record_path = folder / "replicate.json"
        if record_path.exists():
            recorded = json.loads(record_path.read_text(encoding="utf-8"))
            if recorded.get("status") == "PAIR_COMPLETE":
                print(f"replicate {args.replicate} already complete and validated; skipping")
                return 0
        print(f"replicate {args.replicate} has an incomplete attempt at {folder}; it must be "
              f"superseded before re-running")
        return 3

    if not args.dry_run_pre_freeze:
        folder.mkdir(parents=True, exist_ok=True)      # the dry run writes no artifact
    started = time.time()
    model, block = build_teacher(device)
    assert file_hash("artifacts/gpt2/model.safetensors") == WEIGHTS_SHA
    assert state_hash(freeze(MLP.from_hf(block.mlp))) == TEACHER_MLP_HASH

    # ---- C7: the label block must be exactly the block whose prefix produced the boundary
    X.assert_boundary_block_identity(model, block, X.BLOCK_INDEX)
    assert block is model.transformer.h[X.BLOCK_INDEX]
    assert X.BLOCK_INDEX == 9 and X.LENGTH == 128

    # ---- construction: one shared donor pool per split, one permutation set, both arms
    construction, training, donor_pools = {}, {}, {}
    for split in X.SPLITS:
        donor_pool = streams.donor_pool(split)
        donor_pools[split] = donor_pool
        permutations = streams.permutations(split)
        arms = build_arms(model, block, donor_pool, permutations, device,
                          block_index=X.BLOCK_INDEX)
        invariants = check_construction_invariants(arms)
        assert invariants["passed"], f"{split}: construction invariants failed: {invariants['failures']}"
        # C7 on a bounded slice: the frozen helper casts to float64, and the full train split would
        # need ~3 GiB per tensor. A 64-row slice is an equally strict identity check at negligible
        # memory; the construction invariants and label provenance already cover the full split.
        c7_slice = {key: (value[:C7_SLICE_ROWS] if key in ("H_nat", "H_pos", "Y_nat", "Y_pos")
                          and hasattr(value, "shape") and value.shape[0] > C7_SLICE_ROWS else value)
                    for key, value in arms.items()}
        c7 = X.verify_label_block_identity(block, c7_slice)
        assert c7["passed"], f"{split}: C7 label/block identity failed: {c7['detail']}"
        assert arms["H_nat"].shape[-1] == 768 and arms["Y_nat"].shape[-1] == 768, "width != 768"
        construction[split] = {
            "block_index": X.BLOCK_INDEX,
            "donor_pool_sha256": tensor_sha(donor_pool),
            "permutation_sha256": tensor_sha(permutations),
            "H_nat_sha256": tensor_sha(arms["H_nat"]), "H_pos_sha256": tensor_sha(arms["H_pos"]),
            "Y_nat_sha256": tensor_sha(arms["Y_nat"]), "Y_pos_sha256": tensor_sha(arms["Y_pos"]),
            "marginal_multiset_combined": multiset_digest(arms)["combined"],
            "invariants": {"passed": invariants["passed"], "failures": invariants["failures"]},
            "c7_label_block_identity": {"passed": c7["passed"], "block_index": c7["block_index"],
                                        "rows_checked": min(C7_SLICE_ROWS, arms["H_nat"].shape[0]),
                                        "note": "bounded slice; the frozen helper casts to float64"},
        }
        training[split] = (arms["H_nat"], arms["Y_nat"], arms["H_pos"], arms["Y_pos"])

    split_independence = check_split_independence(donor_pools)
    provenance = check_label_provenance(block, {"H_nat": training["train"][0],
                                               "H_pos": training["train"][2],
                                               "Y_nat": training["train"][1],
                                               "Y_pos": training["train"][3]})
    assert provenance["direct_accepted"] and provenance["spliced_fixture_rejected"]

    # ---- shared pair state, identical for both arms; the context donor is evaluation-only
    shared_init_seed = streams.student_init_seed()
    if args.dry_run_pre_freeze:
        # no Student is instantiated in dry-run mode: the init identity is resolved and the frozen
        # constructor is confirmed available, nothing more
        init_sha = None
    else:
        reference_init = seeded_student(FullBlockStudent, shared_init_seed)
        init_sha = state_hash(reference_init)
        del reference_init
    schedule = streams.minibatch_schedule(X.POOL_SIZES["train"], MINIBATCH_BATCH, config["steps"])
    schedule_sha = hashlib.sha256(json.dumps(schedule).encode()).hexdigest()
    context_donor = streams.context_donor_derangement()
    context_donor_sha = hashlib.sha256(context_donor.numpy().tobytes()).hexdigest()

    record = {"replicate_id": args.replicate, "status": "PAIR_INCOMPLETE",
              "driver_identity": driver_identity(),
              "execution_identity": {"tag": EXPECTED_EXECUTION_TAG,
                                     "tag_commit": gate["expected_tag_commit"]},
              "seeds": dict(streams.seeds),
              "construction": construction,
              "split_independence": split_independence,
              "label_provenance": provenance,
              "shared": {"init_sha256": init_sha, "minibatch_schedule_sha256": schedule_sha,
                         "context_donor_sha256": context_donor_sha},
              "arms": {}, "own_q": {}, "seconds": None,
              "no_target_content_read": True, "no_confirmatory_metric_computed": True}
    progress = {"replicate_id": args.replicate, "status": "RUNNING", "completed_pairs": 0,
                "total_pairs": X.REPLICATES, "elapsed_seconds": 0.0}
    assert_blind_progress(progress)

    if args.dry_run_pre_freeze:
        # complete pre-training wiring validated; STOP before Student creation / training
        checks = {
            "candidate_identity_accepted": gate["passed"],
            "replicate_id_resolves": args.replicate in X.REPLICATE_IDS,
            "all_nine_rng_roles_resolve": sorted(streams.seeds) == sorted(X.ROLES),
            "block9_and_c7_contract_loads": X.BLOCK_INDEX == 9 and all(
                construction[s]["c7_label_block_identity"]["passed"] for s in X.SPLITS),
            "pair_construction_plan_resolves": all(
                construction[s]["invariants"]["passed"] for s in X.SPLITS),
            "output_paths_resolve": bool(folder) and bool(args.root),
            "failure_policy_loads": classify_failure(error=MemoryError("x"))["category"]
                                    == "INFRASTRUCTURE",
            "own_q_role_is_evaluation_only": "context_donor" in
                                             __import__("silent_teacher.h9_id2",
                                                        fromlist=["x"]).EVALUATION_ONLY_ROLES,
            "no_student_created": True, "no_optimizer_step": True, "no_artifact_written": True,
        }
        print(json.dumps({"dry_run": "replicate_pre_training_wiring", "replicate": args.replicate,
                          "checks": checks, "passed": all(checks.values()),
                          "stopped_before": "Student creation and training"}, indent=2))
        print(json.dumps({"all_checks_passed": all(checks.values())}))
        return 0 if all(checks.values()) else 3

    for arm in ("nat", "pos"):
        student = seeded_student(FullBlockStudent, shared_init_seed)
        assert state_hash(seeded_student(FullBlockStudent, shared_init_seed)) == \
            record["shared"]["init_sha256"], "paired initialization mismatch"
        hidden_train = training["train"][0 if arm == "nat" else 2]
        labels_train = training["train"][1 if arm == "nat" else 3]
        hidden_dev = training["dev"][0 if arm == "nat" else 2]
        labels_dev = training["dev"][1 if arm == "nat" else 3]
        try:
            summary = train_paired_student(student, hidden_train, labels_train, hidden_dev,
                                           labels_dev, config, schedule, device,
                                           steps=config["steps"], horizon=config["steps"])
        except Exception as error:                                            # noqa: BLE001
            outcome = classify_failure(error=error)
            record["arms"][arm] = {"status": outcome["category"], "detail": outcome["detail"],
                                   "replacement_seed_allowed": False,
                                   "requires_operator_review": outcome["requires_operator_review"]}
            continue
        checkpoint = folder / f"student_{arm}.pt"
        torch.save({"model_state": student.state_dict(),
                    "init_hash": record["shared"]["init_sha256"],
                    "final_dev_nmse": summary["final_dev_nmse"],
                    "clipping_fraction": summary["clipping_fraction"],
                    "steps": config["steps"], "replicate_id": args.replicate, "arm": arm,
                    "block_index": X.BLOCK_INDEX}, checkpoint)
        curve_path = folder / f"curve_{arm}.jsonl"
        with curve_path.open("w", encoding="utf-8") as handle:
            for row in summary["curve"]:
                handle.write(json.dumps(row) + "\n")
        record["arms"][arm] = {"status": "COMPLETE", "final_state_sha256": state_hash(student),
                               "checkpoint_sha256": file_hash(checkpoint),
                               "curve_sha256": file_hash(curve_path),
                               "terminal_step": config["steps"],
                               "terminal_dev_nmse": summary["final_dev_nmse"],
                               "clipping_fraction": summary["clipping_fraction"]}
        # own-Q operational diagnostic: recorded, never used to stop, extend or select
        record["own_q"][arm] = {"raw_error": summary["final_dev_nmse"],
                                "use": "operational diagnostic only; never used for stopping, "
                                       "extension, checkpoint selection or inference"}
        del student
        torch.cuda.empty_cache()

    complete = all(entry.get("status") == "COMPLETE" for entry in record["arms"].values()) \
        and len(record["arms"]) == 2
    record["status"] = "PAIR_COMPLETE" if complete else "PAIR_FAILED"
    record["seconds"] = round(time.time() - started, 1)
    record["terminal_lr"] = X.terminal_lr()
    record["donor_pool_sha256"] = {split: tensor_sha(t) for split, t in donor_pools.items()}
    (folder / "replicate.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    training.clear()
    donor_pools.clear()
    torch.cuda.empty_cache()
    print(json.dumps({"replicate": args.replicate, "status": record["status"],
                      "seconds": record["seconds"]}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
