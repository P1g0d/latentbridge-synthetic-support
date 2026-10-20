"""H9*-ID2 formal replicate runner — one paired replicate (Native + Q_pos).

NOT AUTHORIZED TO RUN IN THIS PHASE. The formal RNG manifest does not exist, so this script exits
immediately with a clear message. It is implemented here so that preflight can inspect it
statically (P3 structural ordering, P4 recipe identity, P6 blindness isolation).

Phase-A only: this process contains no target reader. It reads the frozen protocol, the formal RNG
manifest (seeds) and model weights; it writes the replicate record, curves and checkpoints.
"""
import argparse
import hashlib
import json
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["HF_HUB_OFFLINE"] = "1"
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path("src").resolve()))
from silent_teacher.full_block import FullBlockStudent                              # noqa: E402
from silent_teacher.h9_id1 import (seeded_student, train_paired_student)            # noqa: E402
from silent_teacher.h9_id2 import (LENGTH, POOL_SIZES, ROLES, SPLITS, Id2Streams,     # noqa: E402
                                   assert_blind_progress, assert_training_contract, build_arms,
                                   check_construction_invariants, check_label_provenance,
                                   check_split_independence, execution_authorization,
                                   multiset_digest, tensor_sha, terminal_lr, training_config)
from silent_teacher.modules import MLP, freeze                                       # noqa: E402
from silent_teacher.records import file_hash, state_hash                             # noqa: E402
from silent_teacher.train import setup                                               # noqa: E402
from transformers import GPT2Config, GPT2LMHeadModel                                 # noqa: E402
from safetensors import safe_open                                                    # noqa: E402

WEIGHTS_SHA = "248dfc3911869ec493c76e65bf2fcf7f615828b0254c12b473182f0f81d3a707"
TEACHER_MLP_HASH = "ec48c1f164aa39f19a875afc3bfbf408505dcb2dbb239f5a4f17a603be9b18dc"
FORMAL_RNG_MANIFEST = Path("manifests/h9_id2_rng_manifest.json")
MINIBATCH_BATCH = 32


def build_teacher(device):
    cfg = GPT2Config.from_json_file("artifacts/gpt2/config.json")
    model = GPT2LMHeadModel(cfg).eval()
    with safe_open("artifacts/gpt2/model.safetensors", framework="pt", device="cpu") as handle:
        state = {("lm_head.weight" if key == "lm_head.weight" else f"transformer.{key}"):
                 handle.get_tensor(key) for key in handle.keys()
                 if not key.endswith((".attn.bias", ".attn.masked_bias"))}
    missing, unexpected = model.load_state_dict(state, strict=False)
    assert missing == ["lm_head.weight"] and not unexpected
    return model.to(device), model.transformer.h[5]


def load_formal_manifest(path):
    """ID2's own manifest loader. ID1's `load_rng_manifest` is deliberately NOT reused: it asserts
    ID1's seven role names, which would silently import an ID1-specific assumption."""
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    assert manifest.get("schema", "").startswith("latentbridge.h9_id2"), manifest.get("schema")
    assert len(manifest["replicates"]) == 48, len(manifest["replicates"])
    for entry in manifest["replicates"]:
        missing = [role for role in ROLES if role not in entry]
        assert not missing, f"replicate {entry.get('replicate_id')} missing roles: {missing}"
        extra = [name for name in entry if name not in ROLES and name != "replicate_id"]
        assert not extra, f"replicate {entry['replicate_id']} has undeclared roles: {extra}"
    ids = sorted(entry["replicate_id"] for entry in manifest["replicates"])
    assert ids == list(range(1, 49)), "formal manifest must cover replicates 1..48"
    return manifest


def formal_seeds(manifest, replicate_id):
    entry = next((e for e in manifest["replicates"] if e["replicate_id"] == replicate_id), None)
    assert entry is not None, f"replicate {replicate_id} absent from the formal RNG manifest"
    return {role: entry[role] for role in ROLES}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--replicate", type=int, required=True)
    parser.add_argument("--root", type=Path, default=Path("artifacts/h9_id2/replicates"))
    args = parser.parse_args()

    authorization = execution_authorization()
    if not authorization["authorized"]:
        print("H9*-ID2 formal training is NOT AUTHORIZED: "
              + "; ".join(authorization["reasons"]) + ". Aborting.")
        return 2

    setup()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    assert device == "cuda"
    config = training_config()
    assert_training_contract(config, steps=config["steps"], horizon=config["steps"])
    manifest = load_formal_manifest(FORMAL_RNG_MANIFEST)
    streams = Id2Streams(formal_seeds(manifest, args.replicate), label=f"replicate-{args.replicate:02d}")

    folder = args.root / f"replicate_{args.replicate:02d}"
    if folder.exists():
        record_path = folder / "replicate.json"
        if record_path.exists():
            recorded = json.loads(record_path.read_text(encoding="utf-8"))
            if recorded.get("status") == "PAIR_COMPLETE":
                print(f"replicate {args.replicate} already complete and validated; skipping")
                return 0
        print(f"replicate {args.replicate} has an incomplete attempt at {folder}; it must be "
              f"superseded (moved to superseded/) before re-running")
        return 3

    folder.mkdir(parents=True, exist_ok=True)
    started = time.time()
    model, block = build_teacher(device)
    assert file_hash("artifacts/gpt2/model.safetensors") == WEIGHTS_SHA
    assert state_hash(freeze(MLP.from_hf(block.mlp))) == TEACHER_MLP_HASH

    # ---- construction: one shared donor pool per split, one permutation set, both arms
    construction, training, donor_pools = {}, {}, {}
    for split in SPLITS:
        donor_pool = streams.donor_pool(split)
        donor_pools[split] = donor_pool
        permutations = streams.permutations(split)
        arms = build_arms(model, block, donor_pool, permutations, device)
        invariants = check_construction_invariants(arms)
        assert invariants["passed"], f"{split}: construction invariants failed: {invariants['failures']}"
        construction[split] = {
            "donor_pool_sha256": tensor_sha(donor_pool),
            "permutation_sha256": tensor_sha(permutations),
            "H_nat_sha256": tensor_sha(arms["H_nat"]),
            "H_pos_sha256": tensor_sha(arms["H_pos"]),
            "Y_nat_sha256": tensor_sha(arms["Y_nat"]),
            "Y_pos_sha256": tensor_sha(arms["Y_pos"]),
            "marginal_multiset_combined": multiset_digest(arms)["combined"],
            "invariants": {"passed": invariants["passed"], "failures": invariants["failures"]},
        }
        training[split] = (arms["H_nat"], arms["Y_nat"], arms["H_pos"], arms["Y_pos"])

    split_independence = check_split_independence(donor_pools)
    provenance = check_label_provenance(block, {"H_nat": training["train"][0],
                                               "H_pos": training["train"][2],
                                               "Y_nat": training["train"][1],
                                               "Y_pos": training["train"][3]})
    assert provenance["direct_accepted"] and provenance["spliced_fixture_rejected"]

    # ---- shared pair state (identical for both arms) and the evaluation-only context donor
    shared_init_seed = streams.student_init_seed()
    reference_init = seeded_student(FullBlockStudent, shared_init_seed)
    init_sha = state_hash(reference_init)
    del reference_init
    schedule = streams.minibatch_schedule(POOL_SIZES["train"], MINIBATCH_BATCH, config["steps"])
    schedule_sha = hashlib.sha256(json.dumps(schedule).encode()).hexdigest()
    context_donor = streams.context_donor_derangement()
    context_donor_sha = hashlib.sha256(context_donor.numpy().tobytes()).hexdigest()

    record = {"replicate_id": args.replicate, "status": "PAIR_INCOMPLETE",
              "seeds": dict(streams.seeds),
              "construction": construction,
              "split_independence": split_independence,
              "label_provenance": provenance,
              "shared": {"init_sha256": init_sha, "minibatch_schedule_sha256": schedule_sha,
                         "context_donor_sha256": context_donor_sha},
              "arms": {}, "own_q": {}, "seconds": None}
    progress = {"replicate_id": args.replicate, "status": "RUNNING", "completed_pairs": 0,
                "total_pairs": 48, "elapsed_seconds": 0.0}
    assert_blind_progress(progress)

    # ---- train both arms from the SAME initialization and the SAME schedule
    for arm in ("nat", "pos"):
        student = seeded_student(FullBlockStudent, shared_init_seed)
        record["shared"]["init_sha256"] = record["shared"]["init_sha256"] or state_hash(student)
        assert state_hash(seeded_student(FullBlockStudent, shared_init_seed)) == \
            record["shared"]["init_sha256"], "paired initialization mismatch"
        hidden_train, labels_train = training["train"][0 if arm == "nat" else 2], \
            training["train"][1 if arm == "nat" else 3]
        hidden_dev, labels_dev = training["dev"][0 if arm == "nat" else 2], \
            training["dev"][1 if arm == "nat" else 3]
        try:
            summary = train_paired_student(student, hidden_train, labels_train, hidden_dev,
                                           labels_dev, config, schedule, device,
                                           steps=config["steps"], horizon=config["steps"])
        except RuntimeError as error:
            from silent_teacher.h9_id2 import classify_failure
            outcome = classify_failure(error=error)
            record["arms"][arm] = {"status": outcome["category"], "detail": outcome["detail"],
                                   "replacement_seed_allowed": False}
            continue
        checkpoint = folder / f"student_{arm}.pt"
        torch.save({"model_state": student.state_dict(),
                    "init_hash": record["shared"]["init_sha256"],
                    "final_dev_nmse": summary["final_dev_nmse"],
                    "clipping_fraction": summary["clipping_fraction"],
                    "steps": config["steps"], "replicate_id": args.replicate, "arm": arm},
                   checkpoint)
        curve_path = folder / f"curve_{arm}.jsonl"
        with curve_path.open("w", encoding="utf-8") as handle:
            for row in summary["curve"]:
                handle.write(json.dumps(row) + "\n")
        record["arms"][arm] = {"status": "COMPLETE", "final_state_sha256": state_hash(student),
                               "checkpoint_sha256": file_hash(checkpoint),
                               "terminal_step": config["steps"],
                               "terminal_dev_nmse": summary["final_dev_nmse"],
                               "clipping_fraction": summary["clipping_fraction"]}
        record["own_q"][arm] = {"raw_error": summary["final_dev_nmse"]}
        del student
        torch.cuda.empty_cache()

    complete = all(entry.get("status") == "COMPLETE" for entry in record["arms"].values()) \
        and len(record["arms"]) == 2
    record["status"] = "PAIR_COMPLETE" if complete else "PAIR_FAILED"
    record["seconds"] = round(time.time() - started, 1)
    record["terminal_lr"] = terminal_lr()
    record["donor_pool_sha256"] = {split: tensor_sha(tensor) for split, tensor in donor_pools.items()}
    (folder / "replicate.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    training.clear()
    donor_pools.clear()
    torch.cuda.empty_cache()
    print(json.dumps({"replicate": args.replicate, "status": record["status"],
                      "seconds": record["seconds"]}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
