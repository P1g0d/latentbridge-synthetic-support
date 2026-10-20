"""H9*-XR1 campaign identity: block index, campaign size, the frozen 16-field training contract,
the C7 same-block guard, evaluation eligibility and execution authorization.

Scope: this module is the XR1 counterpart of the campaign constants that `h9_id2.py` hard-codes for
H9*-ID2. It contains NO scientific code of its own — every scientific definition is imported from
the frozen modules:

  * treatment construction, the nine-role RNG schema, the multiset/label invariants and the whole
    failure-semantics block come from `h9_id2` (imported, never copied);
  * the endpoints, the Student, the context evaluation and the audited prefix helpers come from
    `full_block` / `h9_id1` / `attention`;
  * the statistics and the classifier live in `h9_xr1_analysis` (df = 23, no ID2 margins).

Provenance ownership (implementation design v0.2 §9.1, D10): this module deliberately does NOT
carry any code-hash snapshot. RNG identity belongs to the formal RNG manifest, target identity to
the target ledger, and scientific code identity to the execution hash manifest plus its immutable
tag. `execution_identity_reference()` records only paths and a status.

It generates no seed values, materializes no target and trains nothing.
"""
import hashlib
from pathlib import Path

from .full_block_trainer import learning_rate
from .h9_id2 import (REPLICATES as ID2_REPLICATES,  # noqa: F401  (re-exported for comparison only)
                     ROLES as ID2_ROLES, STUDENTS_PER_REPLICATE as ID2_STUDENTS_PER_REPLICATE,
                     TRAINING_CONTRACT as ID2_TRAINING_CONTRACT, evaluation_eligibility)

# ------------------------------------------------------------------------------- campaign identity
#: zero-based block index of the XR1 operator (frozen scientific protocol §1)
BLOCK_INDEX = 9
#: frozen campaign size: m = 24 paired replicates / 48 Students (protocol §4)
REPLICATES = 24
STUDENTS_PER_REPLICATE = 2
PLANNED_STUDENTS = REPLICATES * STUDENTS_PER_REPLICATE
REPLICATE_IDS = tuple(range(1, REPLICATES + 1))
SPLITS = ("train", "dev", "audit")
POOL_SIZES = {"train": 4096, "dev": 256, "audit": 512}
LENGTH = 128
CUT_QUERY = 63
LAST_QUERY = LENGTH - 1

#: the nine frozen RNG roles, inherited unchanged. No tenth role exists.
ROLES = ID2_ROLES
assert len(ROLES) == 9, "the XR1 role schema is exactly the nine frozen roles"
assert STUDENTS_PER_REPLICATE == ID2_STUDENTS_PER_REPLICATE, "pair shape must match the frozen design"

#: XR1's own frozen identities — it may reuse H9*-ID2's DEFINITIONS but must not claim its CLOSURE
EXECUTION_TAG = "h9-xr1-execution-frozen-v1"
FORMAL_RNG_MANIFEST = Path("manifests/h9_xr1_rng_manifest.json")
FROZEN_TARGET_LEDGER = Path("manifests/h9_xr1_target_ledger.json")
PREDECESSOR_INVENTORY = Path("manifests/h9_xr1_predecessor_exposure_inventory.json")
PREDECESSOR_INVENTORY_SHA256 = (
    "1d5cb8d491b02d252516bce0095862440a5a5e4f4f228f0e84518733887a9772")

#: campaign-level target-selection seeds are NOT role seeds; the six environment seeds are fresh for
#: XR1 and the near-duplicate seed is reused by reference (never redrawn).
TARGET_SELECTION_ROLES = ("c4_unit_seed", "c4_window_seed", "pes2o_unit_seed", "pes2o_window_seed",
                          "se_unit_seed", "se_window_seed", "dedup_minhash_seed")
REUSED_DEDUP_SEED = 1862023614778194523          # from H9*-ID1, reused unchanged
EXPECTED_ROLE_SEED_COUNT = REPLICATES * len(ROLES)

# ------------------------------------------------------------------- the frozen training contract
#: The executed H9*-ID2 recipe. 16 fields: the 11 that `h9_id2.assert_training_contract` asserts
#: numerically, plus the 5 determinism/environment fields it does NOT assert, plus eval/checkpoint
#: behaviour. The XR1 contract asserts all of them (implementation design v0.2 §5.2).
TRAINING_CONTRACT = {
    # --- numerically asserted by the historical contract
    "steps": 10000,
    "microbatch": 8,
    "accumulation": 4,
    "optimizer": "AdamW",
    "betas": (0.9, 0.999),
    "eps": 1e-8,
    "weight_decay": 0.0,
    "lr_peak": 1e-3,
    "warmup": 200,
    "clip_norm": 1.0,
    "lr_final_ratio": 0.1,
    # --- enforced historically only by train.setup() and the driver env var
    "precision": "fp32",
    "amp": False,
    "tf32": False,
    "deterministic_algorithms": True,
    "cublas_workspace_config": ":4096:8",
    # --- endpoint behaviour
    "horizon": 10000,
    "effective_batch": 32,
    "lr_final": 1e-4,
    "eval_every": 500,
    "terminal_checkpoint_step": 10000,
    "checkpoint_selection": "none",
    "early_stopping": "none",
    "extension": "none",
    "augmentation": "none",
}
#: `h9_id1.train_paired_student` is the ACTUALLY EXECUTED trainer. `full_block_trainer.train_full_block`
#: must not be substituted: it writes `last.pt` at every eval and carries a `fixed_lr_extension` hook.
EXECUTED_TRAINER = "silent_teacher.h9_id1.train_paired_student"
FORBIDDEN_TRAINER = "silent_teacher.full_block_trainer.train_full_block"


def training_config():
    """The `config` dict consumed by the inherited paired trainer."""
    return {"microbatch": TRAINING_CONTRACT["microbatch"],
            "accumulation": TRAINING_CONTRACT["accumulation"],
            "warmup": TRAINING_CONTRACT["warmup"],
            "lr_peak": TRAINING_CONTRACT["lr_peak"],
            "lr_final_ratio": TRAINING_CONTRACT["lr_final_ratio"],
            "clip_norm": TRAINING_CONTRACT["clip_norm"],
            "eval_every": TRAINING_CONTRACT["eval_every"],
            "weight_decay": TRAINING_CONTRACT["weight_decay"],
            "betas": list(TRAINING_CONTRACT["betas"]),
            "eps": TRAINING_CONTRACT["eps"],
            "steps": TRAINING_CONTRACT["steps"]}


def assert_training_contract(config=None, steps=None, horizon=None, runtime=None):
    """Raise unless the live configuration equals the frozen XR1 contract exactly.

    Covers all 16 recipe fields, not just the 11 the historical ID2 assertion covered. `runtime` is
    an optional mapping of the determinism settings actually in force (precision/amp/tf32/
    deterministic_algorithms/cublas_workspace_config); when supplied it is asserted too.
    """
    expected = TRAINING_CONTRACT
    if config is not None:
        live = {"microbatch": config["microbatch"], "accumulation": config["accumulation"],
                "warmup": config["warmup"], "lr_peak": config["lr_peak"],
                "lr_final_ratio": config["lr_final_ratio"], "clip_norm": config["clip_norm"],
                "eval_every": config["eval_every"], "weight_decay": config["weight_decay"],
                "betas": tuple(config["betas"]), "eps": config["eps"], "steps": config["steps"]}
        for key, value in live.items():
            assert value == expected[key], \
                f"XR1 training contract mismatch: {key} {value!r} != {expected[key]!r}"
    if runtime is not None:
        for key in ("precision", "amp", "tf32", "deterministic_algorithms",
                    "cublas_workspace_config"):
            if key in runtime:
                assert runtime[key] == expected[key], \
                    f"XR1 determinism contract mismatch: {key} {runtime[key]!r} != {expected[key]!r}"
    assert float(expected["lr_peak"]) * float(expected["lr_final_ratio"]) == expected["lr_final"]
    assert expected["microbatch"] * expected["accumulation"] == expected["effective_batch"]
    if steps is not None:
        assert int(steps) == expected["steps"], f"steps {steps} != {expected['steps']}"
    if horizon is not None:
        assert int(horizon) == expected["horizon"], f"horizon {horizon} != {expected['horizon']}"
    return True


def terminal_lr():
    """The LR the inherited schedule returns exactly at the terminal step (must be 1e-4)."""
    return learning_rate(TRAINING_CONTRACT["steps"], TRAINING_CONTRACT["lr_peak"],
                         TRAINING_CONTRACT["lr_final_ratio"], TRAINING_CONTRACT["warmup"],
                         TRAINING_CONTRACT["horizon"])


# ------------------------------------------------------------------------------ C7 block identity
def assert_boundary_block_identity(model, block, block_index=BLOCK_INDEX):
    """C7: the label block must be exactly the block whose prefix boundary produced H."""
    resolved = model.transformer.h[block_index]
    assert block is resolved, (
        f"C7 VIOLATION: the target block is not model.transformer.h[{block_index}] "
        f"(got {block!r}, expected {resolved!r}). A different block's boundary must never be paired "
        f"with this block's labels, even when the shapes match.")
    return resolved


def verify_label_block_identity(block, arms, tolerance=0.0):
    """C7 (independent route): recompute the labels from the boundary tensors and require identity.

    A bitwise (tolerance 0.0) comparison by default: `teacher_update` is deterministic, so any
    difference means the recorded labels did not come from this block and this boundary.
    """
    import torch

    from .h9_id1 import teacher_update
    recomputed = {"nat": teacher_update(block, arms["H_nat"]),
                  "pos": teacher_update(block, arms["H_pos"])}
    detail = {}
    for arm in ("nat", "pos"):
        recorded = arms[f"Y_{arm}"]
        max_abs = float((recomputed[arm].double() - recorded.double()).abs().max())
        detail[arm] = {"max_abs_difference": max_abs,
                       "bitwise_equal": bool(torch.equal(recomputed[arm], recorded)),
                       "accepted": bool(max_abs <= tolerance)}
    return {"passed": all(entry["accepted"] for entry in detail.values()),
            "block_index": arms.get("block_index"), "detail": detail, "tolerance": tolerance}


# ----------------------------------------------------------------------- evaluation authorization
def evaluation_eligibility_xr1(arm_statuses):
    """XR1 evaluation eligibility over the 24 XR1 replicate ids (see `h9_id2.evaluation_eligibility`)."""
    return evaluation_eligibility(arm_statuses, expected_students=PLANNED_STUDENTS,
                                  replicates=REPLICATE_IDS)


def assert_no_formal_seeds():
    """Preflight guard: XR1's own formal RNG manifest must not exist yet."""
    manifest = Path(FORMAL_RNG_MANIFEST)
    return {"formal_rng_manifest": str(manifest), "formal_rng_manifest_exists": manifest.exists(),
            "expected": False, "passed": not manifest.exists(),
            "note": "XR1 may not re-derive H9*-ID2's seeds and must not create its own in this phase"}


def execution_identity_reference():
    """D10: the XR1 code identity lives in the execution closure, not here.

    Returns a lightweight POINTER (paths plus a status), never a hash snapshot, so nothing in this
    module can go stale or compete with the authoritative closure.
    """
    return {
        "rng_identity_owner": {"artifact": str(FORMAL_RNG_MANIFEST), "status": "TO_BE_FROZEN_LATER"},
        "target_identity_owner": {"artifact": str(FROZEN_TARGET_LEDGER),
                                  "status": "TO_BE_MATERIALIZED_AND_FROZEN_LATER",
                                  "predecessor_corpus": str(PREDECESSOR_INVENTORY)},
        "code_identity_owner": {"artifact": "manifests/h9_xr1_execution_hashes.json",
                                "tag": EXECUTION_TAG, "status": "TO_BE_FROZEN_LATER"},
        "rule": "this module records no code hashes; the execution hash manifest and its immutable "
                "tag own scientific code identity (implementation design v0.2 §9.1)",
    }


def _file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def execution_authorization(tag=EXECUTION_TAG):
    """Is XR1 formal training authorized? Authorization state only — no execution semantics.

    Authorized iff XR1's own frozen RNG manifest and target ledger exist and match their recorded
    hashes, and XR1's own immutable execution tag resolves. `--verify --quiet` is required: plain
    `rev-parse <rev>` echoes its argument on failure, which once made an absent tag look resolved.
    """
    checks, reasons = {}, []
    for label, path, recorded in (("rng_manifest", FORMAL_RNG_MANIFEST, None),
                                  ("target_ledger", FROZEN_TARGET_LEDGER, None)):
        exists = Path(path).exists()
        checks[f"{label}_exists"] = exists
        if not exists:
            reasons.append(f"{path} is absent")
            checks[f"{label}_hash_matches"] = False
            continue
        if recorded is None:
            checks[f"{label}_hash_matches"] = None      # recorded at freeze time, not here
        else:
            checks[f"{label}_hash_matches"] = _file_sha256(path) == recorded
    resolved = ""
    try:
        import subprocess
        completed = subprocess.run(("git", "-c", f"safe.directory={Path.cwd().as_posix()}",
                                    "rev-parse", "--verify", "--quiet", f"{tag}^{{commit}}"),
                                   capture_output=True, text=True)
        resolved = completed.stdout.strip() if completed.returncode == 0 else ""
    except Exception as error:                                    # noqa: BLE001
        reasons.append(f"cannot resolve execution tag: {type(error).__name__}: {error}")
    checks["execution_tag_resolves"] = bool(resolved)
    checks["execution_tag_commit"] = resolved or None
    if not resolved:
        reasons.append(f"execution tag {tag} does not resolve")
    authorized = (checks["rng_manifest_exists"] and checks.get("rng_manifest_hash_matches") is not False
                  and checks["target_ledger_exists"]
                  and checks.get("target_ledger_hash_matches") is not False
                  and checks["execution_tag_resolves"])
    return {"authorized": authorized, "tag": tag, "checks": checks, "reasons": reasons,
            "note": "hash equality for the RNG manifest and the target ledger is recorded in the "
                    "execution closure at freeze time, not in this module"}
