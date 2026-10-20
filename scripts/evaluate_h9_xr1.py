"""H9*-XR1 formal target-evaluation library (Phase B).

Phase-B ONLY. Every function here is imported by the evaluation driver, which asserts the eligibility
barrier before calling any of them. This module reads the frozen target arrays only inside
`load_frozen_targets`, which the driver calls after the barrier has passed.

All scientific definitions are IMPORTED from frozen, hash-verified modules; nothing here reimplements
a frozen rule:
    * the endpoints and classifier           -> src/silent_teacher/h9_xr1_analysis.py
    * the context evaluation                 -> src/silent_teacher/full_block.py
    * the block-9 prefix and Teacher update  -> src/silent_teacher/h9_id1.py (via prefix_states/teacher_update)
    * the frozen context-donor convention    -> src/silent_teacher/h9_id2.py (Id2Streams)
"""
import hashlib
import json
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path("src").resolve()))
from silent_teacher import h9_xr1 as X                                              # noqa: E402
from silent_teacher import h9_xr1_analysis as A                                     # noqa: E402
from silent_teacher.full_block import FullBlockStudent, full_block_context_evaluation  # noqa: E402
from silent_teacher.functional_replacement import (aggregate_functional_results,   # noqa: E402
                                                   functional_metrics, run_teacher,
                                                   run_with_replacement)
from silent_teacher.h9_id1 import prefix_states, teacher_update                     # noqa: E402
from silent_teacher.h9_id2 import ROLES, Id2Streams                                 # noqa: E402
from silent_teacher.records import file_hash                                        # noqa: E402
from silent_teacher.train import setup                                              # noqa: E402
from safetensors import safe_open                                                   # noqa: E402
from transformers import GPT2Config, GPT2LMHeadModel                                # noqa: E402

LEDGER = Path("manifests/h9_xr1_target_ledger.json")
RNG_MANIFEST = Path("manifests/h9_xr1_rng_manifest_v5.json")
EXECUTION_MANIFEST = Path("manifests/h9_xr1_execution_hashes.json")
FUNCTIONAL_BATCH = 4
CUT_QUERY = 63
LAST_QUERY = X.LENGTH - 1


def array_content_sha256(path):
    """Content identity of a token array: sha256 of the int64 byte image (the ledger's convention)."""
    return hashlib.sha256(np.load(path).tobytes()).hexdigest()


def formal_seeds(manifest, replicate_id):
    entry = next(e for e in manifest["replicates"] if e["replicate_id"] == replicate_id)
    return {role: entry[role] for role in ROLES}


def build_teacher(device):
    """The frozen Teacher and the block-9 operator this campaign targets."""
    cfg = GPT2Config.from_json_file("artifacts/gpt2/config.json")
    model = GPT2LMHeadModel(cfg).eval()
    with safe_open("artifacts/gpt2/model.safetensors", framework="pt", device="cpu") as handle:
        state = {("lm_head.weight" if key == "lm_head.weight" else f"transformer.{key}"):
                 handle.get_tensor(key) for key in handle.keys()
                 if not key.endswith((".attn.bias", ".attn.masked_bias"))}
    missing, unexpected = model.load_state_dict(state, strict=False)
    assert missing == ["lm_head.weight"] and not unexpected
    model = model.to(device)
    block = model.transformer.h[X.BLOCK_INDEX]
    X.assert_boundary_block_identity(model, block, X.BLOCK_INDEX)
    return model, block


def load_frozen_targets(targets_dir, ledger):
    """Load the frozen token arrays and re-verify each one against the frozen ledger."""
    tokens, identity = {}, {}
    for environment in ledger["environment_order"]:
        path = Path(ledger["environments"][environment]["token_ids_file"])
        got = array_content_sha256(path)
        recorded = ledger["environments"][environment]["token_ids_sha256"]
        assert got == recorded, f"{environment}: target array identity mismatch"
        array = torch.from_numpy(np.load(path))
        assert tuple(array.shape) == (X.REPLICATES * 0 + 1024, 128), array.shape
        tokens[environment] = array
        identity[environment] = {"path": path.as_posix(), "content_sha256": got}
    return tokens, identity


@torch.no_grad()
def value_nmse(student, hidden, labels, device, microbatch=8):
    """Frozen Value endpoint: corpus ratio of sums ||S(H) - F9(H)||^2 / ||F9(H)||^2."""
    numerator = denominator = 0.0
    for start in range(0, len(hidden), microbatch):
        target = labels[start:start + microbatch].to(device)
        output = student(hidden[start:start + microbatch].to(device))
        difference = output.double() - target.double()
        numerator += float(difference.square().sum())
        denominator += float(target.double().square().sum())
    return numerator / denominator


@torch.no_grad()
def functional_scores(model, student, token_ids, device):
    """Frozen secondary functional diagnostics (KL, NLL, top-1, top-5). Never a terminal input."""
    entries = []
    for start in range(0, len(token_ids), FUNCTIONAL_BATCH):
        chunk = token_ids[start:start + FUNCTIONAL_BATCH].to(device)
        teacher_logits = run_teacher(model, chunk)
        replaced = run_with_replacement(model, student, chunk, block_index=X.BLOCK_INDEX)
        entries.append(functional_metrics(teacher_logits, replaced, chunk))
        del teacher_logits, replaced
    return aggregate_functional_results(entries)


#: EVERY key the XR1 evaluator/driver consumes from the frozen functional-secondary result mapping.
#: This list is the contract: it must remain a subset of what the frozen aggregator actually returns,
#: and no key may be added merely because the mapping happens to contain it.
FUNCTIONAL_CONSUMED_KEYS = ("kl_mean_over_tokens", "top1_agreement", "top5_set_overlap",
                            "nll_replaced", "nll_teacher", "delta_nll")


def consume_functional_result(functional):
    """Read the frozen secondaries through ONE audited access point.

    The v8 evaluation attempt failed here with `KeyError: 'delta_nll_mean'` — a hand-written key that
    the frozen aggregator does not return. Routing every access through this function, and asserting
    the contract explicitly, is what closes that class of defect.
    """
    missing = [key for key in FUNCTIONAL_CONSUMED_KEYS if key not in functional]
    if missing:
        raise KeyError(f"functional result is missing consumed key(s) {missing}; "
                       f"available keys are {sorted(functional)}")
    return {key: functional[key] for key in FUNCTIONAL_CONSUMED_KEYS}


def assemble_student_metrics(per_environment, own_q_terminal_dev_nmse):
    """Assemble the persisted Student metrics payload from the computed pieces.

    `per_environment[environment] = {"value_nmse": float, "context": {...}, "functional": {...}}`.
    The two endpoints are the frozen EQUAL-ENVIRONMENT aggregations; the functional block is read
    through the audited consumer. Shared by the driver and by the preflight fixture so that the
    fixture exercises the real access path rather than imitating it.
    """
    from silent_teacher import h9_xr1_analysis as _A
    assert set(per_environment) == {"c4", "pes2o", "se"}, sorted(per_environment)
    environments = {}
    for environment, entry in per_environment.items():
        environments[environment] = {
            "value_nmse": entry["value_nmse"],
            "context": {"pooled": entry["context"]["pooled"], "t63": entry["context"]["t63"],
                        "t127": entry["context"]["t127"],
                        "delta_cosine": entry["context"]["delta_cosine"]},
            "functional": consume_functional_result(entry["functional"])}
    # the frozen aggregators take FLAT per-environment mappings: value_endpoint reads the normalized
    # Value error, context_endpoint reads t63/t127 directly (it forms the equal-query mean itself)
    value_map = {n: environments[n]["value_nmse"] for n in ("c4", "pes2o", "se")}
    context_map = {n: {"t63": environments[n]["context"]["t63"],
                       "t127": environments[n]["context"]["t127"]}
                   for n in ("c4", "pes2o", "se")}
    return {"R_P": _A.value_endpoint(value_map),
            "C_P": _A.context_endpoint(context_map),
            "environments": environments,
            "own_q_terminal_dev_nmse": own_q_terminal_dev_nmse}


def evaluation_barrier(root):
    """Frozen eligibility barrier: every planned Student must be in a resolved, eligible state."""
    statuses = {}
    for replicate in X.REPLICATE_IDS:
        record_path = Path(root) / f"replicate_{replicate:02d}" / "replicate.json"
        if not record_path.exists():
            continue
        record = json.loads(record_path.read_text(encoding="utf-8"))
        statuses[replicate] = {arm: entry.get("status")
                               for arm, entry in record.get("arms", {}).items()}
    return X.evaluation_eligibility_xr1(statuses)
