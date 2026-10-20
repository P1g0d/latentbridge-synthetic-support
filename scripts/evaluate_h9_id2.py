"""H9*-ID2 Phase-B target evaluation and frozen analysis.

NOT AUTHORIZED TO RUN IN THIS PHASE. Phase-B only: it is launched as a separate process after the
training barrier, and asserts that barrier before doing anything else.

Barrier: every planned Student must have a terminal status recorded in the campaign summary. If the
barrier fails, this script exits without computing anything — so no partial R_P, D_value, D_context,
ContextNMSE, KL, NLL, significance or arm comparison can be surfaced during training.

Usage (NOT AUTHORIZED): python scripts/evaluate_h9_id2.py --root artifacts/h9_id2
"""
import argparse
import json
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["HF_HUB_OFFLINE"] = "1"
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path("src").resolve()))
from silent_teacher.b0_phase1 import derangement                                       # noqa: E402
from silent_teacher.full_block import (ExactFullBlock, FullBlockStudent,                # noqa: E402
                                       ZeroUpdateControl, full_block_context_evaluation)
from silent_teacher.functional_replacement import (aggregate_functional_results,        # noqa: E402
                                                   functional_metrics, run_teacher,
                                                   run_with_replacement)
from silent_teacher.h9_id1 import prefix_states, teacher_update                         # noqa: E402
from silent_teacher.h9_id2 import (LENGTH, ROLES, evaluation_eligibility)                # noqa: E402
from silent_teacher.h9_id2_analysis import (A_CONTEXT, A_VALUE, DELTA_CONTEXT,          # noqa: E402
                                            DELTA_VALUE, ENVIRONMENTS, classify,
                                            context_endpoint, paired_differences,
                                            value_endpoint)
from silent_teacher.modules import MLP, freeze                                          # noqa: E402
from silent_teacher.records import file_hash, state_hash                                # noqa: E402
from silent_teacher.train import setup                                                  # noqa: E402
from transformers import GPT2Config, GPT2LMHeadModel                                    # noqa: E402
from safetensors import safe_open                                                       # noqa: E402

REPLICATES = 48
CUT_QUERY = 63
FUNCTIONAL_BATCH = 4
WEIGHTS_SHA = "248dfc3911869ec493c76e65bf2fcf7f615828b0254c12b473182f0f81d3a707"
TEACHER_MLP_HASH = "ec48c1f164aa39f19a875afc3bfbf408505dcb2dbb239f5a4f17a603be9b18dc"


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


def evaluation_barrier(root):
    """Blindness + eligibility barrier.

    Two distinct questions, deliberately not conflated:

      * OPERATIONALLY STOPPED — the attempt has ended (which is why no further training output is
        expected), and
      * RESOLVED AND ELIGIBLE — the attempt ended in a state that formal target evaluation may use.

    An unresolved unknown failure is operationally stopped but is NOT an eligible resolved state, so
    target evaluation must refuse while any replicate carries it.
    """
    summary_path = root / "replicates" / "driver_summary.json"
    assert summary_path.exists(), f"barrier: {summary_path} absent"
    statuses = {}
    for replicate in range(1, REPLICATES + 1):
        record_path = root / "replicates" / f"replicate_{replicate:02d}" / "replicate.json"
        if not record_path.exists():
            continue
        record = json.loads(record_path.read_text(encoding="utf-8"))
        statuses[replicate] = {arm: entry.get("status")
                               for arm, entry in record.get("arms", {}).items()}
    return evaluation_eligibility(statuses)


@torch.no_grad()
def value_nmse(student, hidden, labels, device, microbatch=8):
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
    entries = []
    for start in range(0, len(token_ids), FUNCTIONAL_BATCH):
        chunk = token_ids[start:start + FUNCTIONAL_BATCH].to(device)
        teacher_logits = run_teacher(model, chunk)
        replaced = run_with_replacement(model, student, chunk)
        entries.append(functional_metrics(teacher_logits, replaced, chunk))
        del teacher_logits, replaced
    return aggregate_functional_results(entries)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("artifacts/h9_id2"))
    parser.add_argument("--synthetic-classifier-only", action="store_true",
                        help="exercise the classifier on synthetic fixtures only")
    args = parser.parse_args()

    if args.synthetic_classifier_only:
        from silent_teacher.h9_id2_analysis import run_fixtures
        run = run_fixtures()
        print(json.dumps({name: {"state": entry["state"], "reason": entry["reason"],
                                 "expected": entry["expected"], "passed": entry["passed"]}
                          for name, entry in run["fixtures"].items()}, indent=2))
        print(json.dumps({"all_fixtures_passed": run["passed"], "failed": run["failed"]}))
        return 0 if run["passed"] else 1

    barrier = evaluation_barrier(args.root)
    assert barrier["eligible"], (
        f"target evaluation REFUSED: {barrier['resolved_students']}/"
        f"{barrier['expected_students']} Students are in a resolved evaluation-eligible state. "
        f"Blocking: {barrier['blocking'][:5]} Missing: {barrier['missing'][:5]}. An operationally "
        f"stopped attempt is not an eligible resolved state.")
    print("H9*-ID2 target evaluation is NOT AUTHORIZED in this phase.")
    return 2


if __name__ == "__main__":
    sys.exit(main())
