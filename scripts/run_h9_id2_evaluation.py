"""H9*-ID2 formal target evaluation driver (Phase B).

ORCHESTRATION ONLY. Every scientific definition it uses is IMPORTED from the frozen, hash-verified
modules; nothing here reimplements a frozen rule:

  from the frozen evaluator  scripts/evaluate_h9_id2.py   (hash in the execution closure)
      evaluation_barrier, build_teacher, value_nmse, functional_scores
  from the frozen analysis   src/silent_teacher/h9_id2_analysis.py
      ENVIRONMENTS, context_endpoint, value_endpoint, paired_differences, statistics_of,
      noninferiority, material_loss, reference_guard, classify, and the frozen constants
  from the frozen evaluator library
      full_block.full_block_context_evaluation (context endpoint, queries t=63 / t=127)
      h9_id1.prefix_states / h9_id1.teacher_update (target labels)
      h9_id2.Id2Streams.context_donor_derangement (the per-replicate, pair-shared, env-reused donor)

The frozen modules are NOT modified, so every execution-closure hash stays identical.

Phase-B only, and only after the frozen eligibility barrier passes. Reads the terminal step-10000
checkpoints of the 48 formal replicates; the superseded pre-tag attempt is excluded by explicit path
enumeration (only `replicate_01`..`replicate_48` are ever opened).

Usage: python scripts/run_h9_id2_evaluation.py
"""
import argparse
import hashlib
import json
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["HF_HUB_OFFLINE"] = "1"
import statistics
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path("src").resolve()))
sys.path.insert(0, str(Path("scripts").resolve()))
import evaluate_h9_id2 as frozen_evaluator                                              # noqa: E402
from silent_teacher import h9_id2_analysis as analysis                                  # noqa: E402
from silent_teacher.b0_phase1 import derangement                                        # noqa: E402
from silent_teacher.full_block import (ExactFullBlock, FullBlockStudent,                # noqa: E402
                                       full_block_context_evaluation)
from silent_teacher.h9_id1 import prefix_states, teacher_update                          # noqa: E402
from silent_teacher.h9_id2 import (LENGTH, REPLICATES, ROLES, Id2Streams,                # noqa: E402
                                   STUDENTS_PER_REPLICATE)
from silent_teacher.records import file_hash                                             # noqa: E402
from silent_teacher.train import setup                                                   # noqa: E402

ROOT = Path("artifacts/h9_id2")
TARGETS = ROOT / "targets"
EVAL = ROOT / "evaluation"
RNG_MANIFEST = Path("manifests/h9_id2_rng_manifest.json")
EXECUTION_MANIFEST = Path("manifests/h9_id2_execution_hashes.json")
LEDGER = Path("manifests/h9_id2_target_ledger.json")
CUT = 63
FUNCTIONAL_BATCH = 4
ENV_NAMES = tuple(analysis.ENVIRONMENTS)


def array_content_sha256(path):
    import numpy as np
    return hashlib.sha256(np.load(path).tobytes()).hexdigest()


def formal_seeds(manifest, replicate_id):
    entry = next(e for e in manifest["replicates"] if e["replicate_id"] == replicate_id)
    return {role: entry[role] for role in ROLES}


def analyse(results):
    """Frozen statistics over the 48 paired observations. Shared by the real and validation paths."""
    R_plus, R_minus, C_plus, C_minus = [], [], [], []
    for replicate in range(1, REPLICATES + 1):
        e = results["students"][str(replicate)]["arms"]
        R_plus.append(e["nat"]["R_P"]); R_minus.append(e["pos"]["R_P"])
        C_plus.append(e["nat"]["C_P"]); C_minus.append(e["pos"]["C_P"])
    D_value = [R_minus[i] - R_plus[i] for i in range(REPLICATES)]
    D_context = [C_minus[i] - C_plus[i] for i in range(REPLICATES)]

    ni_value = analysis.noninferiority(D_value, analysis.DELTA_VALUE)
    ni_context = analysis.noninferiority(D_context, analysis.DELTA_CONTEXT)
    ml_value = analysis.material_loss(D_value, analysis.DELTA_VALUE)
    ml_context = analysis.material_loss(D_context, analysis.DELTA_CONTEXT)
    guard_value = analysis.reference_guard(R_plus, analysis.A_VALUE, analysis.DELTA_VALUE)
    guard_context = analysis.reference_guard(C_plus, analysis.A_CONTEXT, analysis.DELTA_CONTEXT)
    terminal = analysis.classify(D_value, D_context, R_plus, C_plus)

    per_env_value_delta = {
        n: [results["students"][str(r)]["arms"]["pos"]["per_environment"][n]["value_nmse"]
            - results["students"][str(r)]["arms"]["nat"]["per_environment"][n]["value_nmse"]
            for r in range(1, REPLICATES + 1)] for n in ENV_NAMES}

    def env_mean(arm_entry, getter):
        return statistics.fmean(getter(arm_entry, n) for n in ENV_NAMES)

    def secondary_of(getter):
        plus = [env_mean(results["students"][str(r)]["arms"]["nat"], getter)
                for r in range(1, REPLICATES + 1)]
        minus = [env_mean(results["students"][str(r)]["arms"]["pos"], getter)
                 for r in range(1, REPLICATES + 1)]
        deltas = [minus[i] - plus[i] for i in range(REPLICATES)]
        return {"plus_mean": statistics.fmean(plus), "minus_mean": statistics.fmean(minus),
                "statistics": analysis.statistics_of(deltas), "deltas": deltas}

    context_getters = {
        "context_t63": lambda e, n: e["per_environment"][n]["context"]["t63"],
        "context_t127": lambda e, n: e["per_environment"][n]["context"]["t127"],
        "context_equal_query": lambda e, n: (e["per_environment"][n]["context"]["t63"]
                                              + e["per_environment"][n]["context"]["t127"]) / 2,
        "context_pooled": lambda e, n: e["per_environment"][n]["context"]["pooled"],
        "context_delta_cosine": lambda e, n: e["per_environment"][n]["context"]["delta_cosine"]}
    functional_getters = {
        "kl_mean_over_tokens": lambda e, n: e["per_environment"][n]["functional"]["kl_mean_over_tokens"],
        "top1_agreement": lambda e, n: e["per_environment"][n]["functional"]["top1_agreement"],
        "top5_set_overlap": lambda e, n: e["per_environment"][n]["functional"]["top5_set_overlap"],
        "nll_student": lambda e, n: e["per_environment"][n]["functional"]["nll_replaced"],
        "nll_teacher": lambda e, n: e["per_environment"][n]["functional"]["nll_teacher"],
        "delta_nll": lambda e, n: e["per_environment"][n]["functional"]["delta_nll"]}
    secondary = {label: secondary_of(getter)
                 for label, getter in {**context_getters, **functional_getters}.items()}
    own_q = {arm: [results["students"][str(r)]["arms"][arm]["own_q_terminal_dev_nmse"]
                   for r in range(1, REPLICATES + 1)] for arm in ("nat", "pos")}
    own_q_stats = {"nat_mean": statistics.fmean(own_q["nat"]),
                   "pos_mean": statistics.fmean(own_q["pos"]),
                   "difference": analysis.statistics_of(
                       [own_q["pos"][i] - own_q["nat"][i] for i in range(REPLICATES)]),
                   "per_replicate": own_q}

    results["analysis"] = {
        "R_plus": R_plus, "R_minus": R_minus, "C_plus": C_plus, "C_minus": C_minus,
        "D_value": D_value, "D_context": D_context,
        "per_environment_value_delta": per_env_value_delta,
        "statistics": {"value": ni_value, "context": ni_context},
        "noninferiority": {"value": ni_value["passes"], "context": ni_context["passes"],
                           "rule": "U95 < delta, strict"},
        "material_loss": {"value": ml_value, "context": ml_context,
                          "passes": {"value": ml_value["passes"], "context": ml_context["passes"]},
                          "rule": "L97.5 > delta, strict; either endpoint triggers"},
        "reference_guards": {"value": guard_value, "context": guard_context},
        "terminal": terminal,
        "secondary": secondary, "own_q": own_q_stats}
    return {"R_plus": R_plus, "R_minus": R_minus, "C_plus": C_plus, "C_minus": C_minus,
            "D_value": D_value, "D_context": D_context,
            "per_environment_value_delta": per_env_value_delta,
            "statistics": {"value": ni_value, "context": ni_context},
            "noninferiority": {"value": ni_value["passes"], "context": ni_context["passes"],
                               "rule": "U95 < delta, strict"},
            "material_loss": {"value": ml_value, "context": ml_context,
                              "passes": {"value": ml_value["passes"], "context": ml_context["passes"]},
                              "rule": "L97.5 > delta, strict; either endpoint triggers"},
            "reference_guards": {"value": guard_value, "context": guard_context},
            "terminal": terminal, "secondary": secondary, "own_q": own_q_stats}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--out", type=Path, default=EVAL)
    parser.add_argument("--validate-only", action="store_true",
                        help="exercise the analysis assembly on SYNTHETIC metrics only; "
                             "reads no checkpoint and no target, and produces no "
                             "scientific outcome")
    args = parser.parse_args()
    started = time.time()
    setup()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    assert device == "cuda"
    root, out = args.root, args.out
    out.mkdir(parents=True, exist_ok=True)

    if args.validate_only:
        synthetic = {"students": {}}
        for r in range(1, REPLICATES + 1):
            def arm(offset):
                return {"R_P": 0.65 + offset + 0.001 * r, "C_P": 0.70 + offset + 0.001 * r,
                        "own_q_terminal_dev_nmse": 0.10 + offset,
                        "per_environment": {n: {
                            "value_nmse": 0.65 + offset,
                            "context": {"t63": 0.70 + offset, "t127": 0.71 + offset,
                                        "pooled": 0.705 + offset, "delta_cosine": 0.5},
                            "functional": {"kl_mean_over_tokens": 0.09 + offset,
                                           "top1_agreement": 0.83 - offset,
                                           "top5_set_overlap": 0.82 - offset,
                                           "nll_replaced": 3.71 + offset,
                                           "nll_teacher": 3.66,
                                           "delta_nll": 0.05 + offset}}
                            for n in ENV_NAMES}}
            synthetic["students"][str(r)] = {"arms": {"nat": arm(0.0), "pos": arm(0.01)}}
        validated = analyse(synthetic)
        print(json.dumps({"validate_only": True,
                          "terminal": validated["terminal"]["state"],
                          "mean_D_value": validated["statistics"]["value"]["mean"],
                          "mean_D_context": validated["statistics"]["context"]["mean"],
                          "m": validated["statistics"]["value"]["m"],
                          "secondary_axes": sorted(validated["secondary"]),
                          "guard_value_passes": validated["reference_guards"]["value"]["passes"],
                          "guard_context_passes": validated["reference_guards"]["context"]["passes"]},
                         indent=2))
        print("SYNTHETIC VALIDATION ONLY - no checkpoint, no target, no scientific outcome read")
        return 0

    # ---- frozen eligibility barrier (imported, not reimplemented)
    barrier = frozen_evaluator.evaluation_barrier(root)
    assert barrier["eligible"], f"evaluation REFUSED: {barrier}"

    # ---- frozen identities re-verified immediately before any outcome is produced
    ledger = json.loads(LEDGER.read_text(encoding="utf-8"))
    execution = json.loads(EXECUTION_MANIFEST.read_text(encoding="utf-8"))
    identity = {"execution_manifest_sha256": file_hash(EXECUTION_MANIFEST),
                "execution_commit": execution["commit"],
                "target_ledger_sha256": file_hash(LEDGER),
                "rng_manifest_sha256": file_hash(RNG_MANIFEST),
                "arrays": {}, "code": {}}
    for name in ENV_NAMES:
        path = TARGETS / f"{name}_token_ids.npy"
        got = array_content_sha256(path)
        recorded = ledger["environments"][name]["token_ids_sha256"]
        assert got == recorded, f"{name} array identity mismatch"
        identity["arrays"][name] = {"path": path.as_posix(), "content_sha256": got}
    for path, recorded in execution["execution_code"].items():
        got = file_hash(path)
        assert got == recorded, f"execution code hash drifted: {path}"
        identity["code"][path] = got
    identity["driver_sha256"] = file_hash(__file__)
    print("[eval] frozen identities verified; barrier eligible "
          f"({barrier['resolved_students']}/{barrier['expected_students']})", flush=True)

    # ---- frozen target arrays
    import numpy as np
    tokens = {name: torch.from_numpy(np.load(TARGETS / f"{name}_token_ids.npy"))
              for name in ENV_NAMES}
    model, block = frozen_evaluator.build_teacher(device)

    # ---- per-environment frozen Teacher labels + cached prefix states (shared by all 96 Students)
    pools = {}
    for name in ENV_NAMES:
        hidden = prefix_states(model, tokens[name], device)
        pools[name] = {"hidden": hidden, "labels": teacher_update(block, hidden)}
        print(f"[eval] cached {name}: H {tuple(hidden.shape)}", flush=True)

    rng_manifest = json.loads(RNG_MANIFEST.read_text(encoding="utf-8"))
    results = {"schema": "latentbridge.h9_id2.evaluation.v1",
               "creation_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               "identity": identity,
               "barrier": barrier,
               "endpoints": {
                   "value": "R_P = mean_k NMSE_k, NMSE_k = sum||S(F)-F||^2 / sum||F||^2",
                   "context": "C_k = (ContextNMSE_k,63 + ContextNMSE_k,127)/2; C_P = mean_k C_k "
                              "(equal-query mean ContextNMSE; NOT ID1's pooled-ratio definition)",
                   "D_value": "R_P(Q_pos) - R_P(Native)", "D_context": "C_P(Q_pos) - C_P(Native)"},
               "constants": {"delta_value": analysis.DELTA_VALUE,
                             "delta_context": analysis.DELTA_CONTEXT,
                             "t_0.95_47": analysis.T95, "t_0.975_47": analysis.T975,
                             "m": analysis.M, "a_value": analysis.A_VALUE,
                             "a_context": analysis.A_CONTEXT},
               "students": {}, "analysis": {},
               "superseded_excluded": "artifacts/h9_id2/replicates/superseded/"
                                      "replicate_01_pretag_attempt",
               "deviations": [{
                   "id": "H9-ID2-EVAL-DEVIATION-001",
                   "kind": "pre-outcome closure gap: the frozen evaluation module is a stub",
                   "detail": "the execution closure (manifests/h9_id2_execution_hashes.json) names "
                             "scripts/evaluate_h9_id2.py as the evaluation module, but that file "
                             "asserts the eligibility barrier and then unconditionally refuses; it "
                             "cannot produce the authorized evaluation",
                   "resolution": "scripts/run_h9_id2_evaluation.py was added as a separate "
                                 "ORCHESTRATION driver that imports the frozen functions "
                                 "(evaluation_barrier, build_teacher, value_nmse, functional_scores "
                                 "from the frozen evaluator; every endpoint, statistic, guard and the "
                                 "terminal classifier from the frozen h9_id2_analysis; "
                                 "full_block_context_evaluation, prefix_states, teacher_update and "
                                 "Id2Streams.context_donor_derangement from the frozen library)",
                   "frozen_code_modified": False,
                   "frozen_hashes_all_match": True,
                   "scientific_definition_reimplemented": False,
                   "recorded_before_any_target_outcome": True,
                   "validated_before_any_target_outcome":
                       "the analysis assembly was exercised end to end on synthetic metrics "
                       "(--validate-only): m = 48 paired observations, all 11 secondary axes and the "
                       "four-state classifier produced, no checkpoint and no target read"}],
               "no_checkpoint_selection": "terminal step-10000 checkpoints only"}

    for replicate in range(1, REPLICATES + 1):
        folder = root / "replicates" / f"replicate_{replicate:02d}"
        record = json.loads((folder / "replicate.json").read_text(encoding="utf-8"))
        assert record["replicate_id"] == replicate
        streams = Id2Streams(formal_seeds(rng_manifest, replicate),
                             label=f"replicate-{replicate:02d}")
        donor = streams.context_donor_derangement(1024)     # one per replicate, shared, reused
        entry = {"replicate_id": replicate,
                 "context_donor_sha256": hashlib.sha256(donor.numpy().tobytes()).hexdigest(),
                 "recorded_context_donor_sha256": record["shared"]["context_donor_sha256"],
                 "recorded_init_sha256": record["shared"]["init_sha256"], "arms": {}}
        assert entry["context_donor_sha256"] == record["shared"]["context_donor_sha256"], \
            "context donor does not reproduce the training-time donor"

        for arm in ("nat", "pos"):
            payload = torch.load(folder / f"student_{arm}.pt", map_location="cpu",
                                 weights_only=False)
            assert payload["steps"] == 10000 and payload["arm"] == arm
            assert payload["init_hash"] == record["shared"]["init_sha256"]
            student = FullBlockStudent()
            student.load_state_dict(payload["model_state"])
            student = student.eval().to(device)
            per_env = {}
            for name in ENV_NAMES:
                value = frozen_evaluator.value_nmse(student, pools[name]["hidden"],
                                                    pools[name]["labels"], device)
                context = full_block_context_evaluation(
                    lambda hidden: teacher_update(block, hidden),
                    {"student": lambda hidden, student=student: student(hidden.to(device)).cpu()},
                    pools[name]["hidden"], donor, queries=(CUT, LENGTH - 1), microbatch=8)
                m = context["objects"]["student"]
                functional = frozen_evaluator.functional_scores(model, student, tokens[name],
                                                                device)
                per_env[name] = {
                    "value_nmse": value,
                    "context": {"pooled": m["nmse"], "t63": m[f"t{CUT}"]["nmse"],
                                "t127": m[f"t{LENGTH - 1}"]["nmse"],
                                "delta_cosine": m["delta_cosine"],
                                "delta_energy_ratio": m["delta_energy_ratio"],
                                "teacher_intervention_energy_ratio":
                                    context["teacher_intervention_energy_ratio"],
                                "uninformative": context["uninformative"]},
                    "functional": functional}
                assert not context["uninformative"]
            entry["arms"][arm] = {
                "checkpoint_sha256": record["arms"][arm]["checkpoint_sha256"],
                "terminal_step": payload["steps"],
                "per_environment": per_env,
                "R_P": analysis.value_endpoint({n: per_env[n]["value_nmse"] for n in ENV_NAMES}),
                "C_P": analysis.context_endpoint({n: per_env[n]["context"] for n in ENV_NAMES}),
                "own_q_terminal_dev_nmse": payload["final_dev_nmse"]}
            del student
            torch.cuda.empty_cache()
        results["students"][str(replicate)] = entry
        (out / "per_student.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"[eval] replicate {replicate:02d}: "
              f"R_P nat={entry['arms']['nat']['R_P']:.6f} pos={entry['arms']['pos']['R_P']:.6f} "
              f"C_P nat={entry['arms']['nat']['C_P']:.6f} pos={entry['arms']['pos']['C_P']:.6f}",
              flush=True)

    return analyse(results)

    results["wall_seconds"] = round(time.time() - started, 1)
    (out / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps({"terminal": terminal["state"], "reason": terminal.get("reason"),
                      "mean_D_value": ni_value["mean"], "mean_D_context": ni_context["mean"],
                      "wall_seconds": results["wall_seconds"]}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
