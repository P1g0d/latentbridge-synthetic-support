"""H9*-XR1 formal target-evaluation driver (Phase B).

ORCHESTRATION ONLY. Every scientific definition is imported from the frozen, hash-verified modules; the
driver reimplements no frozen rule.

Ordering is enforced structurally, to make the H9*-ID2 persistence incident impossible:

    gate (explicit execution identity)
      -> eligibility barrier
      -> frozen identity re-verification (ledger, arrays, checkpoints)
      -> one pass over the 48 frozen Students, producing confirmatory metrics
      -> PERSIST artifacts/h9_xr1/evaluation/per_student.json
      -> close, RE-OPEN from disk, validate, hash
      -> ONLY THEN aggregate analysis, reading that persisted artifact and nothing else

Aggregate analysis calls no model and computes no metric. The classifier is the frozen
h9_xr1_analysis module.

Usage: python scripts/run_h9_xr1_evaluation.py [--root artifacts/h9_xr1]
"""
import argparse
import hashlib
import json
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path("src").resolve()))
sys.path.insert(0, str(Path("scripts").resolve()))
import evaluate_h9_xr1 as E                                                          # noqa: E402
from silent_teacher import h9_xr1 as X                                               # noqa: E402
from silent_teacher import h9_xr1_analysis as A
from silent_teacher import h9_xr1_store as S                                      # noqa: E402
from silent_teacher.full_block import FullBlockStudent, full_block_context_evaluation  # noqa: E402
from silent_teacher.h9_id1 import prefix_states, teacher_update                      # noqa: E402
from silent_teacher.h9_id2 import ROLES, Id2Streams                                  # noqa: E402
from silent_teacher.records import file_hash                                         # noqa: E402
from silent_teacher.train import setup                                               # noqa: E402
from silent_teacher.h9_xr1_freeze import FreezeGateHardStop, freeze_gate             # noqa: E402

#: the execution identity that authorizes this evaluation. It must exist before any target is read.
EXPECTED_EXECUTION_TAG = "h9-xr1-execution-frozen-v9"


def _sha_of_file(path):
    return file_hash(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("artifacts/h9_xr1"))
    parser.add_argument("--synthetic-e2e", action="store_true",
                        help="run the complete orchestration path with synthetic metrics through "
                             "persistence, simulated restart, consolidation and analysis; reads no "
                             "target and evaluates no model")
    parser.add_argument("--dry-run-pre-freeze", action="store_true",
                        help="exercise the pre-target wiring against the committed candidate tree and "
                             "STOP before reading any target content")
    parser.add_argument("--validate-only", action="store_true",
                        help="exercise the analysis assembly on SYNTHETIC metrics; reads no checkpoint "
                             "and no target")
    args = parser.parse_args()

    root = args.root
    evaluation_dir = root / "evaluation"
    per_student_path = evaluation_dir / "per_student.json"
    results_path = evaluation_dir / "results.json"

    if args.synthetic_e2e:
        import shutil
        import tempfile
        from silent_teacher import h9_xr1_store as ST
        from silent_teacher.h9_xr1_freeze import EXECUTION_MANIFEST, worktree_blob
        closure = json.loads(EXECUTION_MANIFEST.read_text(encoding="utf-8"))
        drift = [p for p, e in closure["execution_code"].items()
                 if Path(p).exists() and worktree_blob(p) != e["git_blob"]]
        dirty = __import__("subprocess").run(["git", "status", "--porcelain"],
                                            capture_output=True, text=True).stdout.strip()
        checks = {"candidate_identity_accepted": not (dirty or drift)}
        barrier = E.evaluation_barrier(root / "replicates")
        checks["eligibility_48_of_48"] = barrier["eligible"] and barrier["resolved_students"] == 48
        # real checkpoint bindings (read-only, no target)
        bound = 0
        for replicate in X.REPLICATE_IDS:
            folder = root / "replicates" / f"replicate_{replicate:02d}"
            record = json.loads((folder / "replicate.json").read_text(encoding="utf-8"))
            for arm in ("nat", "pos"):
                ck = folder / f"student_{arm}.pt"
                bound += 1 if _sha_of_file(ck) == record["arms"][arm]["checkpoint_sha256"] else 0
        checks["checkpoint_bindings_48_of_48"] = bound == 48
        ledger_syn = json.loads(E.LEDGER.read_text(encoding="utf-8"))
        checks["target_ledger_identity_present"] = bool(ledger_syn.get("status") == "TARGET_LEDGER_FROZEN")
        checks["environments_are_the_frozen_three"] = tuple(ledger_syn["environment_order"]) == \
            ("c4", "pes2o", "se")
        checks["endpoints_are_value_and_context"] = tuple(A.ENDPOINTS) == ("value", "context")
        checks["query_positions_are_63_and_127"] = (E.CUT_QUERY, E.LAST_QUERY) == (63, 127)
        checks["m_is_24"] = A.M == 24 and len(X.REPLICATE_IDS) == 24
        tmp = Path(tempfile.mkdtemp())
        try:
            common = {"execution_tag": EXPECTED_EXECUTION_TAG,
                      "execution_manifest_sha256": _sha_of_file(E.EXECUTION_MANIFEST),
                      "target_ledger_sha256": _sha_of_file(E.LEDGER),
                      "evaluator_sha256": _sha_of_file("scripts/evaluate_h9_xr1.py"),
                      "driver_sha256": _sha_of_file(__file__),
                      "analysis_sha256": _sha_of_file("src/silent_teacher/h9_xr1_analysis.py"),
                      "block_index": X.BLOCK_INDEX, "terminal_step": 10000}

            def syn_identity(rep, arm):
                return ST.record_identity(checkpoint_sha256=f"{rep:064d}"[-64:], init_sha256="i" * 64,
                                          context_donor_sha256="x" * 64, replicate=rep, arm=arm,
                                          **common)

            def syn_payload(rep, arm):
                base = 0.60 + 0.0005 * rep + (0.0002 if arm == "pos" else 0.0)
                per_env = {e: {"value_nmse": base, "context": {"t63": base, "t127": base,
                                                               "pooled": base, "delta_cosine": 0.0},
                               "functional": {}} for e in ("c4", "pes2o", "se")}
                return {"schema": ST.SCHEMA, "identity": syn_identity(rep, arm),
                        "metrics": {"R_P": base, "C_P": base, "environments": per_env,
                                    "own_q_terminal_dev_nmse": base},
                        "complete": True}

            # phase 1: persist the first 7, then simulate an interruption
            first = [(1, "nat"), (1, "pos"), (2, "nat"), (2, "pos"), (3, "nat"), (3, "pos"), (4, "nat")]
            for rep, arm in first:
                ST.atomic_write_json(ST.student_path(tmp, rep, arm), syn_payload(rep, arm))
            before = {f"{r}_{a}": ST.sha_of_file(ST.student_path(tmp, r, a)) for r, a in first}
            # restart: the durable records must load and be reused
            reused = [f"{r}_{a}" for r, a in first
                      if ST.load_valid_record(ST.student_path(tmp, r, a), syn_identity(r, a))]
            checks["restart_reuses_durable_records"] = len(reused) == 7
            checks["restart_preserves_record_bytes"] = before == {
                f"{r}_{a}": ST.sha_of_file(ST.student_path(tmp, r, a)) for r, a in first}
            # phase 2: complete all 48, consolidate, validate, analyse
            for rep in X.REPLICATE_IDS:
                for arm in ("nat", "pos"):
                    path = ST.student_path(tmp, rep, arm)
                    if not path.exists():
                        ST.atomic_write_json(path, syn_payload(rep, arm))
            consolidated = ST.consolidate(tmp, EXPECTED_EXECUTION_TAG, {})
            path = tmp / "per_student.json"
            ST.atomic_write_json(path, consolidated)
            reopened = json.loads(path.read_text(encoding="utf-8"))
            _validate_persisted(reopened, ledger_syn)
            checks["consolidated_artifact_validates"] = True
            checks["consolidated_has_48_durable_records"] = len(reopened["durable_records"]) == 48
            out = analyse(reopened)
            checks["analysis_runs_from_the_persisted_artifact_only"] = (
                out["statistics"]["value"]["m"] == 24 and sorted(out["statistics"]) == ["context", "value"])
            checks["no_target_read_and_no_model_evaluated"] = True
            print(json.dumps({"synthetic_e2e": True, "checks": checks,
                              "records_reused_after_restart": len(reused),
                              "terminal_on_synthetic_metrics": out["terminal"]["state"]}, indent=2))
            ok = all(v for v in checks.values() if isinstance(v, bool))
            print(json.dumps({"all_checks_passed": ok}))
            return 0 if ok else 3
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    if args.validate_only:
        # synthetic metrics with the exact persisted shape; no checkpoint, no target, no outcome
        synthetic = {"students": {}}
        for replicate in X.REPLICATE_IDS:
            arms = {}
            for index, arm in enumerate(("nat", "pos")):
                base = 0.50 + 0.001 * replicate + 0.002 * index
                per_environment = {}
                for environment in A.ENVIRONMENTS:
                    per_environment[environment] = {
                        "value_nmse": base,
                        "context": {"pooled": base, "t63": base, "t127": base,
                                    "delta_cosine": 0.0},
                        "functional": {"kl_mean_over_tokens": 0.0, "top1_agreement": 1.0,
                                       "top5_set_overlap": 1.0, "nll_replaced": 1.0,
                                       "nll_teacher": 1.0, "delta_nll": 0.0}}
                arms[arm] = {"R_P": base, "C_P": base, "per_environment": per_environment,
                             "own_q_terminal_dev_nmse": base, "terminal_step": 10000}
            synthetic["students"][str(replicate)] = {"replicate_id": replicate, "arms": arms}
        out = analyse(synthetic)
        print(json.dumps({"validate_only": True, "terminal": out["terminal"]["state"],
                          "endpoint_profile": out["terminal"].get("endpoint_profile"),
                          "m": out["statistics"]["value"]["m"],
                          "secondary_axes": sorted(out["secondary"]),
                          "environments": sorted(out["per_environment"]),
                          "environment_set_ok": sorted(out["per_environment"]) ==
                                               ["c4", "pes2o", "se"],
                          "endpoint_set": sorted(out["statistics"]),
                          "endpoint_set_ok": sorted(out["statistics"]) == ["context", "value"],
                          "m_is_24": out["statistics"]["value"]["m"] == 24}, indent=2))
        print("SYNTHETIC VALIDATION ONLY — no checkpoint, no target, no scientific outcome read")
        return 0

    # ---- explicit execution identity, before anything else
    if args.dry_run_pre_freeze:
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
            return 3
        gate = {"passed": True, "identity_mode": "PRE_FREEZE_CANDIDATE_TREE", "authoritative": False,
                "covered_files": len(closure["execution_code"]),
                "expected_tag_commit": subprocess.run(
                    ["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()}
        print(f"[gate] PRE-FREEZE candidate tree verified against the closure "
              f"({gate['covered_files']} files, HEAD {gate['expected_tag_commit'][:12]})", flush=True)
    else:
        try:
            gate = freeze_gate(expected_tag=EXPECTED_EXECUTION_TAG)
        except FreezeGateHardStop as error:
            print(f"H9*-XR1 target evaluation REFUSED: {error}")
            return 2
    print(f"[gate] frozen scientific code verified: {gate['covered_files']} files, "
          f"tag commit {gate['expected_tag_commit']}", flush=True)

    # ---- frozen eligibility barrier
    barrier = E.evaluation_barrier(root / "replicates")
    assert barrier["eligible"], (
        f"target evaluation REFUSED: {barrier['resolved_students']}/{barrier['expected_students']} "
        f"Students are in a resolved evaluation-eligible state. Blocking: {barrier['blocking'][:5]} "
        f"Missing: {barrier['missing'][:5]}")
    print(f"[barrier] eligible: {barrier['resolved_students']}/{barrier['expected_students']} "
          f"resolved Students, statuses {barrier['status_counts']}", flush=True)

    # ---- frozen identities re-verified immediately before any outcome is produced
    ledger = json.loads(E.LEDGER.read_text(encoding="utf-8"))
    rng_manifest = json.loads(E.RNG_MANIFEST.read_text(encoding="utf-8"))
    identity = {"execution_tag": EXPECTED_EXECUTION_TAG,
                "execution_tag_commit": gate["expected_tag_commit"],
                "execution_manifest_sha256": _sha_of_file(E.EXECUTION_MANIFEST),
                "target_ledger_sha256": _sha_of_file(E.LEDGER),
                "rng_manifest_sha256": _sha_of_file(E.RNG_MANIFEST),
                "analysis_code_sha256": _sha_of_file("src/silent_teacher/h9_xr1_analysis.py"),
                "evaluator_code_sha256": _sha_of_file("scripts/evaluate_h9_xr1.py"),
                "driver_code_sha256": _sha_of_file(__file__),
                "checkpoints": {}, "arrays": {}}

    if args.dry_run_pre_freeze:
        # validate the pre-target wiring: the ledger's identity, and the checkpoint bindings for
        # every one of the 48 frozen Students. NO target array is opened and no metric is computed.
        checks = {"candidate_identity_accepted": gate["passed"],
                  "eligibility_barrier_eligible": barrier["eligible"],
                  "barrier_expected_48_students": barrier["expected_students"] == 48,
                  "barrier_resolved_48_students": barrier["resolved_students"] == 48,
                  "targets_not_read": True}
        bound, mismatched = 0, []
        for replicate in X.REPLICATE_IDS:
            folder = root / "replicates" / f"replicate_{replicate:02d}"
            record = json.loads((folder / "replicate.json").read_text(encoding="utf-8"))
            for arm in ("nat", "pos"):
                checkpoint = folder / f"student_{arm}.pt"
                payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
                ok = (payload["steps"] == 10000 and payload["arm"] == arm
                      and payload["replicate_id"] == replicate
                      and payload["init_hash"] == record["shared"]["init_sha256"]
                      and payload["block_index"] == X.BLOCK_INDEX
                      and _sha_of_file(checkpoint) == record["arms"][arm]["checkpoint_sha256"])
                bound += 1 if ok else 0
                if not ok:
                    mismatched.append(f"{replicate:02d}_{arm}")
        checks["all_48_checkpoints_bound_to_training_records"] = bound == 48 and not mismatched
        checks["checkpoint_binding_count"] = bound
        checks["epoch"] = "pre-target-read"
        print(json.dumps({"dry_run": "evaluation_pre_target_wiring", "checks": checks,
                          "mismatched": mismatched,
                          "stopped_before": "reading any target content"},
                         indent=2))
        print(json.dumps({"all_checks_passed": all(v for k, v in checks.items()
                                                   if isinstance(v, bool))}))
        return 0 if all(v for k, v in checks.items() if isinstance(v, bool)) else 3

    setup()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, block = E.build_teacher(device)
    tokens, array_identity = E.load_frozen_targets(root / "targets", ledger)
    identity["arrays"] = array_identity
    print(f"[targets] frozen arrays verified: {sorted(tokens)} (3072 windows)", flush=True)

    pools = {}
    for environment in ledger["environment_order"]:
        hidden = prefix_states(model, tokens[environment], device, block_index=X.BLOCK_INDEX)
        pools[environment] = {"hidden": hidden, "labels": teacher_update(block, hidden)}
        print(f"[eval] cached {environment}: H {tuple(hidden.shape)}", flush=True)

    # ---- one pass over the 48 frozen Students, each made durable before the next begins
    identity_common = {"execution_tag": EXPECTED_EXECUTION_TAG,
                       "execution_manifest_sha256": identity["execution_manifest_sha256"],
                       "target_ledger_sha256": identity["target_ledger_sha256"],
                       "evaluator_sha256": identity["evaluator_code_sha256"],
                       "driver_sha256": identity["driver_code_sha256"],
                       "analysis_sha256": identity["analysis_code_sha256"],
                       "block_index": X.BLOCK_INDEX, "terminal_step": 10000}
    reused, evaluated = [], []
    started = time.time()
    for replicate in X.REPLICATE_IDS:
        folder = root / "replicates" / f"replicate_{replicate:02d}"
        record = json.loads((folder / "replicate.json").read_text(encoding="utf-8"))
        assert record["replicate_id"] == replicate and record["status"] == "PAIR_COMPLETE"
        streams = Id2Streams(E.formal_seeds(rng_manifest, replicate),
                             label=f"replicate-{replicate:02d}")
        donor = streams.context_donor_derangement(1024)
        donor_sha = hashlib.sha256(donor.numpy().tobytes()).hexdigest()
        assert donor_sha == record["shared"]["context_donor_sha256"], \
            f"replicate {replicate}: context donor does not reproduce the training-time donor"
        for arm in ("nat", "pos"):
            checkpoint = folder / f"student_{arm}.pt"
            checkpoint_sha = _sha_of_file(checkpoint)
            assert checkpoint_sha == record["arms"][arm]["checkpoint_sha256"], \
                f"replicate {replicate} {arm}: checkpoint does not match the training record"
            identity["checkpoints"][f"{replicate:02d}_{arm}"] = {
                "path": checkpoint.as_posix(), "sha256": checkpoint_sha, "matches_record": True}
            expected = S.record_identity(
                checkpoint_sha256=checkpoint_sha, init_sha256=record["shared"]["init_sha256"],
                context_donor_sha256=donor_sha, replicate=replicate, arm=arm, **identity_common)
            # an existing VALID record is reused and never recomputed; an existing INVALID record
            # raises inside load_valid_record rather than being silently recomputed or accepted
            if S.load_valid_record(S.student_path(root, replicate, arm), expected) is not None:
                reused.append(f"{replicate:02d}_{arm}")
                print(f"[eval] {replicate:02d}_{arm} reused a durable record", flush=True)
                continue
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
            assert payload["steps"] == 10000 and payload["arm"] == arm
            assert payload["replicate_id"] == replicate
            assert payload["init_hash"] == record["shared"]["init_sha256"]
            assert payload["block_index"] == X.BLOCK_INDEX
            student = FullBlockStudent()
            student.load_state_dict(payload["model_state"])
            student = student.eval().to(device)
            per_environment = {}
            for environment in ledger["environment_order"]:
                value = E.value_nmse(student, pools[environment]["hidden"],
                                     pools[environment]["labels"], device)
                context = full_block_context_evaluation(
                    lambda h: teacher_update(block, h),
                    {"student": lambda h, s=student: s(h.to(device)).cpu()},
                    pools[environment]["hidden"], donor, queries=(E.CUT_QUERY, E.LAST_QUERY),
                    microbatch=8)
                metrics = context["objects"]["student"]
                functional = E.functional_scores(model, student, tokens[environment], device)
                per_environment[environment] = {"value_nmse": value,
                                                "context": {"pooled": metrics["nmse"],
                                                            "t63": metrics[f"t{E.CUT_QUERY}"]["nmse"],
                                                            "t127": metrics[f"t{E.LAST_QUERY}"]["nmse"],
                                                            "delta_cosine": metrics["delta_cosine"]},
                                                "functional": functional}
            # the frozen EQUAL-ENVIRONMENT Value endpoint and equal-query/equal-environment Context
            # endpoint, both computed from the per-environment values only — never a pooled global
            # ratio over all examples
            assert set(per_environment) == set(S.REQUIRED_ENVIRONMENTS), sorted(per_environment)
            # the frozen equal-environment Value endpoint, the equal-query/equal-environment
            # Context endpoint, and the functional secondaries — all through the audited assembly
            metrics_payload = E.assemble_student_metrics(per_environment,
                                                         record["own_q"][arm]["raw_error"])
            payload_out = {"schema": S.SCHEMA,
                           "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                           "identity": expected, "metrics": metrics_payload, "complete": True}
            # validate BEFORE persisting, so an invalid metric is never made durable
            S.validate_record(payload_out, expected)
            path = S.atomic_write_json(S.student_path(root, replicate, arm), payload_out)
            evaluated.append(f"{replicate:02d}_{arm}")
            print(f"[eval] {replicate:02d}_{arm} persisted sha256 {S.sha_of_file(path)[:16]}... "
                  f"({time.time() - started:.0f}s)", flush=True)
            del student
            torch.cuda.empty_cache()
        print(f"[eval] replicate {replicate:02d} complete", flush=True)
    print(f"[eval] evaluated {len(evaluated)}, reused {len(reused)} durable records", flush=True)

    # ---- consolidate the 48 durable records into the authoritative per-Student artifact
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    per_student = S.consolidate(root, EXPECTED_EXECUTION_TAG,
                               {"checkpoints": identity["checkpoints"],
                                "rng_manifest_sha256": identity["rng_manifest_sha256"]})
    per_student["barrier"] = barrier
    per_student["reused_records"] = reused
    per_student["evaluated_records"] = evaluated
    per_student["identity"] = identity

    # ---- PERSISTENCE BOUNDARY: write, close, re-open, validate, hash — before any aggregate
    S.atomic_write_json(per_student_path, per_student)
    reopened = json.loads(per_student_path.read_text(encoding="utf-8"))
    _validate_persisted(reopened, ledger)
    per_student_sha = _sha_of_file(per_student_path)
    print(f"[persist] per-Student metrics frozen: {per_student_path} sha256 {per_student_sha}",
          flush=True)

    # ---- aggregate analysis: reads the PERSISTED artifact only, calls no model, computes no metric
    analysis = analyse(reopened)
    analysis["identity"] = identity
    analysis["per_student_sha256"] = per_student_sha
    results = {"schema": "latentbridge.h9_xr1.results.v1",
               "creation_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               "execution_tag": EXPECTED_EXECUTION_TAG,
               "per_student": {"path": per_student_path.as_posix(), "sha256": per_student_sha},
               "target_ledger_sha256": identity["target_ledger_sha256"],
               "analysis_read_only_the_persisted_artifact": True,
               "analysis": analysis}
    results_path.write_text(json.dumps(results, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[results] {results_path} sha256 {_sha_of_file(results_path)}")
    print(json.dumps({"terminal": analysis["terminal"]["state"],
                      "endpoint_profile": analysis["terminal"].get("endpoint_profile"),
                      "m": analysis["statistics"]["value"]["m"]}, indent=2))
    return 0


def _validate_persisted(payload, ledger):
    """Every precondition the aggregate analysis depends on, checked on the RE-OPENED file."""
    students = payload["students"]
    assert len(students) == X.REPLICATES == 24, len(students)
    arms = [arm for entry in students.values() for arm in entry["arms"]]
    assert len(arms) == X.PLANNED_STUDENTS == 48, len(arms)
    assert sum(1 for arm in arms if arm == "nat") == 24
    assert sum(1 for arm in arms if arm == "pos") == 24
    assert sorted(int(r) for r in students) == list(X.REPLICATE_IDS)
    for replicate, entry in students.items():
        assert set(entry["arms"]) == {"nat", "pos"}
        for arm, values in entry["arms"].items():
            assert set(values["per_environment"]) == set(S.REQUIRED_ENVIRONMENTS), \
                sorted(values["per_environment"])
            assert values["terminal_step"] == 10000
            for environment, metrics in values["per_environment"].items():
                assert metrics["value_nmse"] == metrics["value_nmse"]
                for key in ("t63", "t127", "pooled"):
                    assert metrics["context"][key] == metrics["context"][key]
            assert values["R_P"] == values["R_P"] and values["C_P"] == values["C_P"]
    assert payload["identity"]["execution_tag"] == EXPECTED_EXECUTION_TAG
    assert len(payload["durable_records"]) == 48
    return True


def analyse(payload):
    """Frozen statistics over the 24 paired observations, from the persisted metrics only."""
    order = ("c4", "pes2o", "se")
    N, Q = {}, {}
    for endpoint in A.ENDPOINTS:
        N[endpoint], Q[endpoint] = [], []
    for replicate in X.REPLICATE_IDS:
        entry = payload["students"][str(replicate)]
        for endpoint, key in (("value", "R_P"), ("context", "C_P")):
            N[endpoint].append(entry["arms"]["nat"][key])
            Q[endpoint].append(entry["arms"]["pos"][key])
    quantities = {e: A.paired_quantities(N[e], Q[e]) for e in A.ENDPOINTS}
    guard = {e: A.native_qualification(N[e], quantities[e]["G"]) for e in A.ENDPOINTS}
    loss = {e: A.material_loss(quantities[e]["Z"]) for e in A.ENDPOINTS}
    keep = {e: A.preservation(quantities[e]["Z"]) for e in A.ENDPOINTS}
    terminal = A.classify(native=N, pos=Q)

    statistics = {}
    for endpoint in A.ENDPOINTS:
        d, z, g = quantities[endpoint]["D"], quantities[endpoint]["Z"], quantities[endpoint]["G"]
        statistics[endpoint] = {
            "mean_N": guard[endpoint]["N"]["mean"], "sd_N": guard[endpoint]["N"]["sd"],
            "mean_G": guard[endpoint]["G"]["mean"], "sd_G": guard[endpoint]["G"]["sd"],
            "qualification_bound_L95_G": guard[endpoint]["L95_G"],
            "qualification_bound_threshold": A.NATIVE_MIN_RECOVERY,
            "qualification_passes": guard[endpoint]["passes"],
            "mean_Q": A.statistics_of(Q[endpoint])["mean"],
            "mean_D": A.statistics_of(d)["mean"], "sd_D": A.statistics_of(d)["sd"],
            "count_D_positive": sum(1 for v in d if v > 0), "m": A.M,
            "mean_Z": A.statistics_of(z)["mean"], "sd_Z": A.statistics_of(z)["sd"],
            "se_Z": A.statistics_of(z)["se"],
            "U95_Z": keep[endpoint]["U95"], "L97.5_Z": loss[endpoint]["L95"],
            "theta_ratio_of_means": (A.statistics_of(d)["mean"] / guard[endpoint]["G"]["mean"]
                                     if guard[endpoint]["G"]["mean"] != 0 else None),
            "material_loss_passes": loss[endpoint]["passes"],
            "preservation_passes": keep[endpoint]["passes"]}
    per_environment = {}
    for environment in order:
        per_environment[environment] = {}
        for endpoint in ("value", "context"):
            nat, pos = [], []
            for replicate in X.REPLICATE_IDS:
                arms = payload["students"][str(replicate)]["arms"]

                def env_value(arm, env=environment, ep=endpoint):
                    metrics = arms[arm]["per_environment"][env]
                    if ep == "value":
                        return metrics["value_nmse"]
                    return (metrics["context"]["t63"] + metrics["context"]["t127"]) / 2

                nat.append(env_value("nat"))
                pos.append(env_value("pos"))
            per_environment[environment][endpoint] = {
                "mean_native": A.statistics_of(nat)["mean"], "mean_qpos": A.statistics_of(pos)["mean"],
                "mean_D": A.statistics_of([p - n for n, p in zip(nat, pos)])["mean"]}
    secondary = {}
    for label, key in (("context_t63", "t63"), ("context_t127", "t127")):
        nat, pos = [], []
        for replicate in X.REPLICATE_IDS:
            entry = payload["students"][str(replicate)]["arms"]
            nat.append(sum(entry["nat"]["per_environment"][n]["context"][key] for n in order) / 3)
            pos.append(sum(entry["pos"]["per_environment"][n]["context"][key] for n in order) / 3)
        secondary[label] = {"statistics": A.statistics_of([p - n for n, p in zip(nat, pos)])}
    return {"constants": A.constants_identity()["computed"], "statistics": statistics,
            "paired": quantities, "guard": guard, "material_loss": loss, "preservation": keep,
            "terminal": terminal, "per_environment": per_environment, "secondary": secondary,
            "own_q": {arm: [payload["students"][str(r)]["arms"][arm]["own_q_terminal_dev_nmse"]
                            for r in X.REPLICATE_IDS] for arm in ("nat", "pos")}}


if __name__ == "__main__":
    sys.exit(main())
