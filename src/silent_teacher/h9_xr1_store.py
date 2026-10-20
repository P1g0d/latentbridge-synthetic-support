"""H9*-XR1 confirmatory per-Student persistence and recovery.

Persistence/recovery ONLY: this module contains no endpoint mathematics. Its job is to make a produced
confirmatory metric durable before the evaluation advances, and to make a restart reuse rather than
recompute an already-valid metric.

Guarantees:
  * each Student's metric is written ATOMICALLY (temp file in the same directory, fsync, os.replace),
    so a crash can never leave a half-written record in place of a valid one;
  * a valid existing record is REUSED and never overwritten or recomputed;
  * a malformed, partial or identity-mismatched record FAILS CLOSED rather than being read as a metric;
  * consolidation into the authoritative per_student.json is deterministic and validates 24 Native /
    24 Q_pos across exactly 24 paired replicate ids before it is written.
"""
import hashlib
import json
import os
import tempfile
from pathlib import Path

SCHEMA = "latentbridge.h9_xr1.raw_student.v1"
REQUIRED_ENVIRONMENTS = ("c4", "pes2o", "se")
ARMS = ("nat", "pos")
#: the Student-level endpoints that must be present and finite at the metrics level
REQUIRED_ENDPOINT_KEYS = ("R_P", "C_P")
#: the per-environment components that must be present and finite inside every environment
REQUIRED_ENVIRONMENT_VALUE_KEYS = ("value_nmse",)
REQUIRED_ENVIRONMENT_CONTEXT_KEYS = ("t63", "t127", "pooled")


class PersistenceHardStop(RuntimeError):
    """Raised when a persisted record cannot be trusted. Never downgraded to a warning."""


def sha_of_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def student_path(root, replicate, arm):
    return Path(root) / "raw_students" / f"replicate_{replicate:02d}_{arm}.json"


def atomic_write_json(path, payload, fsync=True):
    """Write `payload` atomically: same-directory temp file, flush+fsync, then os.replace."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=str(path.parent),
                                         prefix=path.name + ".", suffix=".tmp", delete=False)
    temp_name = handle.name
    try:
        with handle:
            json.dump(payload, handle, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            if fsync:
                os.fsync(handle.fileno())
        os.replace(temp_name, path)
        if fsync:
            # Durably record the rename where the platform supports it. Opening a directory for
            # fsync is not permitted on Windows, so that step is best-effort there; the atomicity
            # guarantee rests on fsync(file) + os.replace, which is atomic on NTFS.
            try:
                directory = os.open(str(path.parent), os.O_RDONLY)
            except OSError:
                directory = None
            if directory is not None:
                try:
                    os.fsync(directory)
                finally:
                    os.close(directory)
    except BaseException:
        try:
            os.unlink(temp_name)
        except OSError:
            pass
        raise
    return path


def _finite(value):
    return isinstance(value, (int, float)) and value == value and value not in (float("inf"),
                                                                                float("-inf"))


def record_identity(execution_tag, execution_manifest_sha256, target_ledger_sha256,
                    checkpoint_sha256, init_sha256, context_donor_sha256, replicate, arm,
                    terminal_step, block_index, evaluator_sha256, driver_sha256,
                    analysis_sha256):
    """The identity fields a persisted record is bound to. Any drift makes the record invalid."""
    return {"execution_tag": execution_tag,
            "execution_manifest_sha256": execution_manifest_sha256,
            "target_ledger_sha256": target_ledger_sha256,
            "checkpoint_sha256": checkpoint_sha256,
            "init_sha256": init_sha256,
            "context_donor_sha256": context_donor_sha256,
            "replicate_id": replicate, "arm": arm,
            "terminal_step": terminal_step, "block_index": block_index,
            "evaluator_sha256": evaluator_sha256, "driver_sha256": driver_sha256,
            "analysis_sha256": analysis_sha256}


def validate_record(payload, expected_identity):
    """Fail closed unless the record is complete, finite, and bound to the expected identities."""
    if not isinstance(payload, dict):
        raise PersistenceHardStop("record is not an object")
    if payload.get("schema") != SCHEMA:
        raise PersistenceHardStop(f"record schema {payload.get('schema')!r} != {SCHEMA!r}")
    if payload.get("complete") is not True:
        raise PersistenceHardStop("record is marked incomplete")
    recorded = payload.get("identity")
    if not isinstance(recorded, dict):
        raise PersistenceHardStop("record has no identity block")
    for key, want in expected_identity.items():
        got = recorded.get(key)
        if got != want:
            raise PersistenceHardStop(f"identity mismatch on {key!r}: {got!r} != {want!r}")
    metrics = payload.get("metrics")
    if not isinstance(metrics, dict):
        raise PersistenceHardStop("record has no metrics block")
    if set(metrics.get("environments", {})) != set(REQUIRED_ENVIRONMENTS):
        raise PersistenceHardStop(f"environment set {sorted(metrics.get('environments', {}))} != "
                                  f"{list(REQUIRED_ENVIRONMENTS)}")
    for environment, entry in metrics["environments"].items():
        if not isinstance(entry, dict):
            raise PersistenceHardStop(f"{environment}: metrics entry is not an object")
        for key in REQUIRED_ENVIRONMENT_VALUE_KEYS:
            if not _finite(entry.get(key)):
                raise PersistenceHardStop(f"{environment}: {key} is not finite")
        context = entry.get("context") or {}
        for key in REQUIRED_ENVIRONMENT_CONTEXT_KEYS:
            if not _finite(context.get(key)):
                raise PersistenceHardStop(f"{environment}: context[{key!r}] is not finite")
    for key in REQUIRED_ENDPOINT_KEYS:
        if not _finite(metrics.get(key)):
            raise PersistenceHardStop(f"{key} is not finite")
    return True


def load_valid_record(path, expected_identity):
    """Return the record when it exists and validates, None when it does not exist.

    A record that exists but does not validate raises: an unusable record must never be treated as
    absent (that would silently recompute) nor as a metric (that would silently accept garbage).
    """
    path = Path(path)
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    validate_record(payload, expected_identity)
    return payload


def consolidate(root, execution_tag, per_replicate_identity, expected_students=48,
                expected_pairs=24):
    """Build the authoritative per-Student payload from the durable records, deterministically."""
    raw_dir = Path(root) / "raw_students"
    records = {}
    for path in sorted(raw_dir.glob("replicate_*_*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        replicate = int(payload["identity"]["replicate_id"])
        arm = payload["identity"]["arm"]
        key = f"{replicate}_{arm}"
        if key in records:
            raise PersistenceHardStop(f"duplicate Student record for {key}")
        records[key] = {"path": path.as_posix(), "sha256": sha_of_file(path), "payload": payload}
    if len(records) != expected_students:
        raise PersistenceHardStop(f"{len(records)} durable Student records, expected "
                                  f"{expected_students}")
    natives = [k for k in records if k.endswith("_nat")]
    qposs = [k for k in records if k.endswith("_pos")]
    if len(natives) != expected_pairs or len(qposs) != expected_pairs:
        raise PersistenceHardStop(f"arm counts {len(natives)}/{len(qposs)} != "
                                  f"{expected_pairs}/{expected_pairs}")
    students = {}
    for replicate in sorted({int(k.split("_")[0]) for k in records}):
        entry = {"replicate_id": replicate, "arms": {}, "raw_records": {}}
        for arm in ARMS:
            key = f"{replicate}_{arm}"
            payload = records[key]["payload"]
            entry["arms"][arm] = {"R_P": payload["metrics"]["R_P"],
                                  "C_P": payload["metrics"]["C_P"],
                                  "per_environment": payload["metrics"]["environments"],
                                  "own_q_terminal_dev_nmse": payload["metrics"].get(
                                      "own_q_terminal_dev_nmse"),
                                  "terminal_step": payload["identity"]["terminal_step"]}
            entry["raw_records"][arm] = {"path": records[key]["path"],
                                         "sha256": records[key]["sha256"]}
        entry["context_donor_sha256"] = records[f"{replicate}_nat"]["payload"]["identity"][
            "context_donor_sha256"]
        entry["init_sha256"] = records[f"{replicate}_nat"]["payload"]["identity"]["init_sha256"]
        students[str(replicate)] = entry
    if sorted(int(r) for r in students) != list(range(1, expected_pairs + 1)):
        raise PersistenceHardStop("replicate ids are not exactly 1..24")
    return {"schema": "latentbridge.h9_xr1.per_student.v1",
            "identity": {"execution_tag": execution_tag,
                         "per_replicate": per_replicate_identity},
            "endpoints": {
                "value": "R_P = equal-environment mean of NMSE_k, "
                         "NMSE_k = sum||S(H)-F9(H)||^2 / sum||F9(H)||^2",
                "context": "C_k = (ContextNMSE_k,63 + ContextNMSE_k,127)/2; "
                           "C_P = equal-environment mean of C_k"},
            "students": students,
            "durable_records": {k: records[k]["sha256"] for k in sorted(records)},
            "students_evaluated_once": True,
            "consolidated_from": "raw_students/ (one atomic record per Student)"}
