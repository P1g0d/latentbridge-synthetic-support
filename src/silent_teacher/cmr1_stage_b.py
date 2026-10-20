"""CMR1 Stage-B metadata-only execution-control transaction.

EXECUTION-CRITICAL / GOVERNANCE-CONTROL-PLANE CRITICAL.
No target reader, materializer, tensor construction or training API is imported.
JSON filenames here are operational owners, not scientific campaign scheduling rules.
Like the existing authorization boundary, this is auditability, not a sandbox against
an operator who changes committed code. Dynamic/CLI dependencies require exact review.
"""
from __future__ import annotations

import ast
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import weakref

from . import cmr1_authorization as auth, cmr1_rng as rng

SCOPE = auth.SCOPE_STAGE_B_REALIZATION
SCHEMA = "latentbridge.cmr1.stage_b_execution_control.v1"
PREPARED = "PREPARED_REVIEWABLE"
FROZEN = "FROZEN_STAGE_B_EXECUTION_CONTROL"
OUTPUT_DIR = "manifests/cmr1_stage_b"
ADMIN_REBUILD_DIR = "manifests/cmr1_stage_b_admin_rebuild_1"
TEACHER_METADATA_REBUILD_DIR = "manifests/cmr1_stage_b_teacher_metadata_rebuild"
MATERIALIZER_INVENTORY_REBUILD_DIR = "manifests/cmr1_stage_b_materializer_inventory_rebuild"
CUDA_BOOTSTRAP_REBUILD_DIR = "manifests/cmr1_stage_b_cuda_bootstrap_rebuild"
RECOVERY_REBUILD_DIR = "manifests/cmr1_stage_b_infrastructure_recovery_closure"
RECOVERY_FINAL_REBUILD_DIR = "manifests/cmr1_stage_b_infrastructure_recovery_source_binding"
RECOVERY_ATTESTED_REBUILD_DIR = "manifests/cmr1_stage_b_infrastructure_recovery_attested"
VERIFIED_LOCAL_ADMIN_DIR = "manifests/cmr1_stage_b_verified_local_admin_refresh"
FORBIDDEN = ("target_body", "target_label", "target_output", "target_metric",
             "target_materialization", "realized_target_ledger", "campaign_training")
INVENTORY_COVERAGE = ("historical_unit_union", "historical_spent_windows",
                      "unrecoverable_identity_adjudication", "exposure_incidents")
RETRIEVAL_COVERAGE = ("teacher_metadata", "c4", "pes2o", "stackexchange")
CATEGORIES = ("training_runtime", "campaign_rng", "materializer", "exclusion",
              "configuration", "environment", "numerical_mode", "governance")
QUALIFICATION_PATHS = {
    "manifest": "manifests/cmr1_qualification_identity_manifest.json",
    "tolerance": "manifests/cmr1_qualification_tolerances.json",
    "attempt": "reports/cmr1_formal_qualification_attempt.json",
    "result": "reports/cmr1_formal_qualification_result.json",
    "acceptance": "reports/cmr1_formal_qualification_integrity_verification.json",
}
STATIC_PATHS = {
    "protocol": "protocols/cmr1_pythia_cross_model_replication_v0_3.md",
    "b11": "reports/cmr1_provenance_freeze_contract.md",
    "rng": "manifests/cmr1_rng_role_schema.json",
    "failure": "manifests/cmr1_failure_replay_policy.json",
    "access": "manifests/cmr1_real_support_access_policy.json",
    "source": "manifests/cmr1_target_source_contract.json",
    "exclusion": "manifests/cmr1_exposure_exclusion_contract.json",
}
PRODUCTION_TARGET_ENTRYPOINT = "scripts/run_cmr1_target_materializer.py"
NUMERICAL_MODE = {
    "float32_matmul_precision": "highest", "cuda_matmul_allow_tf32": False,
    "cudnn_allow_tf32": False, "cudnn_deterministic": True, "cudnn_benchmark": False,
    "deterministic_algorithms": True, "determinism_warn_only": False,
    "cpu_autocast_active": False, "cuda_autocast_active": False,
    "cublas_workspace_config": ":4096:8", "numerical_mode": "ASSERTED",
}
APPROVED_BASE = "8cf1ea73df2f08535abda2d48741b8f7091e0645"
_REPO = Path(__file__).resolve().parents[2]
_CONTEXTS = weakref.WeakKeyDictionary()
_FROZEN = weakref.WeakKeyDictionary()


class StageBError(RuntimeError):
    """HARD STOP; never reseed or automatically repair a changed scientific identity."""


class StageBContext:
    __slots__ = ("__weakref__",)


class FrozenStageB:
    __slots__ = ("__weakref__",)


def canonical(document):
    return json.dumps(document, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("utf-8")


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def _stop(message):
    raise StageBError("HARD STOP - " + message)


def _path(path):
    if (not isinstance(path, str) or not path or "\\" in path or ":" in path or
            path.startswith("/") or any(x in ("", ".", "..") for x in path.split("/"))):
        _stop("invalid repository-relative path")
    return path


def _git(repo, *args):
    proc = subprocess.run(["git", "--no-replace-objects", *args], cwd=repo,
                          capture_output=True)
    if proc.returncode:
        _stop("committed source missing or git identity unavailable")
    return proc.stdout


def _tree_blobs(repo, revision, paths):
    """Resolve canonical paths to blob IDs in one explicit committed tree."""
    paths = {_path(path) for path in paths}
    commit = _git(repo, "rev-parse", revision + "^{commit}").decode().strip()
    if not re.fullmatch(r"[0-9a-f]{40,64}", commit):
        _stop("invalid committed source identity")
    records = _git(repo, "ls-tree", "-r", "-z", commit, "--", *sorted(paths))
    found = {}
    for record in records.split(b"\0"):
        if not record:
            continue
        try:
            header, raw_path = record.split(b"\t", 1)
            mode, kind, object_id = header.split(b" ")
            path = raw_path.decode("utf-8")
        except (ValueError, UnicodeError) as exc:
            raise StageBError("HARD STOP - malformed committed tree entry") from exc
        if path in paths:
            if mode not in (b"100644", b"100755") or kind != b"blob":
                _stop("source must be a regular committed blob")
            found[path] = object_id.decode("ascii")
    if set(found) != paths:
        _stop("committed source path/object missing")
    return found


def _object_bytes(repo, object_id):
    if not isinstance(object_id, str) or not re.fullmatch(r"[0-9a-f]{40,64}", object_id):
        _stop("invalid immutable object identity")
    return _git(repo, "cat-file", "blob", object_id)


def _blob(repo, revision, path):
    return _object_bytes(repo, _tree_blobs(repo, revision, [path])[path])


def _store_object(repo, raw):
    proc = subprocess.run(["git", "hash-object", "-w", "--stdin"], cwd=repo,
                          input=raw, capture_output=True)
    if proc.returncode:
        _stop("immutable reservation object unavailable")
    object_id = proc.stdout.decode("ascii").strip()
    if _object_bytes(repo, object_id) != raw:
        _stop("immutable reservation object mismatch")
    return object_id


def _json(raw):
    try:
        value = json.loads(raw)
        if not isinstance(value, dict):
            _stop("expected JSON object")
        return value
    except (ValueError, TypeError) as exc:
        raise StageBError("HARD STOP - malformed committed JSON") from exc


def _keys(document, keys):
    if not isinstance(document, dict) or set(document) != set(keys):
        _stop("missing or unexpected contract fields")


def _bound(repo, revision, ref, *, path=None):
    _keys(ref, ("path", "sha256"))
    if path is not None and ref["path"] != path:
        _stop("noncanonical evidence reference")
    if not isinstance(ref["sha256"], str) or not re.fullmatch("[0-9a-f]{64}", ref["sha256"]):
        _stop("invalid evidence digest")
    raw = _blob(repo, revision, ref["path"])
    if digest(raw) != ref["sha256"]:
        _stop("committed evidence identity mismatch")
    return raw


def validate_campaign_records(records, namespace, root_seed, *, context):
    rng._campaign_entry(namespace, context)
    expected = {f"replicate-{r:02d}/{role}": rng.campaign_child_seed(
        root_seed, namespace, r, role, context=context)
        for r in range(1, 49) for role in rng.CAMPAIGN_ROLES}
    records = list(records)
    if len(records) != 432 or set(r.role_key for r in records) != set(expected):
        _stop("campaign requires exactly 48 x 9 canonical role records")
    result = rng.validate_within_domain(records, "campaign")
    if any(r.integer_seed != expected[r.role_key] for r in records):
        _stop("replacement seed or deterministic derivation mismatch")
    return result


def arm_role_reference(replicate, arm, role):
    """Paired identities by reference; Q_pos cannot draw independent donors/init/schedule."""
    if type(replicate) is not int or not 1 <= replicate <= 48 or arm not in ("nat", "pos"):
        _stop("invalid replicate/arm")
    if role not in rng.CAMPAIGN_ROLES or (arm == "nat" and role.startswith("permutation_")):
        _stop("role not consumed by this arm")
    return f"replicate-{replicate:02d}/{role}"


def _qualification(repo, revision, references):
    _keys(references, QUALIFICATION_PATHS)
    docs = {name: _json(_bound(repo, revision, references[name], path=path))
            for name, path in QUALIFICATION_PATHS.items()}
    manifest, tol, attempt, result, accepted = (docs[x] for x in QUALIFICATION_PATHS)
    if (manifest.get("state") != "FROZEN_PRE_EXPOSURE_IDENTITY_BARRIER" or
            manifest.get("review", {}).get("independent_review_performed") is not True):
        _stop("qualification manifest is not frozen/reviewed")
    md = digest(canonical({k: v for k, v in manifest.items() if k not in ("state", "review")}))
    if manifest["review"].get("reviewed_manifest_digest") != md:
        _stop("qualification review digest mismatch")
    if (tol.get("state") != "FROZEN_CALIBRATION_TOLERANCES" or tol.get("frozen") is not True or
            tol.get("retry_count") != 0 or tol.get("reviewed_manifest_digest") != md):
        _stop("qualification tolerances not bound/frozen")
    if (attempt.get("attempt_number") != 1 or attempt.get("retry_count") != 0 or
            result.get("retry_count") != 0 or result.get("terminal") != "PASS" or
            accepted.get("native_terminal") != "PASS" or
            accepted.get("pre_attempt_runtime_refusal", {}).get("formal_retry_count") != 0):
        _stop("qualification must be accepted PASS, attempt 1, retries 0")
    identity = attempt.get("attempt_identity")
    if not identity or result.get("attempt_identity") != identity or accepted.get("attempt_identity") != identity:
        _stop("qualification attempt/result/acceptance identity mismatch")
    for bindings in (attempt.get("input_bindings", {}), result.get("qualification_inputs", {})):
        if (bindings.get("manifest_sha256") != references["manifest"]["sha256"] or
                bindings.get("tolerance_sha256") != references["tolerance"]["sha256"] or
                bindings.get("reviewable_digest") != md):
            _stop("qualification input binding mismatch")
    for key in ("authorization_sha256", "authorization_commit", "execution_critical_commit"):
        if not attempt.get(key) or result.get(key) != attempt[key]:
            _stop("qualification execution identity mismatch")
    seeds = manifest.get("realized_seeds", {})
    records = [rng.RoleRecord("qualification", r, seeds.get("role:" + r))
               for r in rng.QUALIFICATION_ROLES]
    rng.validate_qualification_records(records)
    return records


def _closure(repo, revision, closure):
    _keys(closure, ("categories", "blob_sha256"))
    _keys(closure["categories"], CATEGORIES)
    paths = set()
    for category, members in closure["categories"].items():
        if not isinstance(members, list) or not members or len(members) != len(set(members)):
            _stop("empty/duplicate closure category " + category)
        paths.update(members)
    if paths != set(closure["blob_sha256"]):
        _stop("closure owner/category mismatch")
    core = {"src/silent_teacher/" + x + ".py" for x in
            ("cmr1_stage_b", "cmr1_authorization", "cmr1_rng", "cmr1_bootstrap", "cmr1_runtime")}
    if not core <= paths or not {"requirements-lock.txt", ".gitattributes"} <= paths:
        _stop("incomplete Stage-B control/runtime closure")
    tracked = set(_git(repo, "ls-tree", "-r", "--name-only", revision).decode().splitlines())
    raw_by_path = {}
    for path in paths:
        _path(path)
        if not (path.startswith(("src/", "scripts/", "configs/")) or
                path in ("requirements-lock.txt", ".gitattributes", "pyproject.toml")):
            _stop("closure path is not production code/configuration")
        raw_by_path[path] = _bound(repo, revision, {"path": path, "sha256": closure["blob_sha256"][path]})
    # Enforce static repository dependencies too. Dynamic imports/CLI/config dependencies
    # must be explicitly listed and independently reviewed as part of the candidate.
    for path, raw in raw_by_path.items():
        if not path.endswith(".py"):
            continue
        try:
            tree = ast.parse(raw.decode("utf-8-sig"))
        except (SyntaxError, UnicodeError) as exc:
            raise StageBError("HARD STOP - invalid closure source") from exc
        package = PurePosixPath(path).parent
        names = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names.extend("src/" + a.name.replace(".", "/") for a in node.names)
            elif isinstance(node, ast.ImportFrom):
                base = package
                if node.level:
                    for _ in range(node.level - 1):
                        base = base.parent
                    name = str(base) + ("/" + node.module.replace(".", "/") if node.module else "")
                else:
                    name = "src/" + (node.module or "").replace(".", "/")
                names.extend([name] + [name + "/" + a.name for a in node.names])
        names.extend(str(parent) for parent in PurePosixPath(path).parents if str(parent).startswith("src/"))
        for name in names:
            for candidate in (name + ".py", name + "/__init__.py"):
                if candidate in tracked and candidate not in paths:
                    _stop("omitted static code dependency: " + candidate)
    return deepcopy(closure)


def _environment(record):
    _keys(record, ("python", "python_implementation", "python_executable", "packages", "runtime"))
    if not record["python_executable"] or record["python_implementation"] != "CPython":
        _stop("runtime/interpreter identity missing")
    from .cmr1_bootstrap import REQUIRED_PINS
    if (record["python"] != "3.12.3" or record["packages"] != REQUIRED_PINS or
            record["runtime"].get("torch_cuda_version") != "12.8" or
            record["runtime"].get("numerical_mode_asserted") != NUMERICAL_MODE):
        _stop("environment/dependency/numerical-mode mismatch")
    return deepcopy(record)


def _capture_environment(runtime_context):
    from . import cmr1_bootstrap as bootstrap, cmr1_runtime as runtime
    import platform
    import sys
    # A previously attested process must not hide a changed mode by resetting it.
    import torch
    actual_mode = {
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "determinism_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
        "cpu_autocast_active": torch.is_autocast_enabled("cpu"),
        "cuda_autocast_active": torch.is_autocast_enabled("cuda"),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "numerical_mode": "ASSERTED",
    }
    if actual_mode != NUMERICAL_MODE:
        _stop("runtime numerical mode changed after attestation")
    bootstrap.assert_governed_entry(runtime_context, scope=SCOPE)
    lock = bootstrap.assert_complete_lock()
    return _environment({"python": platform.python_version(),
                         "python_implementation": platform.python_implementation(),
                         "python_executable": sys.executable, "packages": lock["observed"],
                         "runtime": runtime.runtime_identity()})


# Narrow pre-read ownership manifest, bound by the existing committed authorization.
# It is an execution input, not a target ledger or a new scientific contract.
METADATA_REGISTRATION_PATH = "manifests/cmr1_stage_b_metadata_ownership.json"
METADATA_REGISTRATION_SCHEMA = "latentbridge.cmr1.stage_b_metadata_ownership.v1"
METADATA_CLASSES = {
    "historical_exposure_inventory": ("exclusion", "exclusion"),
    "predecessor_exposure_metadata": ("exclusion", "exclusion"),
    "exclusion_evidence": ("exclusion", "exclusion"),
    "pinned_source_evidence": ("source", "materializer"),
    "frozen_selection_contract": ("rng", "governance"),
}


def _metadata_registry(repo, revision, evidence):
    ref = evidence.get("registration")
    doc = _json(_bound(repo, revision, ref, path=METADATA_REGISTRATION_PATH))
    _keys(doc, ("schema", "entries"))
    if doc["schema"] != METADATA_REGISTRATION_SCHEMA or not isinstance(doc["entries"], list):
        _stop("invalid metadata ownership registration")
    entries = {}
    for item in doc["entries"]:
        _keys(item, ("artifact", "class", "owner", "producer", "references"))
        _keys(item["artifact"], ("path", "sha256"))
        path = _path(item["artifact"]["path"])
        if path in entries:
            _stop("ambiguous metadata ownership registration")
        entries[path] = item
    return entries


def _metadata_graph(refs, registry, closure):
    """Authorize the entire declared reference graph before reading any member.

    Bodies remain opaque; no embedded URL/path triggers a fetch. Only references
    owned by this separately committed manifest may be followed. Each edge is
    checked independently, including edges from otherwise allowed metadata.
    """
    admitted, active = {}, set()

    def visit(ref):
        _keys(ref, ("path", "sha256"))
        path = _path(ref["path"])
        item = registry.get(path)
        if item is None or item["artifact"] != ref:
            _stop("unregistered metadata artifact or identity mismatch")
        kind = item["class"]
        if not isinstance(kind, str) or kind not in METADATA_CLASSES:
            _stop("forbidden or ambiguous Stage-B metadata class")
        owner, category = METADATA_CLASSES[kind]
        if item["owner"] != STATIC_PATHS[owner]:
            _stop("metadata owner/type mismatch")
        producer = item["producer"]
        _keys(producer, ("path", "sha256"))
        if (producer["path"] not in closure["categories"][category] or
                closure["blob_sha256"].get(producer["path"]) != producer["sha256"]):
            _stop("metadata producer mismatch")
        if not isinstance(item["references"], list):
            _stop("invalid metadata reference registration")
        if path in active:
            _stop("cyclic metadata reference registration")
        if path in admitted:
            return
        active.add(path)
        for child in item["references"]:
            visit(child)
        active.remove(path)
        admitted[path] = ref

    for ref in refs:
        visit(ref)
    return admitted


def _metadata_read(repo, revision, refs, evidence, closure):
    registry = _metadata_registry(repo, revision, evidence)
    admitted = _metadata_graph(refs, registry, closure)
    # No read/hash of referenced bytes above this line. Trust derives from the
    # authorization-bound committed registration, never a caller's type or name.
    return {path: _bound(repo, revision, ref) for path, ref in admitted.items()}


def _exposure(repo, revision, evidence, closure):
    _keys(evidence, ("inventories", "implementation", "retrieval_evidence",
                     "inventory_coverage", "retrieval_coverage", "registration"))
    if not evidence["inventories"] or not evidence["retrieval_evidence"]:
        _stop("required inventory/source metadata retrieval evidence absent")
    _metadata_graph(evidence["inventories"] + evidence["retrieval_evidence"],
                    _metadata_registry(repo, revision, evidence), closure)
    for field, names, refs in (("inventory_coverage", INVENTORY_COVERAGE, evidence["inventories"]),
                               ("retrieval_coverage", RETRIEVAL_COVERAGE, evidence["retrieval_evidence"])):
        _keys(evidence[field], names)
        for ref in evidence[field].values():
            if ref not in refs:
                _stop("mandatory coverage lacks an exact evidence reference")
    _metadata_read(repo, revision, evidence["inventories"] + evidence["retrieval_evidence"],
                   evidence, closure)  # opaque bytes, never parse inventory identities
    if not evidence["implementation"] or set(evidence["implementation"]) != set(closure["categories"]["exclusion"]):
        _stop("exposure/exclusion implementation ownership mismatch")
    return deepcopy(evidence)


def _target_spec(spec, closure, metadata_refs, *, production=False):
    _keys(spec, ("roles", "reference_roles", "consumer_bindings", "reuse_by_reference"))
    if not isinstance(spec["roles"], dict) or not spec["roles"] or not isinstance(spec["reference_roles"], dict):
        _stop("prospective target-selection roles missing")
    if set(spec["roles"]) & set(spec["reference_roles"]):
        _stop("duplicate derived/reference target role")
    all_roles = {**spec["roles"], **spec["reference_roles"]}
    for role, consumer in all_roles.items():
        if not isinstance(role, str) or not re.fullmatch(r"[A-Za-z0-9_-]+(?:/[A-Za-z0-9_-]+)*", role):
            _stop("invalid target role")
        fields = ("code_path", "symbol", "source", "pointer") if role in spec["reference_roles"] else ("code_path", "symbol")
        _keys(consumer, fields)
        if consumer["code_path"] not in closure["categories"]["materializer"] or not consumer["symbol"]:
            _stop("target RNG consumer not bound to materializer closure")
        if role in spec["reference_roles"]:
            _keys(consumer["source"], ("path", "sha256"))
            if consumer["source"] not in metadata_refs or not isinstance(consumer["pointer"], str) or not consumer["pointer"].startswith("/"):
                _stop("historical RNG reference must bind registered committed evidence")
    bindings = spec["consumer_bindings"]
    if not isinstance(bindings, dict) or not {"unit_seed", "window_seed"} <= set(bindings):
        _stop("unit/window selection consumers must be declared")
    if any(role not in all_roles for role in bindings.values()) or set(bindings.values()) != set(all_roles):
        _stop("undeclared or unconsumed target RNG role")
    if production:
        # B9 leaves the concrete prospective roles to Stage B. Bind every
        # environment's two consumers to the reviewed CMR1 implementation,
        # never to an H9/XR1 placeholder or an unresolvable symbol string.
        consumer_path = "src/silent_teacher/cmr1_target_materializer.py"
        required = {f"{env}.{kind}" for env in ("c4", "pes2o", "stackexchange")
                    for kind in ("unit_seed", "window_seed")}
        if not required <= set(bindings):
            _stop("CMR1 target RNG consumer mapping incomplete")
        for key in required:
            role = bindings[key]
            expected_symbol = "unit_rank" if key.endswith("unit_seed") else "window_start"
            if (role not in spec["roles"] or
                    spec["roles"][role] != {"code_path": consumer_path, "symbol": expected_symbol}):
                _stop("CMR1 target RNG consumer is not its frozen selector")
        historical_role = bindings.get("historical_minhash_seed")
        if (historical_role not in spec["reference_roles"] or
                spec["reference_roles"][historical_role]["code_path"] != consumer_path or
                spec["reference_roles"][historical_role]["symbol"] != "minhash_permutations" or
                spec["reference_roles"][historical_role]["source"]["path"] != STATIC_PATHS["exclusion"] or
                spec["reference_roles"][historical_role]["pointer"] !=
                    "/historical_exposure_inventory/xr1_signature_reconstruction/frozen_reconstruction_identity/seed"):
            _stop("historical MinHash identity must be a declared reference consumer")
        if consumer_path not in closure["categories"]["materializer"]:
            _stop("CMR1 production materializer missing from execution closure")
    for pair in spec["reuse_by_reference"]:
        if not isinstance(pair, list) or len(pair) != 2 or pair[0] == pair[1] or any(x not in spec["reference_roles"] for x in pair):
            _stop("reuse is not a declared historical identity reference")
        left, right = (spec["reference_roles"][x] for x in pair)
        if (left["source"], left["pointer"]) != (right["source"], right["pointer"]):
            _stop("reuse-by-reference cannot waive an unrelated seed collision")
    return deepcopy(spec)


def _historical_reference_bytes(repo, revision, ref, *, governed, evidence, closure):
    raw = _metadata_read(repo, revision, [ref["source"]], evidence, closure)[ref["source"]["path"]]
    if governed and raw != _blob(repo, APPROVED_BASE, ref["source"]["path"]):
        _stop("historical RNG source changed from the already-accepted baseline")
    return raw


def _target_records(repo, revision, spec, root, namespace, *, context=None):
    rng._campaign_entry(namespace, context)
    entry = _context(context)
    if (repo != entry["repo"] or revision != entry["revision"] or
            spec != entry["plan"]["target_selection"] or root != rng.campaign_root_seed(namespace, context=context)):
        _stop("target RNG derivation differs from attested Stage-B declaration")
    records = []
    for role in spec["roles"]:
        rng._campaign_entry(namespace, context)
        records.append(rng.RoleRecord("target_selection", role,
                       rng.derive(root, namespace + "/target-selection/" + role)))
    for role, ref in spec["reference_roles"].items():
        value = _json(_historical_reference_bytes(repo, revision, ref, governed=not entry["test_only"],
                    evidence=entry["plan"]["exposure"], closure=entry["plan"]["closure"]))
        try:
            for component in ref["pointer"].split("/")[1:]:
                value = value[component.replace("~1", "/").replace("~0", "~")]
        except (KeyError, TypeError) as exc:
            raise StageBError("HARD STOP - historical RNG reference pointer absent") from exc
        records.append(rng.RoleRecord("target_selection", role, value))
    return records


def _execution_spec(execution, closure, environment):
    """Bind prospective commands without invoking either executable or entrypoint."""
    _keys(execution, ("campaign", "materializer"))
    for operation, category in (("campaign", "training_runtime"), ("materializer", "materializer")):
        spec = execution[operation]
        _keys(spec, ("entrypoint", "arguments", "environment", "configurations"))
        if spec["entrypoint"] not in closure["categories"][category]:
            _stop("prospective command entrypoint outside its execution closure")
        if (not isinstance(spec["arguments"], list) or
                any(not isinstance(arg, str) for arg in spec["arguments"])):
            _stop("prospective command arguments must be explicit strings")
        if (not isinstance(spec["environment"], dict) or
                any(not isinstance(k, str) or not isinstance(v, str) for k, v in spec["environment"].items()) or
                spec["environment"].get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8"):
            _stop("prospective command environment not bound to numerical mode")
        if (not isinstance(spec["configurations"], list) or
                any(path not in closure["categories"]["configuration"] for path in spec["configurations"])):
            _stop("prospective command configuration outside closure")
    # Interpreter comes from the exact bound environment; there is no default Python.
    return {operation: [environment["python_executable"], spec["entrypoint"], *spec["arguments"]]
            for operation, spec in execution.items()}


def _validate_plan(repo, revision, plan, observed_environment):
    _keys(plan, ("namespace", "qualification", "contracts", "closure", "environment",
                 "exposure", "target_selection", "execution"))
    rng.validate_campaign_namespace(plan["namespace"])
    _keys(plan["contracts"], STATIC_PATHS)
    for key, path in STATIC_PATHS.items():
        raw = _bound(repo, revision, plan["contracts"][key], path=path)
        if repo == _REPO and raw != _blob(repo, APPROVED_BASE, path):
            _stop("closed scientific contract changed from the approved baseline")
    if repo == _REPO:
        # Previously qualified scientific production files cannot be silently replaced by
        # a new Stage-B closure. Only this task's authorization/RNG additions are exceptions.
        for path in ("src/silent_teacher/cmr1_student.py", "src/silent_teacher/cmr1_teacher.py",
                     "src/silent_teacher/cmr1_reference.py"):
            if path not in plan["closure"]["blob_sha256"] or _blob(repo, revision, path) != _blob(repo, APPROVED_BASE, path):
                _stop("qualified Teacher/Student implementation changed or unbound")
        for name, path in QUALIFICATION_PATHS.items():
            if _bound(repo, revision, plan["qualification"][name], path=path) != _blob(repo, APPROVED_BASE, path):
                _stop("accepted qualification evidence changed from the approved baseline")
    records = _qualification(repo, revision, plan["qualification"])
    closure = _closure(repo, revision, plan["closure"])
    expected = _environment(plan["environment"])
    if observed_environment != expected:
        _stop("runtime differs from the explicitly authorized environment")
    _exposure(repo, revision, plan["exposure"], closure)
    _target_spec(plan["target_selection"], closure,
                 list(plan["contracts"].values()) + plan["exposure"]["inventories"],
                 production=repo == _REPO)
    _metadata_read(repo, revision, [x["source"] for x in plan["target_selection"]["reference_roles"].values()],
                   plan["exposure"], closure)
    _execution_spec(plan["execution"], closure, expected)
    if repo == _REPO and (
            plan["execution"]["materializer"]["entrypoint"] != PRODUCTION_TARGET_ENTRYPOINT or
            PRODUCTION_TARGET_ENTRYPOINT not in closure["categories"]["materializer"] or
            PRODUCTION_TARGET_ENTRYPOINT not in closure["categories"]["exclusion"]):
        _stop("CMR1 target execution entrypoint/loader missing from immutable closure")
    return records


def _authorized_source_paths(repo, revision, plan):
    """The complete immutable input set whose tree mapping the context binds."""
    evidence, closure = plan["exposure"], plan["closure"]
    roots = (evidence["inventories"] + evidence["retrieval_evidence"] +
             [x["source"] for x in plan["target_selection"]["reference_roles"].values()])
    registry = _metadata_registry(repo, revision, evidence)
    metadata = _metadata_graph(roots, registry, closure)
    return (set(closure["blob_sha256"]) | set(metadata) | set(STATIC_PATHS.values()) |
            set(QUALIFICATION_PATHS.values()) |
            {auth.AUTHORIZATION_ARTIFACT_PATH, METADATA_REGISTRATION_PATH})


def _safe_output(repo, name, *, output_dir=OUTPUT_DIR):
    if output_dir not in (OUTPUT_DIR, ADMIN_REBUILD_DIR, TEACHER_METADATA_REBUILD_DIR, MATERIALIZER_INVENTORY_REBUILD_DIR, CUDA_BOOTSTRAP_REBUILD_DIR, RECOVERY_REBUILD_DIR, RECOVERY_FINAL_REBUILD_DIR, RECOVERY_ATTESTED_REBUILD_DIR, VERIFIED_LOCAL_ADMIN_DIR):
        _stop("unrecognized Stage-B output owner")
    path = repo / output_dir / _path(name)
    for part in (path, *path.parents):
        if part.is_symlink() or (part.exists() and getattr(part.stat(), "st_file_attributes", 0) & 0x400):
            _stop("output symlink/junction/reparse point forbidden")
        if part == repo:
            break
    if repo.resolve() not in path.resolve().parents:
        _stop("output escaped repository")
    return path


def _write_new(repo, name, document, *, output_dir=OUTPUT_DIR):
    path = _safe_output(repo, name, output_dir=output_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    _safe_output(repo, name, output_dir=output_dir)
    raw = canonical(document)
    try:
        with path.open("xb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError as exc:
        raise StageBError("HARD STOP - Stage-B identity exists; no replacement/retry") from exc
    return raw


def _context(context):
    if type(context) is not StageBContext or context not in _CONTEXTS:
        raise rng.GovernedQualificationNotAuthorized("attested Stage-B context required")
    return _CONTEXTS[context]


def assert_stage_b_context(context, *, namespace):
    entry = _context(context)
    if namespace != entry["plan"]["namespace"] or not entry["reserved"]:
        raise rng.GovernedQualificationNotAuthorized("Stage-B namespace not committed/reserved")
    if entry["test_only"]:
        rng.require_test_only_namespace(namespace)
    scope = auth.SCOPE_TEST_ONLY if entry["test_only"] else SCOPE
    if (not auth.capability_authorizes(entry["capability"], scope) or
            entry["capability"].authorized_phase != SCOPE):
        raise rng.GovernedQualificationNotAuthorized("exact Stage-B capability required")
    _attest_fresh(entry)


def load_stage_b_context(*, repo=_REPO, revision="HEAD", capability=None, runtime_context=None,
                         test_only=False, test_environment=None, output_dir=OUTPUT_DIR):
    """Validate a committed declaration and reserve it BEFORE root derivation.

    TEST-ONLY requires loader-issued test scope, Stage-B phase and a disposable repo.
    Production refuses caller-provided runtime evidence and has no namespace default.
    """
    repo = Path(repo).resolve()
    if output_dir not in (OUTPUT_DIR, ADMIN_REBUILD_DIR, TEACHER_METADATA_REBUILD_DIR, MATERIALIZER_INVENTORY_REBUILD_DIR, CUDA_BOOTSTRAP_REBUILD_DIR, RECOVERY_REBUILD_DIR, RECOVERY_FINAL_REBUILD_DIR, RECOVERY_ATTESTED_REBUILD_DIR, VERIFIED_LOCAL_ADMIN_DIR) or (test_only and output_dir != OUTPUT_DIR):
        _stop("unrecognized Stage-B output owner")
    if output_dir == ADMIN_REBUILD_DIR:
        invalidation = _json(_blob(repo, revision, OUTPUT_DIR + "/administrative_invalidation.json"))
        old_frozen = _json(_blob(repo, invalidation["old_freeze_commit"], OUTPUT_DIR + "/frozen.json"))
        if (invalidation.get("state") != "INVALIDATED_PRE_TARGET_INTERFACE_MISMATCH" or
                invalidation.get("reason") != "EXPOSURE_UNION_SCHEMA_INTERFACE_MISMATCH_DISCOVERED_BEFORE_ANY_TARGET_ACCESS" or
                invalidation.get("old_freeze_commit") != "65c7936274669b62965766d315182fd77f84cccb" or
                invalidation.get("old_frozen_sha256") != "d3d13dbb2c5487d49cf52ec9d3416951fda816fe7c598858629b6438d3f75f5b" or
                digest(canonical(old_frozen)) != invalidation["old_frozen_sha256"] or
                old_frozen.get("target_access") != "NO_TARGET_CONTENT_ACCESSED" or
                subprocess.run(["git", "show-ref", "--verify", "--quiet",
                                "refs/cmr1/target-materialization-attempt"], cwd=repo).returncode == 0):
            _stop("pre-target administrative invalidation missing or stale")
    if output_dir == TEACHER_METADATA_REBUILD_DIR:
        invalidation = _json(_blob(repo, revision,
            ADMIN_REBUILD_DIR + "/teacher_metadata_invalidation.json"))
        previous = _blob(repo, invalidation["old_freeze_commit"],
                         ADMIN_REBUILD_DIR + "/frozen.json")
        if (invalidation.get("state") != "INVALIDATED_PRE_TARGET_METADATA_INTERFACE_MISMATCH" or
                invalidation.get("reason") != "TEACHER_TOKENIZER_METADATA_PATH_MISMATCH_DISCOVERED_BEFORE_ANY_TARGET_ACCESS" or
                invalidation.get("old_freeze_commit") != "940d1749e3e4ef2f302c5d2ae517da4eed9e8806" or
                invalidation.get("old_frozen_sha256") != "e47ba70343447e1f66dc248d2f6dc7838f028e524326e7b6209907e992ebba40" or
                digest(previous) != invalidation["old_frozen_sha256"] or
                _json(previous).get("target_access") != "NO_TARGET_CONTENT_ACCESSED" or
                subprocess.run(["git", "show-ref", "--verify", "--quiet",
                                "refs/cmr1/target-materialization-attempt"], cwd=repo).returncode == 0):
            _stop("pre-target Teacher metadata invalidation missing or stale")
    if output_dir == MATERIALIZER_INVENTORY_REBUILD_DIR:
        invalidation = _json(_blob(repo, revision,
            TEACHER_METADATA_REBUILD_DIR + "/materializer_inventory_invalidation.json"))
        previous = _blob(repo, invalidation["old_freeze_commit"],
                         TEACHER_METADATA_REBUILD_DIR + "/frozen.json")
        if (invalidation.get("state") != "INVALIDATED_PRE_TARGET_METADATA_INTERFACE_MISMATCH" or
                invalidation.get("reason") != "HISTORICAL_UNIT_UNION_EXPECTED_SCHEMA_BINDING_MISMATCH_BEFORE_ANY_TARGET_ACCESS" or
                digest(previous) != invalidation.get("old_frozen_sha256") or
                _json(previous).get("target_access") != "NO_TARGET_CONTENT_ACCESSED" or
                subprocess.run(["git", "show-ref", "--verify", "--quiet",
                                "refs/cmr1/target-materialization-attempt"], cwd=repo).returncode == 0):
            _stop("pre-target materializer inventory invalidation missing or stale")
    if output_dir == CUDA_BOOTSTRAP_REBUILD_DIR:
        invalidation = _json(_blob(repo, revision,
            MATERIALIZER_INVENTORY_REBUILD_DIR + "/cuda_bootstrap_invalidation.json"))
        previous = _blob(repo, invalidation["old_freeze_commit"],
                         MATERIALIZER_INVENTORY_REBUILD_DIR + "/frozen.json")
        if (invalidation.get("state") != "INVALIDATED_PRE_TARGET_RUNTIME_ORDERING_MISMATCH" or
                invalidation.get("reason") != "CUDA_BOOTSTRAP_ORDERING_MISMATCH_DISCOVERED_BEFORE_ANY_TARGET_ACCESS" or
                digest(previous) != invalidation.get("old_frozen_sha256") or
                _json(previous).get("target_access") != "NO_TARGET_CONTENT_ACCESSED" or
                subprocess.run(["git", "show-ref", "--verify", "--quiet",
                                "refs/cmr1/target-materialization-attempt"], cwd=repo).returncode == 0):
            _stop("pre-target CUDA bootstrap invalidation missing or stale")
    if output_dir in (RECOVERY_REBUILD_DIR, RECOVERY_FINAL_REBUILD_DIR, RECOVERY_ATTESTED_REBUILD_DIR):
        old_ref = subprocess.run(["git", "--no-replace-objects", "rev-parse",
                                  "refs/cmr1/target-materialization-attempt"],
                                 cwd=repo, capture_output=True)
        old_dir = repo / "artifacts/cmr1/targets/materialization_attempt"
        try:
            terminal = _json((old_dir / "terminal.json").read_bytes())
            incident = _json((old_dir / "incident.json").read_bytes())
        except OSError as exc:
            raise StageBError("HARD STOP - recovery evidence absent") from exc
        if (old_ref.returncode != 0 or
                old_ref.stdout.decode().strip() != "58529918c704bac01710536b150d8dd881808415" or
                terminal.get("state") != "INTERRUPTED" or
                incident.get("materialization") != "NOT_STARTED" or
                incident.get("attempt_ref_oid") != old_ref.stdout.decode().strip() or
                subprocess.run(["git", "show-ref", "--verify", "--quiet",
                                "refs/cmr1/target-materialization-recovery-1"], cwd=repo).returncode == 0):
            _stop("infrastructure recovery closure prerequisite mismatch")
    if output_dir == VERIFIED_LOCAL_ADMIN_DIR:
        for ref, directory, expected_oid, state in (
            ("refs/cmr1/target-materialization-attempt", "materialization_attempt",
             "58529918c704bac01710536b150d8dd881808415", "INTERRUPTED"),
            ("refs/cmr1/target-materialization-recovery-1", "materialization_recovery_1",
             "9d9ded2c6eb52534b3e39936ef06f09308589340", "FAILED")):
            result = subprocess.run(["git", "--no-replace-objects", "rev-parse", ref],
                                    cwd=repo, capture_output=True)
            try:
                terminal = _json((repo / "artifacts/cmr1/targets" / directory /
                                  "terminal.json").read_bytes())
            except OSError as exc:
                raise StageBError("HARD STOP - prior infrastructure incident absent") from exc
            if (result.returncode != 0 or result.stdout.decode().strip() != expected_oid or
                    terminal.get("state") != state):
                _stop("verified-local administrative closure prerequisite mismatch")
        if (repo / "artifacts/cmr1/targets/stage_c_candidate").exists():
            _stop("verified-local administrative closure after materialization")
    if type(test_only) is not bool or (test_only and repo == _REPO) or (not test_only and repo != _REPO):
        _stop("repository/test-only trust-domain mismatch")
    scope = auth.SCOPE_TEST_ONLY if test_only else SCOPE
    if not auth.capability_authorizes(capability, scope) or capability.authorized_phase != SCOPE:
        _stop("exact Stage-B authorization required")
    revision = _git(repo, "rev-parse", revision + "^{commit}").decode().strip()
    if (_git(repo, "rev-parse", "HEAD").decode().strip() != revision or
            _git(repo, "status", "--porcelain=v1", "-uall").strip()):
        _stop("Stage-B realization requires clean authorized HEAD")
    raw = _blob(repo, revision, auth.AUTHORIZATION_ARTIFACT_PATH)
    if digest(raw) != capability.artifact_identity:
        _stop("capability does not bind this committed declaration")
    declared = _json(raw)
    if declared.get("repository_commit") != _git(repo, "rev-parse", revision + "^").decode().strip():
        _stop("authorization must identify its execution base")
    plan = declared.get("stage_b")
    if not isinstance(plan, dict):
        _stop("explicit committed Stage-B plan required; no default namespace")
    if test_only:
        rng.require_test_only_namespace(plan.get("namespace"))
        observed = _environment(test_environment)
    else:
        if rng.TEST_ONLY_MARKER in plan.get("namespace", "") or test_environment is not None:
            _stop("TEST-ONLY inputs cannot become campaign authority")
        observed = _capture_environment(runtime_context)
    records = _validate_plan(repo, revision, plan, observed)
    sources = _tree_blobs(repo, revision, _authorized_source_paths(repo, revision, plan))
    reservation = {"namespace": plan["namespace"], "authorization_sha256": digest(raw),
                   "authorization_commit": revision, "test_only": test_only,
                   "generation_time_utc": datetime.now(timezone.utc).isoformat(),
                   "status": "RESERVED_BEFORE_DERIVATION_NO_REPLACEMENT"}
    # Fixed exclusive owner; a different namespace/output location cannot create another root.
    if _git(repo, "log", "--all", "--format=%H", "--", output_dir + "/root_commitment.json").strip():
        _stop("a campaign root commitment already exists in Git history; no replacement")
    reservation_oid = _store_object(repo, canonical(reservation))
    _write_new(repo, "root_commitment.json", reservation, output_dir=output_dir)
    entry = {"repo": repo, "revision": revision, "plan": deepcopy(plan), "capability": capability,
             "runtime_context": runtime_context, "test_only": test_only, "reserved": True,
             "authorization_sha256": digest(raw), "qualification_records": records,
             "prepared": None, "reservation": reservation, "reservation_oid": reservation_oid,
             "source_bindings": sources, "plan_sha256": digest(canonical(plan)),
             "verified_commit": revision, "test_environment": test_environment,
             "output_dir": output_dir}
    context = StageBContext()
    _CONTEXTS[context] = entry
    return context


def _fresh_environment(entry):
    return (_environment(entry["test_environment"]) if entry["test_only"]
            else _capture_environment(entry["runtime_context"]))


def _attest_fresh(entry):
    """Re-attest exact committed path/object bindings before every RNG identity."""
    _revalidate(entry)


def _revalidate(entry):
    repo, revision, plan = entry["repo"], entry["revision"], entry["plan"]
    current = _git(repo, "rev-parse", "HEAD^{commit}").decode().strip()
    scope = auth.SCOPE_TEST_ONLY if entry["test_only"] else SCOPE
    if (not auth.capability_authorizes(entry["capability"], scope) or
            entry["capability"].authorized_phase != SCOPE):
        _stop("Stage-B authorization stale")
    if digest(canonical(plan)) != entry["plan_sha256"]:
        _stop("Stage-B context plan changed")
    # Equality to an already verified *commit object ID* proves the whole tree
    # mapping unchanged. A new commit must prove every bound path/object anew.
    if current != entry["verified_commit"]:
        if _git(repo, "merge-base", revision, current).decode().strip() != revision:
            _stop("current commit no longer descends from Stage-B authorization")
        if _tree_blobs(repo, current, entry["source_bindings"]) != entry["source_bindings"]:
            _stop("authorization-bound immutable source identity changed")
        authorization = _object_bytes(repo, entry["source_bindings"][auth.AUTHORIZATION_ARTIFACT_PATH])
        if (digest(authorization) != entry["authorization_sha256"] or
                _json(authorization).get("stage_b") != plan):
            _stop("committed Stage-B authorization changed")
    observed = _fresh_environment(entry)
    if observed != plan["environment"]:
        _stop("runtime differs from the immutable authorized environment")
    if _object_bytes(repo, entry["reservation_oid"]) != canonical(entry["reservation"]):
        _stop("immutable root commitment changed")
    if current != entry["verified_commit"]:
        committed_root = _git(repo, "ls-tree", "-r", "-z", current, "--",
                              entry["output_dir"] + "/root_commitment.json")
        if committed_root and _tree_blobs(repo, current, [entry["output_dir"] + "/root_commitment.json"])[
                entry["output_dir"] + "/root_commitment.json"] != entry["reservation_oid"]:
            _stop("committed root commitment differs from reserved object")
    # The local output path is checked only for one-shot presence/ownership.
    # Its mutable bytes are never opened or used as RNG authority.
    if not _safe_output(repo, "root_commitment.json", output_dir=entry["output_dir"]).is_file():
        _stop("root commitment missing")
    entry["verified_commit"] = current


def _candidate_document(context):
    entry = _context(context)
    assert_stage_b_context(context, namespace=entry["plan"]["namespace"])
    namespace = entry["plan"]["namespace"]
    root = rng.campaign_root_seed(namespace, context=context)
    records = [rng.RoleRecord("campaign", f"replicate-{r:02d}/{role}",
               rng.campaign_child_seed(root, namespace, r, role, context=context))
               for r in range(1, 49) for role in rng.CAMPAIGN_ROLES]
    within = validate_campaign_records(records, namespace, root, context=context)
    spec = entry["plan"]["target_selection"]
    target = _target_records(entry["repo"], entry["revision"], spec, root, namespace, context=context)
    target_check = rng.validate_within_domain(target, "target_selection",
                         declared_reuse_by_reference=spec["reuse_by_reference"])
    cross = rng.validate_cross_domain([r.integer_seed for r in records],
              [r.integer_seed for r in entry["qualification_records"]], [r.integer_seed for r in target])
    document = {"schema": SCHEMA, "state": PREPARED, "test_only": entry["test_only"],
                "authoritative": False, "stage_c": "NOT_STAGE_C",
                "target_access": "NO_TARGET_CONTENT_ACCESSED", "forbidden": list(FORBIDDEN),
                "authorization_commit": entry["revision"],
                "authorization_sha256": entry["authorization_sha256"],
                "root_commitment": entry["reservation"], "root_seed": root,
                "plan": deepcopy(entry["plan"]),
                "campaign_records": [{"domain": r.domain, "role_key": r.role_key,
                                      "integer_seed": r.integer_seed} for r in records],
                "target_selection_records": [{"domain": r.domain, "role_key": r.role_key,
                                              "integer_seed": r.integer_seed} for r in target],
                "checks": {"campaign": within, "target_selection": target_check, "cross_domain": cross},
                "pairing": {"arms": ["nat", "pos"], "independent_qpos_bank": False,
                            "shared_roles": [r for r in rng.CAMPAIGN_ROLES if not r.startswith("permutation_")],
                            "permutation_consumers": ["pos"]},
                "failure_policy": entry["plan"]["contracts"]["failure"],
                "training_authorized": False, "materialization_authorized": False}
    if entry["output_dir"] in (RECOVERY_REBUILD_DIR, RECOVERY_FINAL_REBUILD_DIR, RECOVERY_ATTESTED_REBUILD_DIR, VERIFIED_LOCAL_ADMIN_DIR):
        document["prior_acquisition_attempt"] = {
            "ref_oid": "58529918c704bac01710536b150d8dd881808415",
            "terminal": "INTERRUPTED", "materialization_started": False,
            "classification": "PRE_MATERIALIZATION_INFRASTRUCTURE_ACQUISITION_FAILURE"}
    return document


def prepare_stage_b(context):
    entry = _context(context)
    if entry["prepared"] is not None:
        _stop("candidate already prepared; no new realization")
    _revalidate(entry)
    document = _candidate_document(context)
    raw = _write_new(entry["repo"], "candidate.json", document, output_dir=entry["output_dir"])
    entry["prepared"] = raw
    return deepcopy(document)


def freeze_stage_b(context, *, acknowledgement_revision="HEAD",
                   acknowledgement_path="manifests/cmr1_stage_b/review_acknowledgement.json"):
    entry = _context(context)
    assert_stage_b_context(context, namespace=entry["plan"]["namespace"])
    if entry["prepared"] is None:
        _stop("no prepared Stage-B candidate")
    _revalidate(entry)
    raw = _safe_output(entry["repo"], "candidate.json", output_dir=entry["output_dir"]).read_bytes()
    if raw != entry["prepared"]:
        _stop("prepared candidate mutated")
    if _blob(entry["repo"], acknowledgement_revision, entry["output_dir"] + "/candidate.json") != raw:
        _stop("review must reference the exact committed candidate")
    ack_raw = _blob(entry["repo"], acknowledgement_revision, acknowledgement_path)
    ack = _json(ack_raw)
    _keys(ack, ("decision", "candidate_sha256", "authorization_sha256", "test_only", "reviewer"))
    decision = "TEST-ONLY_STAGE_B_APPROVED" if entry["test_only"] else "CMR1_STAGE_B_FREEZE_APPROVED"
    if (ack["decision"] != decision or ack["candidate_sha256"] != digest(raw) or
            ack["authorization_sha256"] != entry["authorization_sha256"] or
            ack["test_only"] is not entry["test_only"] or not ack["reviewer"]):
        _stop("stale, missing or wrong-domain review acknowledgement")
    document = _json(raw)
    document.update(state=FROZEN, authoritative=not entry["test_only"],
                    review={"acknowledgement_path": acknowledgement_path,
                            "acknowledgement_revision": _git(entry["repo"], "rev-parse", acknowledgement_revision).decode().strip(),
                            "acknowledgement_sha256": digest(ack_raw), "candidate_sha256": digest(raw)})
    frozen_raw = _write_new(entry["repo"], "frozen.json", document, output_dir=entry["output_dir"])
    attestation = FrozenStageB()
    _FROZEN[attestation] = (frozen_raw, entry["test_only"], "UNCOMMITTED_FREEZE")
    return attestation


def consume_frozen_stage_b(attestation, *, governed=True):
    if type(attestation) is not FrozenStageB or attestation not in _FROZEN:
        _stop("unfrozen candidate cannot satisfy Stage-C prerequisite")
    raw, test_only, source = _FROZEN[attestation]
    if governed and source != "COMMITTED_FROZEN":
        _stop("Stage-C prerequisite requires a committed Stage-B frozen record")
    if governed and test_only:
        _stop("TEST-ONLY freeze is not real campaign authority")
    return deepcopy(_json(raw))


def reject_target_or_training(context, operation):
    """Stage B never authorizes content access, Stage C or campaign training."""
    _context(context)
    _stop("Stage-B scope cannot authorize " + str(operation))


def restore_stage_b_for_review(*, repo=_REPO, revision="HEAD", capability=None,
                               runtime_context=None, test_only=False, test_environment=None,
                               output_dir=OUTPUT_DIR):
    """Read-only continuation of ONE committed candidate; creates no reservation/root/artifact.

    Re-derivation here validates existing identities; it cannot select or persist another root.
    The original committed authorization remains required even across review commits.
    """
    repo = Path(repo).resolve()
    if type(test_only) is not bool or (test_only and repo == _REPO) or (not test_only and repo != _REPO):
        _stop("repository/test-only trust-domain mismatch")
    scope = auth.SCOPE_TEST_ONLY if test_only else SCOPE
    if not auth.capability_authorizes(capability, scope) or capability.authorized_phase != SCOPE:
        _stop("exact Stage-B authorization required")
    if output_dir not in (OUTPUT_DIR, ADMIN_REBUILD_DIR, TEACHER_METADATA_REBUILD_DIR, MATERIALIZER_INVENTORY_REBUILD_DIR, CUDA_BOOTSTRAP_REBUILD_DIR, RECOVERY_REBUILD_DIR, RECOVERY_FINAL_REBUILD_DIR, RECOVERY_ATTESTED_REBUILD_DIR, VERIFIED_LOCAL_ADMIN_DIR) or (test_only and output_dir != OUTPUT_DIR):
        _stop("unrecognized Stage-B output owner")
    raw = _blob(repo, revision, output_dir + "/candidate.json")
    candidate = _json(raw)
    if candidate.get("state") != PREPARED or candidate.get("test_only") is not test_only:
        _stop("wrong candidate lifecycle/domain")
    authorization_revision = candidate.get("authorization_commit")
    authorization_raw = _blob(repo, authorization_revision, auth.AUTHORIZATION_ARTIFACT_PATH)
    if digest(authorization_raw) != capability.artifact_identity:
        _stop("candidate authorization mismatch")
    declaration = _json(authorization_raw)
    plan = declaration.get("stage_b")
    if plan != candidate.get("plan"):
        _stop("candidate differs from committed authorized plan")
    if test_only:
        rng.require_test_only_namespace(plan.get("namespace"))
        observed = _environment(test_environment)
    else:
        if test_environment is not None or rng.TEST_ONLY_MARKER in plan.get("namespace", ""):
            _stop("TEST-ONLY input cannot become authority")
        observed = _capture_environment(runtime_context)
    records = _validate_plan(repo, authorization_revision, plan, observed)
    reservation = _json(_blob(repo, revision, output_dir + "/root_commitment.json"))
    reservation_oid = _tree_blobs(repo, revision, [output_dir + "/root_commitment.json"])[
        output_dir + "/root_commitment.json"]
    if (candidate.get("root_commitment") != reservation or reservation.get("namespace") != plan["namespace"] or
            reservation.get("authorization_sha256") != capability.artifact_identity or
            reservation.get("authorization_commit") != authorization_revision or
            reservation.get("test_only") is not test_only):
        _stop("candidate root commitment mismatch")
    context = StageBContext()
    _CONTEXTS[context] = {"repo": repo, "revision": authorization_revision, "plan": deepcopy(plan),
                         "capability": capability, "runtime_context": runtime_context,
                         "test_only": test_only, "reserved": True,
                         "authorization_sha256": capability.artifact_identity,
                         "qualification_records": records, "prepared": raw, "reservation": reservation,
                         "reservation_oid": reservation_oid,
                         "source_bindings": _tree_blobs(repo, authorization_revision,
                             _authorized_source_paths(repo, authorization_revision, plan)),
                         "plan_sha256": digest(canonical(plan)),
                         "verified_commit": authorization_revision,
                         "test_environment": test_environment, "output_dir": output_dir}
    if canonical(_candidate_document(context)) != raw:
        _stop("candidate RNG/closure/lifecycle fields changed")
    _revalidate(_context(context))
    return context


def load_committed_frozen_stage_b(*, repo=_REPO, revision="HEAD", capability=None,
                                 runtime_context=None, test_only=False, test_environment=None,
                                 output_dir=OUTPUT_DIR):
    """Persistent Stage-C prerequisite loader; conveys NO materialization/training scope."""
    context = restore_stage_b_for_review(repo=repo, revision=revision, capability=capability,
             runtime_context=runtime_context, test_only=test_only, test_environment=test_environment,
             output_dir=output_dir)
    entry = _context(context)
    raw = _blob(entry["repo"], revision, output_dir + "/frozen.json")
    doc = _json(raw)
    review = doc.get("review", {})
    _keys(review, ("acknowledgement_path", "acknowledgement_revision", "acknowledgement_sha256", "candidate_sha256"))
    ack_raw = _blob(entry["repo"], review["acknowledgement_revision"], review["acknowledgement_path"])
    ack = _json(ack_raw)
    decision = "TEST-ONLY_STAGE_B_APPROVED" if test_only else "CMR1_STAGE_B_FREEZE_APPROVED"
    _keys(ack, ("decision", "candidate_sha256", "authorization_sha256", "test_only", "reviewer"))
    if (digest(ack_raw) != review["acknowledgement_sha256"] or
            ack["candidate_sha256"] != digest(entry["prepared"]) or
            review["candidate_sha256"] != ack["candidate_sha256"] or
            ack["authorization_sha256"] != capability.artifact_identity or
            ack["decision"] != decision or ack["test_only"] is not test_only or not ack["reviewer"]):
        _stop("committed freeze review mismatch")
    if _blob(entry["repo"], review["acknowledgement_revision"], output_dir + "/candidate.json") != entry["prepared"]:
        _stop("acknowledgement did not bind committed candidate")
    expected = _json(entry["prepared"])
    expected.update(state=FROZEN, authoritative=not test_only, review=review)
    if canonical(expected) != raw:
        _stop("committed frozen record changed from reviewed candidate")
    attestation = FrozenStageB()
    _FROZEN[attestation] = (raw, test_only, "COMMITTED_FROZEN")
    return attestation
