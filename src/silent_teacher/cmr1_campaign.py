"""CMR1 paired campaign execution from committed Stage-B and Stage-C identities.

No target array enters donor construction or the optimization loop. The Stage-C
arrays are read only for their frozen content identity during launch preflight.
"""
from __future__ import annotations

import hashlib
import errno
import json
import math
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import torch

from . import cmr1_fixtures, cmr1_rng, cmr1_student, cmr1_teacher
from . import cmr1_bootstrap
from .full_block_trainer import learning_rate
from .h9_id2 import TRAINING_CONTRACT, assert_training_contract, training_config
from .records import state_hash

REPO = Path(__file__).resolve().parents[2]
STAGE_B_PATH = "manifests/cmr1_stage_b_verified_local_admin_refresh/frozen.json"
STAGE_C_PATH = "manifests/cmr1_stage_c/frozen.json"
STAGE_B_SHA = "ef7f39e905aae1084db8a4a81ec162750c58b46033110df0866c90aac3bbbf55"
STAGE_C_SHA = "3bdb00ad53803510d1ed4b34d3facc7c5d4e7a05752f59a922832008d552798d"
SPLITS = {"train": 4096, "dev": 256, "audit": 512}
ARMS = ("nat", "pos")


class CampaignHardStop(RuntimeError):
    pass


class AlgorithmicNonfinite(RuntimeError):
    pass


def _need(ok: bool, reason: str) -> None:
    if not ok:
        raise CampaignHardStop(reason)


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _canonical(obj: object) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("utf-8")


def _committed(path: str) -> bytes:
    return subprocess.run(("git", "show", "HEAD:" + path), cwd=REPO,
                          check=True, capture_output=True).stdout


def _record(path: str, expected: str) -> dict:
    raw = _committed(path)
    _need(_sha(raw) == expected and (REPO / path).read_bytes() == raw,
          "COMMITTED_IDENTITY_MISMATCH: " + path)
    return json.loads(raw)


def load_frozen_campaign() -> tuple[dict, dict, dict]:
    """Fail closed on the fixed 48x9 roles and the committed target ledger."""
    stage_b = _record(STAGE_B_PATH, STAGE_B_SHA)
    stage_c = _record(STAGE_C_PATH, STAGE_C_SHA)
    _need(stage_b["state"] == "FROZEN_STAGE_B_EXECUTION_CONTROL" and
          stage_c["state"] == "FROZEN_STAGE_C_TARGET_LEDGER" and
          stage_c["stage_b_frozen_sha256"] == STAGE_B_SHA and
          stage_c["training_authorized"] is False and
          stage_c["materialization_terminal_state"] == "SUCCEEDED" and
          stage_b["root_seed"] == stage_c["campaign_root"] == 2742875683296071956 and
          stage_b["root_commitment"]["namespace"] == stage_c["namespace"] ==
          "latentbridge/cmr1/v1/formal-v1", "FROZEN_CAMPAIGN_BINDING_MISMATCH")
    roles = {row["role_key"]: row["integer_seed"] for row in stage_b["campaign_records"]}
    _need(len(roles) == 432 and len(set(roles.values())) == 432 and
          set(roles) == {f"replicate-{i:02d}/{role}" for i in range(1, 49)
                         for role in cmr1_rng.CAMPAIGN_ROLES} and
          _sha(_canonical(stage_b["campaign_records"])) == stage_c["campaign_records_sha256"],
          "CAMPAIGN_RNG_MANIFEST_MISMATCH")
    ledger_ref = stage_c["candidate_ledger"]
    ledger_raw = _committed(ledger_ref["path"])
    _need(_sha(ledger_raw) == ledger_ref["sha256"] and
          (REPO / ledger_ref["path"]).read_bytes() == ledger_raw,
          "STAGE_C_LEDGER_MISMATCH")
    ledger = json.loads(ledger_raw)
    _need(ledger["stage_b_frozen_sha256"] == STAGE_B_SHA and
          ledger["target_selection_records_sha256"] == stage_c["target_selection_records_sha256"] and
          set(ledger["environments"]) == {"c4", "pes2o", "stackexchange"},
          "STAGE_C_LEDGER_BINDING_MISMATCH")
    for env, entry in stage_c["environments"].items():
        path = entry["array_path"]
        raw = _committed(path)
        _need(_sha(raw) == entry["array_sha256"] == ledger["environments"][env]["array_sha256"] and
              (REPO / path).read_bytes() == raw and
              entry["shape"] == [1024, 128] and entry["dtype"] == "int64",
              "STAGE_C_ARRAY_MISMATCH: " + env)
        values = np.load(REPO / path, mmap_mode="r", allow_pickle=False)
        _need(values.shape == (1024, 128) and values.dtype == np.dtype("int64"),
              "STAGE_C_ARRAY_LAYOUT_MISMATCH: " + env)
    return stage_b, stage_c, roles


def role_seeds(roles: dict, replicate: int) -> dict[str, int]:
    _need(type(replicate) is int and 1 <= replicate <= 48, "REPLICATE_OUT_OF_RANGE")
    result = {role: roles[f"replicate-{replicate:02d}/{role}"]
              for role in cmr1_rng.CAMPAIGN_ROLES}
    _need(len(result) == 9 and len(set(result.values())) == 9, "REPLICATE_RNG_MISMATCH")
    return result


def production_config() -> dict:
    config = training_config()
    assert_training_contract(config, steps=10000, horizon=10000)
    _need(config["microbatch"] == 8 and config["accumulation"] == 4 and
          config["steps"] == 10000 and config["warmup"] == 200 and
          TRAINING_CONTRACT["terminal_checkpoint_step"] == 10000 and
          TRAINING_CONTRACT["checkpoint_selection"] == "none" and
          TRAINING_CONTRACT["early_stopping"] == "none", "TRAINING_RECIPE_MISMATCH")
    return config


def launch_preflight(teacher_root: Path, output_root: Path) -> dict:
    stage_b, stage_c, roles = load_frozen_campaign()
    production_config()
    cmr1_bootstrap.assert_complete_lock(REPO / "requirements-lock.txt")
    cmr1_bootstrap.assert_governed_numerical_boundary()
    _need(os.environ.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8" and
          torch.cuda.is_available() and not torch.backends.cuda.matmul.allow_tf32 and
          not torch.backends.cudnn.allow_tf32, "NUMERICAL_RUNTIME_NOT_READY")
    existing_parent = output_root.parent
    while not existing_parent.exists() and existing_parent != existing_parent.parent:
        existing_parent = existing_parent.parent
    _need(not output_root.exists() and existing_parent.is_dir() and
          os.access(existing_parent, os.W_OK) and
          shutil.disk_usage(existing_parent).free >= 8 * (1 << 30),
          "OUTPUT_PATH_OR_DISK_NOT_READY")
    _need((teacher_root / "model.safetensors").is_file() and
          (teacher_root / "config.json").is_file(), "PINNED_TEACHER_ASSETS_ABSENT")
    _need(cmr1_teacher.sha256_of_file(teacher_root / "model.safetensors") ==
          cmr1_teacher.PINNED_WEIGHT_SHA256, "PINNED_TEACHER_WEIGHTS_MISMATCH")
    teacher_config = json.loads((teacher_root / "config.json").read_text(encoding="utf-8"))
    _need(all(teacher_config.get(key) == value for key, value in
              cmr1_teacher.FROZEN_CONFIG_FACTS.items()), "PINNED_TEACHER_CONFIG_MISMATCH")
    return {"stage_b_sha256": STAGE_B_SHA, "stage_c_sha256": STAGE_C_SHA,
            "namespace": stage_c["namespace"], "replicates": 48, "arms": list(ARMS),
            "campaign_roles": len(roles), "teacher_root": str(teacher_root),
            "output_root": str(output_root)}


def _generator(seed: int) -> torch.Generator:
    _need(type(seed) is int and 0 <= seed < cmr1_rng.MODULUS, "ROLE_SEED_INVALID")
    result = torch.Generator(device="cpu")
    result.manual_seed(seed)
    return result


def native_token_ids(seed: int, count: int) -> torch.Tensor:
    """Frozen iid uniform accepted IDs; no corpus or target content is consulted."""
    ids = torch.randint(0, 50277, (count, 128), generator=_generator(seed), dtype=torch.int64)
    _need(ids.shape == (count, 128) and int(ids.min()) >= 0 and int(ids.max()) < 50277,
          "NATIVE_TOKEN_LAW_MISMATCH")
    return ids


def _tensor_sha(value: torch.Tensor) -> str:
    return _sha(value.detach().cpu().contiguous().numpy().tobytes(order="C"))


def failure_class(error: BaseException) -> str:
    """Frozen B10 classes, without automatic replay or replacement."""
    if isinstance(error, AlgorithmicNonfinite):
        return "ALGORITHMIC_NONFINITE"
    if isinstance(error, CampaignHardStop):
        return "SCIENTIFIC_IMPLEMENTATION_MISMATCH"
    if isinstance(error, (MemoryError, ConnectionError, TimeoutError)) or (
            isinstance(error, OSError) and error.errno in {
                errno.ENOSPC, errno.ENOMEM, errno.EIO, errno.EMFILE, errno.ENFILE,
                errno.EAGAIN, errno.ETIMEDOUT, errno.ENETDOWN, errno.ENETUNREACH,
                errno.EHOSTUNREACH}):
        return "INFRASTRUCTURE"
    return "UNKNOWN_UNCLASSIFIED"


def build_banks(teacher, seeds: dict, split: str, device: str, *, count: int) -> tuple[dict, dict]:
    """One Native bank shared by both arms; independent position permutations."""
    _need(split in SPLITS, "UNKNOWN_SPLIT")
    token_ids = native_token_ids(seeds["donor_" + split], count)
    chunks = []
    with torch.no_grad():
        for part in token_ids.split(8):
            hidden = teacher.capture_boundary(part.to(device))
            _need(hidden.shape == (len(part), 128, 768) and hidden.dtype == torch.float32 and
                  bool(torch.isfinite(hidden).all()), "DONOR_BOUNDARY_MISMATCH")
            chunks.append(hidden.cpu())
    native = torch.cat(chunks).contiguous()
    generator = _generator(seeds["permutation_" + split])
    permutations = [torch.randperm(count, generator=generator).tolist() for _ in range(128)]
    pos = cmr1_fixtures.apply_qpos_permutation(native, permutations)
    cmr1_fixtures.assert_positionwise_multiset_invariant(native, pos)
    evidence = {"native_sha256": _tensor_sha(native), "qpos_sha256": _tensor_sha(pos),
                "token_ids_sha256": _tensor_sha(token_ids),
                "permutations_sha256": _sha(_canonical(permutations)),
                "donor_role": "donor_" + split, "permutation_role": "permutation_" + split,
                "count": count}
    return {"nat": native, "pos": pos}, evidence


def minibatch_permutation(seed: int, count: int) -> torch.Tensor:
    return torch.randperm(count, generator=_generator(seed))


def train_arm(student, teacher, train_bank: torch.Tensor, dev_bank: torch.Tensor,
              permutation: torch.Tensor, config: dict, device: str, folder: Path,
              *, test_only: bool = False) -> dict:
    """Terminal-only AdamW run; Teacher labels recomputed for each actual input."""
    _need(isinstance(train_bank, torch.Tensor) and isinstance(dev_bank, torch.Tensor) and
          train_bank.ndim == dev_bank.ndim == 3 and
          train_bank.shape[1:] == dev_bank.shape[1:] == (128, 768) and
          train_bank.dtype == dev_bank.dtype == torch.float32,
          "SYNTHETIC_HIDDEN_BANK_REQUIRED_NO_TARGET_TOKEN_ARRAY")
    steps = int(config["steps"])
    if not test_only:
        assert_training_contract(config, steps=steps, horizon=10000)
    else:
        _need(1 <= steps <= 2, "TEST_ONLY_STEP_LIMIT")
    micro, accumulation = int(config["microbatch"]), int(config["accumulation"])
    effective = micro * accumulation
    _need(len(train_bank) >= effective and len(train_bank) % effective == 0 and
          len(permutation) == len(train_bank) and not folder.exists(),
          "TRAINING_INPUT_OR_OUTPUT_MISMATCH")
    folder.mkdir(parents=True, exist_ok=False)
    student = student.to(device).train()
    optimizer = torch.optim.AdamW(student.parameters(), lr=config["lr_peak"],
                                  betas=tuple(config["betas"]), eps=config["eps"],
                                  weight_decay=config["weight_decay"])
    completed = 0
    try:
        with (folder / "curve.jsonl").open("x", encoding="utf-8") as log:
            for step in range(1, steps + 1):
                lr = learning_rate(step, config["lr_peak"], config["lr_final_ratio"],
                                   config["warmup"], 10000)
                for group in optimizer.param_groups:
                    group["lr"] = lr
                optimizer.zero_grad(set_to_none=True)
                offset = ((step - 1) % (len(train_bank) // effective)) * effective
                loss_sum = 0.0
                for micro_index in range(accumulation):
                    indices = permutation[offset + micro_index * micro:
                                          offset + (micro_index + 1) * micro]
                    hidden = train_bank[indices].to(device)
                    with torch.no_grad():
                        labels = teacher.residual_update(hidden)
                    loss = (student(hidden) - labels).square().mean() / accumulation
                    if not bool(torch.isfinite(loss)):
                        raise AlgorithmicNonfinite("NONFINITE_LOSS")
                    loss.backward()
                    loss_sum += float(loss.detach())
                norm = torch.nn.utils.clip_grad_norm_(student.parameters(), config["clip_norm"],
                                                      error_if_nonfinite=False)
                if not bool(torch.isfinite(norm)) or any(
                        p.grad is not None and not bool(torch.isfinite(p.grad).all())
                        for p in student.parameters()):
                    raise AlgorithmicNonfinite("NONFINITE_GRADIENT")
                optimizer.step()
                if any(not bool(torch.isfinite(p).all()) for p in student.parameters()):
                    raise AlgorithmicNonfinite("NONFINITE_PARAMETER")
                completed = step
                row = {"step": step, "loss": loss_sum, "gradient_norm": float(norm), "lr": lr}
                if step % int(config["eval_every"]) == 0 or step == steps:
                    student.eval()
                    numerator = denominator = 0.0
                    with torch.no_grad():
                        for part in dev_bank.split(micro):
                            hidden = part.to(device)
                            labels = teacher.residual_update(hidden)
                            numerator += float((student(hidden).double() - labels.double()).square().sum())
                            denominator += float(labels.double().square().sum())
                    student.train()
                    if denominator <= 0 or not np.isfinite(numerator / denominator):
                        raise AlgorithmicNonfinite("SYNTHETIC_DEV_NONFINITE")
                    row["synthetic_dev_nmse"] = numerator / denominator
                log.write(json.dumps(row, allow_nan=False) + "\n")
                log.flush()
        _need(completed == steps, "OPTIMIZER_STEP_COUNT_MISMATCH")
        checkpoint = folder / "terminal.pt"
        temp = folder / "terminal.pt.tmp"
        torch.save({"student": student.state_dict(), "optimizer": optimizer.state_dict(),
                    "step": completed, "config": config, "stage_c_sha256":
                    None if test_only else STAGE_C_SHA}, temp)
        temp.replace(checkpoint)
        result = {"state": "COMPLETE", "optimizer_steps": completed,
                  "checkpoint_sha256": _sha(checkpoint.read_bytes()),
                  "test_only": test_only}
        (folder / "terminal.json").write_bytes(_canonical(result))
        return result
    except BaseException as error:
        (folder / "failure.json").write_bytes(_canonical(
            {"state": "FAILED_HARD_STOP", "class": failure_class(error), "optimizer_steps": completed,
             "error": repr(error), "automatic_retry": False, "replacement_seed": False}))
        raise


def run_replicate(teacher, roles: dict, replicate: int, output_root: Path, device: str) -> dict:
    seeds = role_seeds(roles, replicate)
    folder = output_root / f"replicate_{replicate:02d}"
    _need(not folder.exists(), "REPLICATE_OUTPUT_ALREADY_EXISTS")
    folder.mkdir(parents=True, exist_ok=False)
    config = production_config()
    record = {"replicate": replicate, "state": "PAIR_INCOMPLETE", "seeds": seeds,
              "stage_c_sha256": STAGE_C_SHA, "construction": {}, "arms": {},
              "context_donor_role_reserved_for_evaluation": seeds["context_donor"]}
    try:
        banks = {}
        for split, count in SPLITS.items():
            pair, evidence = build_banks(teacher, seeds, split, device, count=count)
            record["construction"][split] = evidence
            if split != "audit":
                banks[split] = pair
        permutation = minibatch_permutation(seeds["minibatch_schedule"], SPLITS["train"])
        record["minibatch_permutation_sha256"] = _tensor_sha(permutation)
        for arm in ARMS:
            student = cmr1_student.build_student(seed=seeds["student_init"])
            initialization = state_hash(student)
            if arm == "nat":
                record["shared_initialization_sha256"] = initialization
            else:
                _need(initialization == record["shared_initialization_sha256"],
                      "PAIRED_INITIALIZATION_MISMATCH")
            _need(_tensor_sha(permutation) == record["minibatch_permutation_sha256"],
                  "PAIRED_MINIBATCH_SCHEDULE_MISMATCH")
            result = train_arm(student, teacher, banks["train"][arm], banks["dev"][arm],
                               permutation, config, device, folder / arm)
            record["arms"][arm] = result
            (folder / "progress.json").write_bytes(_canonical(record))
            del student
            torch.cuda.empty_cache()
        record["state"] = "PAIR_COMPLETE"
        (folder / "replicate.json").write_bytes(_canonical(record))
        return record
    except BaseException as error:
        record["state"] = "PAIR_FAILED_HARD_STOP"
        record["failure_class"] = failure_class(error)
        record["error"] = repr(error)
        record["automatic_retry"] = False
        (folder / "replicate.json").write_bytes(_canonical(record))
        raise


def load_pinned_teacher_wrapper(teacher_root: Path, device: str):
    """Adapt the pinned loader's (model, block) result for the campaign path."""
    model, block = cmr1_teacher.build_pinned_teacher(teacher_root, device=device)
    teacher = cmr1_teacher.BlockFiveWrapper(model)
    _need(teacher.block is block, "PINNED_TEACHER_BLOCK_MISMATCH")
    cmr1_teacher.assert_pinned_teacher_wrapper(teacher)
    return teacher


def run_test_only_entrypoint_smoke(teacher_root: Path) -> dict:
    """Exercise the production Teacher/donor/Student path without formal seeds or outputs."""
    launch_preflight(teacher_root, REPO / "runs/cmr1/formal-v1")
    torch.cuda.reset_peak_memory_stats()
    teacher = load_pinned_teacher_wrapper(teacher_root, "cuda")
    seeds = {role: index + 101 for index, role in enumerate(cmr1_rng.CAMPAIGN_ROLES)}
    train, train_evidence = build_banks(teacher, seeds, "train", "cuda", count=8)
    dev, dev_evidence = build_banks(teacher, seeds, "dev", "cuda", count=2)
    permutation = minibatch_permutation(seeds["minibatch_schedule"], 8)
    config = production_config()
    config.update({"steps": 1, "microbatch": 1, "accumulation": 1, "eval_every": 1})
    arms = {}
    with tempfile.TemporaryDirectory(prefix="cmr1_TEST_ONLY_entrypoint_") as temporary:
        for arm in ARMS:
            student = cmr1_student.build_student(seed=seeds["student_init"])
            folder = Path(temporary) / arm
            result = train_arm(student, teacher, train[arm], dev[arm], permutation,
                               config, "cuda", folder, test_only=True)
            curve = json.loads((folder / "curve.jsonl").read_text(encoding="utf-8").strip())
            _need(result["optimizer_steps"] == curve["step"] == 1 and
                  result["test_only"] is True and (folder / "terminal.pt").is_file() and
                  all(math.isfinite(curve[key]) for key in
                      ("loss", "gradient_norm", "synthetic_dev_nmse")),
                  "TEST_ONLY_ENTRYPOINT_SMOKE_MISMATCH")
            arms[arm] = {"optimizer_steps": 1, "loss": curve["loss"],
                         "gradient_norm": curve["gradient_norm"],
                         "checkpoint_sha256": result["checkpoint_sha256"],
                         "checkpoint_path_verified": str(folder / "terminal.pt"),
                         "test_only": True}
            del student
            torch.cuda.empty_cache()
    return {"state": "TEST_ONLY_SMOKE_PASS", "teacher_wrapper": type(teacher).__name__,
            "donor_evidence": {"train": train_evidence, "dev": dev_evidence},
            "arms": arms, "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "gpu_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "formal_training": "NOT_STARTED"}


def run_campaign(teacher_root: Path, output_root: Path) -> None:
    report = launch_preflight(teacher_root, output_root)
    print(json.dumps({"preflight": "PASS", **report}, sort_keys=True), flush=True)
    teacher = load_pinned_teacher_wrapper(teacher_root, "cuda")
    _, _, roles = load_frozen_campaign()
    output_root.mkdir(parents=True, exist_ok=False)
    for replicate in range(1, 49):
        run_replicate(teacher, roles, replicate, output_root, "cuda")
