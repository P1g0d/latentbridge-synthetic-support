"""Stage B full-block trainer (pre-freeze hardened).

Hardening over the first pre-protocol version:
- MANDATORY frozen target identity: scientific training mode REQUIRES an
  `expected_target_identity` artifact (created outside the trainer); missing → raise before
  optimizer construction. Unit tests may pass
  `unsafe_skip_target_verification_for_test_only=True` (default False; production runners
  must never set it).
- Independent verification: the trainer receives the frozen TargetIdentityArtifact and
  verifies the ACTUAL target against it (positive identity: parameter/probe-input/probe-
  output hashes and name) — before optimizer step 1 and again after training.
- Call accounting: `verification_probe_calls` (verifier bookkeeping) vs
  `training_target_calls` (scientific loss evaluations) are counted separately.
- Identity-bound resumable checkpoints: schedule artifact SHA-256, train/dev pool hashes,
  target identity artifact SHA-256, config SHA-256, horizon, warmup, final LR, LR phase,
  constant_lr, global step, schedule position, optimizer/model state, RNG states — resume
  refuses any identity mismatch.
- True RNG restore on resume (outside any fork scope), with a controlled-random-op test.
- LR phases: "warmup_cosine" and "fixed_lr_extension" (a 10k checkpoint resumed into the
  20k extension keeps global step/optimizer/schedule and switches phase explicitly).
- Loss logging: `loss_micro_sum` and `loss_effective_mean` (= micro_sum / accumulation).
"""
import hashlib
import json
import os
import pickle
import time
from pathlib import Path
import torch

from .full_block import state_hash_of, verify_target_identity
from .records import dump, file_hash, state_hash
from .seed_plumbing import rng_state_snapshots, tensor_sha256


def learning_rate(step, peak, final_ratio, warmup, horizon):
    final = peak * final_ratio
    if step <= warmup:
        return peak * step / warmup
    progress = (step - warmup) / max(1, horizon - warmup)
    return final + 0.5 * (peak - final) * (1 + torch.cos(torch.tensor(3.141592653589793 * progress))).item()


def stable_tensor_sha256(tensor):
    """Device-agnostic stable tensor hash (CUDA tensors are copied to CPU first)."""
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


@torch.no_grad()
def evaluate_full_block(student, target, pool, device, microbatch=8):
    """Pooled energy NMSE in U-space plus secondary statistics."""
    was_training = student.training
    student.eval()
    numerator = denominator = 0.0
    student_energy = 0.0
    for start in range(0, len(pool), microbatch):
        hidden = pool[start:start + microbatch].to(device)
        update_target = target(hidden)
        update_student = student(hidden)
        numerator += float((update_student.double() - update_target.double()).square().sum())
        denominator += float(update_target.double().square().sum())
        student_energy += float(update_student.double().square().sum())
    if was_training:
        student.train()
    return {"nmse": numerator / denominator, "numerator": numerator, "denominator": denominator,
            "energy_norm_ratio": (student_energy / denominator) ** 0.5, "sequences": int(len(pool))}


def _restore_rng(payload):
    """TRUE RNG restore (outside any fork scope): the checkpoint's Python / torch-CPU /
    CUDA RNG states become the live states of the training process."""
    import random as random_module
    snapshots = payload["rng_snapshots"]
    random_module.setstate(pickle.loads(snapshots["python_random_state"]))
    torch.set_rng_state(snapshots["torch_cpu_rng_state"].cpu().to(torch.uint8))
    if torch.cuda.is_available() and "torch_cuda_rng_state" in snapshots:
        torch.cuda.set_rng_state(snapshots["torch_cuda_rng_state"].cpu().to(torch.uint8))


def _checkpoint_payload(student, optimizer, global_step, target, schedule_position,
                        init_provenance, lr_phase, identity_bindings, horizon):
    snapshots, rng_hashes = rng_state_snapshots()
    return {"model_state": student.state_dict(), "optimizer_state": optimizer.state_dict(),
            "global_step": global_step, "lr_phase": lr_phase,
            "schedule_position": schedule_position,
            "init_provenance": init_provenance,
            "target_identity": {"name": target.name, "role": target.role,
                                "parameter_state_hash": target.parameter_state_hash},
            "identity_bindings": identity_bindings,
            "horizon": horizon, "constant_lr": identity_bindings["constant_lr"],
            "config_sha256": identity_bindings["config_sha256"],
            "rng_snapshots": snapshots, "rng_hashes": rng_hashes,
            "student_state_hash": state_hash(student),
            "counters": dict(target.counters)}


def save_checkpoint(path, payload):
    temporary = Path(str(path) + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def train_full_block(student, target, config, folder, train_pool, dev_pool, device, schedule,
                     total_steps=None, constant_lr=None, init_provenance=None,
                     expected_target_identity=None, target_identity_artifact_path=None,
                     schedule_artifact=None, resume=None, teacher_block=None, start_step=None,
                     horizon=None, unsafe_skip_target_verification_for_test_only=False,
                     probe_records=None, extension_authorization=None):
    """Train the full-block Student under the frozen recipe with hardened provenance.

    `expected_target_identity` (a frozen TargetIdentityArtifact dict, generated outside the
    trainer) is REQUIRED for scientific training mode; omitting it raises before optimizer
    construction. Unit tests may pass
    `unsafe_skip_target_verification_for_test_only=True` explicitly."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=False)
    dump(folder / "config.json", config)
    steps = int(total_steps if total_steps is not None else config["steps"])
    peak_lr = float(config["lr_peak"])
    microbatch = int(config["microbatch"])
    accumulation = int(config["accumulation"])

    # ---- preflight C: target identity artifact file vs metadata ---------------------------
    if expected_target_identity is not None and target_identity_artifact_path is not None:
        artifact_file = Path(target_identity_artifact_path)
        if not artifact_file.exists():
            raise RuntimeError("TARGET IDENTITY ARTIFACT BINDING FAILED: "
                               f"{artifact_file} does not exist")
        on_disk = json.loads(artifact_file.read_text(encoding="utf-8"))
        if on_disk != expected_target_identity:
            raise RuntimeError("TARGET IDENTITY ARTIFACT BINDING FAILED: the artifact file content "
                               "differs from the expected identity passed to the trainer")

    # ---- mandatory frozen target identity (E/F) ----------------------------------------
    if expected_target_identity is None and not unsafe_skip_target_verification_for_test_only:
        raise RuntimeError("expected_target_identity is REQUIRED for scientific training "
                           "(frozen TargetIdentityArtifact missing); pass the frozen artifact "
                           "or the explicit test-only bypass")

    # ---- preflight C: schedule artifact binding (metadata vs actual artifact) -------------
    # Two independent requirements: (1) the artifact FILE content must match the claimed
    # metadata hash (catches "metadata says A, execution uses B"); (2) the schedule actually
    # fed to training must be a contiguous slice of that frozen artifact (phase A uses a
    # prefix; the phase-B extension uses the continuation).
    if isinstance(schedule_artifact, dict) and schedule_artifact.get("schedule_sha256"):
        artifact_path = schedule_artifact.get("path")
        if not artifact_path or not Path(artifact_path).exists():
            raise RuntimeError(f"SCHEDULE ARTIFACT BINDING FAILED: artifact file missing: "
                               f"{artifact_path!r}")
        artifact_payload = torch.load(artifact_path, map_location="cpu", weights_only=True)
        actual_artifact_sha = stable_tensor_sha256(artifact_payload)
        if actual_artifact_sha != schedule_artifact["schedule_sha256"]:
            raise RuntimeError("SCHEDULE ARTIFACT BINDING FAILED: metadata says "
                               f"{str(schedule_artifact['schedule_sha256'])[:16]} but the artifact "
                               f"file hashes to {actual_artifact_sha[:16]}")
        used = (torch.stack(schedule) if torch.is_tensor(schedule[0])
                else torch.tensor(schedule, dtype=torch.int64))
        total, count = artifact_payload.shape[0], used.shape[0]
        candidates = [0, max(0, total - count)] if total >= count else []
        matches = any(torch.equal(used, artifact_payload[offset:offset + count]) for offset in candidates)
        if not matches and total >= count:
            matches = any(torch.equal(used, artifact_payload[offset:offset + count])
                          for offset in range(0, total - count + 1))
        if not matches:
            raise RuntimeError("SCHEDULE ARTIFACT BINDING FAILED: the schedule actually used is not "
                               "a contiguous slice of the frozen artifact")

    # ---- identity bindings + resume restore (I/J) ---------------------------------------
    identity_bindings = {
        "schedule_artifact_sha256": (schedule_artifact or {}).get("schedule_sha256"),
        "train_pool_sha256": tensor_sha256(train_pool),
        "dev_pool_sha256": tensor_sha256(dev_pool),
        "target_identity_artifact_sha256": (file_hash(target_identity_artifact_path)
                                            if target_identity_artifact_path else None),
        "config_sha256": hashlib.sha256(json.dumps(
            {key: value for key, value in config.items() if key != "steps"},
            sort_keys=True).encode()).hexdigest(),
        "horizon": horizon if horizon is not None else (start_step or 0) + steps,
        "warmup": int(config["warmup"]),
        "final_lr": peak_lr * float(config["lr_final_ratio"]),
        "constant_lr": constant_lr,
    }
    if resume is not None:
        payload = torch.load(resume, map_location="cpu", weights_only=False)
        bindings = payload["identity_bindings"]
        mismatches = [key for key in ("schedule_artifact_sha256", "train_pool_sha256",
                                      "dev_pool_sha256", "target_identity_artifact_sha256",
                                      "config_sha256", "horizon", "warmup", "final_lr")
                      if bindings.get(key) != identity_bindings.get(key)]
        if identity_bindings["constant_lr"] is None and payload.get("lr_phase") == "fixed_lr_extension":
            mismatches.append("cannot resume a fixed_lr_extension checkpoint into a cosine phase")
        if mismatches:
            raise RuntimeError("CHECKPOINT IDENTITY MISMATCH ON RESUME: " + "; ".join(mismatches))
        # ---- preflight B: extension legality (audit §9B) --------------------------------
        if payload.get("lr_phase") == "warmup_cosine" and constant_lr is not None:
            if extension_authorization is None:
                raise RuntimeError("EXTENSION NOT AUTHORIZED: resuming into fixed_lr_extension "
                                   "requires an explicit global EXTEND decision record")
            if extension_authorization.get("decision") != "EXTEND_ALL_9_TO_20K":
                raise RuntimeError("EXTENSION NOT AUTHORIZED: global decision is "
                                   f"{extension_authorization.get('decision')!r}, not EXTEND_ALL_9_TO_20K")
            if abs(float(constant_lr) - float(extension_authorization.get("extension_lr", 1e-4))) > 0:
                raise RuntimeError("EXTENSION LR MISMATCH: "
                                   f"{constant_lr} != {extension_authorization.get('extension_lr')}")
            # The formal phase-A endpoint is 10,000. Shortened-horizon state-machine tests may
            # name a different endpoint, but only through an explicitly named unsafe flag.
            phase_a_step = int(extension_authorization.get("phase_a_step", 10000))
            if phase_a_step != 10000 and not extension_authorization.get(
                    "unsafe_shortened_phase_a_for_test_only", False):
                raise RuntimeError("EXTENSION PHASE-A STEP OVERRIDE REQUIRES "
                                   "unsafe_shortened_phase_a_for_test_only=True (test-only)")
            if int(payload["global_step"]) != phase_a_step:
                raise RuntimeError("EXTENSION MUST START AT GLOBAL STEP "
                                   f"{phase_a_step}, got {payload['global_step']}")
        student.load_state_dict(payload["model_state"])
        student = student.to(device)
        start_step = int(payload["global_step"])
        schedule_position = int(payload["schedule_position"])
        lr_phase = payload["lr_phase"]
        if constant_lr is not None and lr_phase != "fixed_lr_extension":
            lr_phase = "fixed_lr_extension"     # explicit 10k-cosine -> extension transition
        init_provenance = payload["init_provenance"]
        _restore_rng(payload)                   # TRUE restore, outside any fork scope
        optimizer = torch.optim.AdamW(student.parameters(), lr=1.0,
                                      betas=tuple(config["betas"]), eps=float(config["eps"]),
                                      weight_decay=float(config["weight_decay"]))
        optimizer.load_state_dict(payload["optimizer_state"])
        print(f"{folder.name}: resumed from {resume} at global step {start_step} "
              f"(lr_phase={lr_phase})", flush=True)
    else:
        optimizer = torch.optim.AdamW(student.parameters(), lr=peak_lr,
                                      betas=tuple(config["betas"]), eps=float(config["eps"]),
                                      weight_decay=float(config["weight_decay"]))
        start_step = int(start_step or 0)
        lr_phase = "constant" if constant_lr is not None else "warmup_cosine"
        schedule_position = 0

    student = student.to(device)
    student_state_hash = state_hash(student)

    # ---- step-0 provenance (before optimizer step 1) ------------------------------------
    probe_batch = train_pool[:2].to(device)
    snapshots, rng_hashes = rng_state_snapshots()
    torch.save(snapshots, folder / "step0_rng_snapshots.pt")
    optimizer_state_sha256 = ("fresh-empty" if resume is None else
                              hashlib.sha256(pickle.dumps(optimizer.state_dict())).hexdigest())
    step0_provenance = {
        "model_parameter_sha256": student_state_hash,
        "init_provenance": init_provenance,
        "optimizer": {"class": "AdamW", "betas": list(config["betas"]), "eps": config["eps"],
                      "weight_decay": config["weight_decay"], "peak_lr": peak_lr,
                      "initial_state_provenance": "fresh AdamW (empty exp-avg state)"
                                                   if resume is None else f"loaded from {resume}",
                      "initial_state_sha256": optimizer_state_sha256},
        "train_pool_sha256": tensor_sha256(train_pool),
        "dev_pool_sha256": tensor_sha256(dev_pool),
        "target_parameter_state_hash": target.parameter_state_hash,
        "target_probe_output_sha256": target.output_sha256(probe_batch),
        "schedule_artifact": schedule_artifact,
        "first_effective_batches": [list(map(int, schedule[i]))
                                    for i in range(min(5, len(schedule)))],
        "lr_schedule": {"peak": peak_lr, "final_ratio": config["lr_final_ratio"],
                        "warmup": config["warmup"], "constant": constant_lr,
                        "horizon": identity_bindings["horizon"], "terminal_step": steps},
        "lr_phase": lr_phase,
        "precision": {"dtype": "FP32", "amp": False, "tf32": False},
        "deterministic": {"cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
                          "use_deterministic_algorithms": torch.are_deterministic_algorithms_enabled()},
        "code_commit": _git_commit(),
        "rng_state_hashes": rng_hashes,
        "resume_from": str(resume) if resume is not None else None,
        "target_identity_verified": expected_target_identity is not None}
    dump(folder / "step0_provenance.json", step0_provenance)

    # ---- hard frozen-target-identity verification BEFORE optimizer step 1 (E/F/G) --------
    if expected_target_identity is not None:
        verdict = verify_target_identity(target, expected_target_identity, probe_batch)
        if not verdict["passed"]:
            raise RuntimeError("TARGET IDENTITY VERIFICATION FAILED BEFORE STEP 1: "
                               + "; ".join(verdict["failures"]))
        if teacher_block is not None and expected_target_identity["role"] == \
                "qualification_diagnostic_target" and \
                expected_target_identity["parameter_sha256"] == state_hash_of(teacher_block):
            raise RuntimeError("proxy diagnostic wired the full Teacher block as its target")
    else:
        assert unsafe_skip_target_verification_for_test_only

    optimizer.zero_grad(set_to_none=True)
    save_checkpoint(folder / "last.pt", _checkpoint_payload(
        student, optimizer, start_step, target, schedule_position, init_provenance, lr_phase,
        identity_bindings, identity_bindings["horizon"]))
    start = time.perf_counter()
    clipped = 0
    target.set_mode("training")
    with (folder / "curve.jsonl").open("w", encoding="utf-8") as log:
        for offset in range(1, steps + 1):
            step = start_step + offset
            lr = (float(constant_lr) if constant_lr is not None else
                  learning_rate(step, peak_lr, float(config["lr_final_ratio"]),
                                int(config["warmup"]), identity_bindings["horizon"]))
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.zero_grad(set_to_none=True)
            indices = schedule[schedule_position]
            schedule_position += 1
            loss_micro_sum = 0.0
            for micro in range(accumulation):
                selection = indices[micro * microbatch:(micro + 1) * microbatch]
                hidden = train_pool[selection].to(device)
                with torch.no_grad():
                    update_target = target(hidden)
                update_student = student(hidden)
                loss = (update_student - update_target).square().mean() / accumulation
                if not torch.isfinite(loss):
                    raise RuntimeError(f"Non-finite loss {float(loss):g} at micro {micro}, step {step}")
                loss.backward()
                loss_micro_sum += float(loss.detach()) * accumulation
            loss_effective_mean = loss_micro_sum / accumulation
            norm = torch.nn.utils.clip_grad_norm_(student.parameters(), float(config["clip_norm"]),
                                                  error_if_nonfinite=True)
            optimizer.step()
            clipped += norm.item() > float(config["clip_norm"])
            row = {"step": step, "loss_micro_sum": loss_micro_sum,
                   "loss_effective_mean": loss_effective_mean,
                   "gradient_norm": norm.item(), "lr": lr}
            if probe_records is not None and str(step) in probe_records:
                record = probe_records[str(step)]
                snapshot = {name: tensor.detach().cpu().clone()
                            for name, tensor in student.state_dict().items()}
                torch.save(snapshot, folder / f"params_step{step}.pt")
                record.update({"loss_effective_mean": loss_effective_mean,
                               "loss_micro_sum": loss_micro_sum,
                               "gradient_norm": norm.item(), "lr": lr,
                               "parameter_state_sha256": state_hash(student)})
            if offset == 1 or offset % int(config["eval_every"]) == 0 or offset == steps:
                target.set_mode("verification")
                row["dev"] = evaluate_full_block(student, target, dev_pool, device, microbatch)
                target.set_mode("training")
                save_checkpoint(folder / "last.pt", _checkpoint_payload(
                    student, optimizer, step, target, schedule_position, init_provenance,
                    lr_phase, identity_bindings, identity_bindings["horizon"]))
                print(f"{folder.name} step={step} dev_nmse={row['dev']['nmse']:.6g} "
                      f"lr={lr:.3g} seconds={time.perf_counter() - start:.1f}", flush=True)
            log.write(json.dumps(row) + "\n")
            if "dev" in row:
                log.flush()
    target.set_mode("verification")
    final_dev = evaluate_full_block(student, target, dev_pool, device, microbatch)
    summary = {"status": "complete", "steps_requested": steps,
               "global_step_range": [start_step + 1, start_step + steps],
               "final_dev": final_dev, "clipping_fraction": clipped / steps,
               "seconds": time.perf_counter() - start,
               "student_state_hash": state_hash(student),
               "lr_phase": lr_phase,
               "target_record": {"target_name": target.name, "role": target.role,
                                 "parameter_state_hash": target.parameter_state_hash,
                                 "counters": dict(target.counters),
                                 "probe_output_sha256": step0_provenance["target_probe_output_sha256"]},
               "step0_provenance": step0_provenance}
    dump(folder / "status.json", summary)
    return summary


def _git_commit():
    import subprocess
    output = subprocess.run(("git", "-c", f"safe.directory={Path.cwd().as_posix()}",
                             "rev-parse", "HEAD"), capture_output=True, text=True)
    return output.stdout.strip()
