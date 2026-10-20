import math
import time
from pathlib import Path
import torch
from .evaluate import evaluate
from .probes import generator, gaussian
from .records import dump, state_hash


def setup():
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)


def learning_rate(step, c):
    if step <= c["warmup"]:
        return c["lr"] * step / c["warmup"]
    progress = (step - c["warmup"]) / (c["steps"] - c["warmup"])
    return c["min_lr"] + .5 * (c["lr"] - c["min_lr"]) * (1 + math.cos(math.pi * progress))


def optimizer(model, c):
    return torch.optim.AdamW(model.parameters(), lr=c["lr"], betas=tuple(c["betas"]), eps=c["eps"], weight_decay=c["weight_decay"])


def update(student, teacher, opt, x, lr, clip):
    for group in opt.param_groups:
        group["lr"] = lr
    with torch.no_grad():
        target = teacher(x)
    opt.zero_grad(set_to_none=True)
    loss = (student(x) - target).square().mean()
    if not torch.isfinite(loss):
        raise RuntimeError("Non-finite loss")
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(student.parameters(), clip, error_if_nonfinite=True)
    opt.step()
    return loss.item(), norm.item()


def save_checkpoint(path, student, opt, rng, step, best, c):
    temporary = Path(str(path) + ".tmp")
    torch.save({"student": student.state_dict(), "optimizer": opt.state_dict(),
                "probe_rng": rng.get_state(), "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                "step": step, "best_dev_mse": best, "config": c,
                "schedule": "stateless learning_rate(step,config)"}, temporary)
    temporary.replace(path)


def restore(path, student, opt, rng):
    ck = torch.load(path, map_location="cpu", weights_only=True)
    student.load_state_dict(ck["student"])
    opt.load_state_dict(ck["optimizer"])
    rng.set_state(ck["probe_rng"])
    torch.set_rng_state(ck["torch_rng"])
    if ck["cuda_rng"]:
        torch.cuda.set_rng_state_all(ck["cuda_rng"])
    return ck


def train(student, teacher, c, folder, dev, device, fixed=None, stop_ratio=None, probe=None,
          pool=None, permutation=None):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=False)
    dump(folder / "config.json", c)
    teacher_before = state_hash(teacher)
    opt = optimizer(student, c)
    assert not ({id(p) for p in teacher.parameters()} & {id(p) for g in opt.param_groups for p in g["params"]})
    stream = "train" if probe is None else probe["stream"]
    rng = generator(c["seed"], stream)
    transform = None if probe is None else probe.get("transform")
    initial = evaluate(student, teacher, dev, device)
    best = initial["mse"]
    save_checkpoint(folder / "step0.pt", student, opt, rng, 0, best, c)
    save_checkpoint(folder / "best.pt", student, opt, rng, 0, best, c)
    start = time.perf_counter()
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    clipped, queries, best_step = 0, 0, 0
    validation_queries = len(dev)
    dump(folder / "status.json", {"status": "running"})
    try:
        with (folder / "curve.jsonl").open("w", encoding="utf-8") as log:
            import json
            log.write(json.dumps({"step": 0, "dev": initial}) + "\n")
            for step in range(1, c["steps"] + 1):
                if pool is not None:
                    from .teacher_prefix import batch_indices
                    x = pool[batch_indices(permutation, c["batch_size"], step)]
                else:
                    x = fixed if fixed is not None else gaussian(c["batch_size"], c["width"], rng)
                    if transform is not None and fixed is None:
                        x = transform(x)
                lr = learning_rate(step, c)
                loss, norm = update(student, teacher, opt, x.to(device), lr, c["clip_norm"])
                queries += len(x)
                clipped += norm > c["clip_norm"]
                row = {"step": step, "loss": loss, "gradient_norm": norm, "lr": lr}
                should_stop = False
                if step % c["eval_every"] == 0 or step == c["steps"]:
                    metrics = evaluate(student, teacher, dev, device)
                    validation_queries += len(dev)
                    row["dev"] = metrics
                    if metrics["mse"] < best:
                        best, best_step = metrics["mse"], step
                        save_checkpoint(folder / "best.pt", student, opt, rng, step, best, c)
                    save_checkpoint(folder / "last.pt", student, opt, rng, step, best, c)
                    print(f"{folder.name} step={step} dev_nmse={metrics['nmse']:.6g} ratio={best/initial['mse']:.6g} seconds={time.perf_counter()-start:.1f}", flush=True)
                    should_stop = stop_ratio is not None and best / initial["mse"] <= stop_ratio
                log.write(json.dumps(row, allow_nan=False) + "\n")
                if "dev" in row:
                    log.flush()
                if should_stop:
                    break
        assert state_hash(teacher) == teacher_before
        assert all(p.grad is None and not p.requires_grad for p in teacher.parameters())
        summary = {"status": "complete", "initial_dev": initial, "best_dev_mse": best,
                   "best_step": best_step, "best_initial_ratio": best / initial["mse"],
                   "steps_completed": step, "train_teacher_queries": queries,
                   "validation_teacher_queries": validation_queries, "clipping_fraction": clipped / step,
                   "seconds": time.perf_counter() - start, "teacher_hash": teacher_before,
                   "peak_allocated_bytes": torch.cuda.max_memory_allocated() if device == "cuda" else 0}
        dump(folder / "status.json", summary)
        student.load_state_dict(torch.load(folder / "best.pt", map_location=device, weights_only=True)["student"])
        return summary
    except BaseException as error:
        dump(folder / "status.json", {"status": "failed", "error": repr(error), "seconds": time.perf_counter()-start})
        raise
