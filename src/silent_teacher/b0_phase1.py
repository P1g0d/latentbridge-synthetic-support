"""Stage B0 Phase 1 shared scientific helpers.

Pure, unit-tested functions for the frozen rules of
`protocols/stage_b0_attention_feasibility_v1.md` (§9 extension rule, §12 context test)
as amended by v2. Kept out of the run scripts so the decisions are testable without a GPU.
"""
import json
from pathlib import Path
import torch


def extension_decision(dev_nmse_early, dev_nmse_terminal, threshold=0.10):
    """Frozen extension rule: relative improvement between the 8k and 10k dev evaluations.

    The rule is evaluated only when the terminal (10k) dev NMSE is still above the
    representable gate; the caller applies that condition.
    """
    relative_improvement = (dev_nmse_early - dev_nmse_terminal) / dev_nmse_early
    return {"relative_improvement": relative_improvement, "triggered": bool(relative_improvement >= threshold)}


def derangement(size, rng):
    """Random permutation with no fixed points, deterministic given the generator.

    The fixed-point-removal swaps make this *not* exactly uniform over derangements; the
    protocol only requires a fixed-point-free permutation, which is what this guarantees.
    """
    permutation = torch.randperm(size, generator=rng)
    index = torch.arange(size)
    while True:
        fixed = (permutation == index).nonzero().flatten().tolist()
        if not fixed:
            return permutation
        for position in fixed:
            other = (position + 1) % size
            permutation[position], permutation[other] = permutation[other].clone(), permutation[position].clone()


def splice_history(hidden, donor_indices, query):
    """Strict context splice: positions [0, query) from the donor.

    `hidden[query]` and `hidden[query+1:]` are left bit-identical to the original —
    inclusive slicing or any ambiguity is forbidden by the frozen protocol.
    """
    spliced = hidden.clone()
    spliced[:, :query, :] = hidden[donor_indices][:, :query, :]
    return spliced


def splice_future(hidden, donor_indices, query):
    """Causality negative control: only positions [query+1, L) come from the donor."""
    spliced = hidden.clone()
    spliced[:, query + 1:, :] = hidden[donor_indices][:, query + 1:, :]
    return spliced


def pooled_nmse(delta_student, delta_teacher):
    """ContextNMSE = Σ||ΔS − ΔT||² / Σ||ΔT||² over pooled deltas (never per-token averaging)."""
    numerator = float((delta_student.double() - delta_teacher.double()).square().sum())
    denominator = float(delta_teacher.double().square().sum())
    student_energy = float(delta_student.double().square().sum())
    return {"numerator": numerator, "denominator": denominator,
            "nmse": numerator / denominator if denominator > 0 else None,
            "delta_energy_ratio": student_energy / denominator if denominator > 0 else None}


def position_buckets(length=128):
    """Frozen position buckets of protocol §11, clipped to `length`.

    The protocol defines the buckets at `L = 128` (0-31 / 32-63 / 64-127). Shorter
    sequences (unit tests) yield only the buckets that intersect `[0, length)`.
    """
    frozen = {"0-31": slice(0, 32), "32-63": slice(32, 64), "64-127": slice(64, 128)}
    return {name: slice(window.start, min(window.stop, length))
            for name, window in frozen.items() if window.start < length}


@torch.no_grad()
def stream_evaluation(forward, pool, microbatch=8):
    """Pooled value metrics plus frozen position-bucket NMSE, accumulated streaming.

    `forward(hidden)` receives a CPU batch of sequences and returns `(target, output)` for
    it (the caller owns device placement). Pooling is always Σ||S−T||² / Σ||T||² over the
    whole split — never an average of per-batch or per-token NMSEs. Cosine excludes tokens
    with a near-zero norm on either side, and the error quantiles are over per-token
    squared error sums, exactly as in `b0_train.evaluate`.
    """
    buckets = position_buckets(pool.shape[1])
    totals = {name: {"numerator": 0.0, "denominator": 0.0} for name in buckets}
    numerator = denominator = 0.0
    student_energy = 0.0
    cosine_sum = 0.0
    tokens = 0
    errors = []
    for start in range(0, len(pool), microbatch):
        hidden = pool[start:start + microbatch]
        target, output = forward(hidden)
        difference = (output - target).double()
        numerator += float(difference.square().sum())
        denominator += float(target.double().square().sum())
        student_energy += float(output.double().square().sum())
        flat_output = output.reshape(-1, output.shape[-1]).double()
        flat_target = target.reshape(-1, target.shape[-1]).double()
        output_norm = flat_output.norm(dim=-1)
        target_norm = flat_target.norm(dim=-1)
        valid = (output_norm >= 1e-8) & (target_norm >= 1e-8)
        if valid.any():
            cosine_sum += float((flat_output[valid] * flat_target[valid]).sum(dim=-1).div(
                (output_norm[valid] * target_norm[valid])).sum())
            tokens += int(valid.sum())
        errors.append(difference.square().sum(-1).flatten().cpu())
        squared = difference.square()
        target_squared = target.double().square()
        for name, window in buckets.items():
            totals[name]["numerator"] += float(squared[:, window, :].sum())
            totals[name]["denominator"] += float(target_squared[:, window, :].sum())
    errors = torch.cat(errors)
    return {"nmse": numerator / denominator, "numerator": numerator, "denominator": denominator,
            "raw_mse": numerator / (len(pool) * pool.shape[1] * pool.shape[2]),
            "cosine": cosine_sum / tokens if tokens else None, "cosine_excluded": int(len(errors) - tokens),
            "energy_norm_ratio": (student_energy / denominator) ** 0.5,
            "error_p50": float(errors.quantile(0.5)), "error_p90": float(errors.quantile(0.9)),
            "error_p99": float(errors.quantile(0.99)), "sequences": int(len(pool)),
            "buckets": {name: {**values, "nmse": values["numerator"] / values["denominator"]}
                        for name, values in totals.items()}}


@torch.no_grad()
def response_deltas(forward, pool, donor, queries, microbatch=8):
    """Δ = f(spliced history)_t − f(H)_t at each query position, for the frozen splice.

    Returns one [N, width] tensor per query position; position `t` of the spliced input is
    bit-identical to the original (asserted) and positions `[t+1, L)` are unchanged.
    """
    deltas, baseline = {}, {}
    for query in queries:
        spliced = splice_history(pool, donor, query)
        assert torch.equal(spliced[:, query], pool[:, query]), "query position must stay bit-identical"
        assert torch.equal(spliced[:, query + 1:], pool[:, query + 1:]), "future positions must be unchanged"
        base_chunks, spliced_chunks = [], []
        for start in range(0, len(pool), microbatch):
            base_chunks.append(forward(pool[start:start + microbatch])[:, query].cpu())
            spliced_chunks.append(forward(spliced[start:start + microbatch])[:, query].cpu())
        base = torch.cat(base_chunks)
        baseline[query] = base
        deltas[query] = torch.cat(spliced_chunks) - base
    return deltas, baseline


@torch.no_grad()
def future_change_at_query(forward, pool, donor, queries, microbatch=8):
    """Causality negative control: max abs change at `t` when only [t+1, L) is replaced.

    Note: `t = L-1` replaces the empty slice `[L, L)` — the input is bit-identical and the
    0.0 result is true by construction, so such a query is a no-op and provides no evidence
    (only strictly interior `t < L-1` interventions are informative).
    """
    worst = {}
    for query in queries:
        spliced = splice_future(pool, donor, query)
        assert torch.equal(spliced[:, :query + 1], pool[:, :query + 1]), "history must be unchanged"
        assert not torch.equal(spliced[:, query + 1:], pool[:, query + 1:]) or query + 1 >= pool.shape[1]
        maximum = 0.0
        for start in range(0, len(pool), microbatch):
            base = forward(pool[start:start + microbatch])[:, query]
            changed = forward(spliced[start:start + microbatch])[:, query]
            maximum = max(maximum, float((base - changed).abs().max()))
        worst[query] = maximum
    return worst


def read_curve(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def curve_rows(rows):
    """Compact per-evaluation records from a saved training curve."""
    return [{"step": row["step"], "dev_nmse": row["dev"]["nmse"], "dev_cosine": row["dev"]["cosine"],
             "loss": row["loss"], "gradient_norm": row["gradient_norm"], "lr": row["lr"],
             "train_nmse": (row["train"]["nmse"] if "train" in row else None)}
            for row in rows if "dev" in row]


def gradient_summary(rows, window=50):
    first = [row["gradient_norm"] for row in rows[:window]]
    last = [row["gradient_norm"] for row in rows[-window:]]
    return {"first_mean": sum(first) / len(first), "last_mean": sum(last) / len(last),
            "max": max(row["gradient_norm"] for row in rows)}
