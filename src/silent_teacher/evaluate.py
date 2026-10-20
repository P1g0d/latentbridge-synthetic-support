import numpy as np
import torch


@torch.no_grad()
def evaluate(model, teacher, inputs, device, output=None):
    rows = []
    for x in inputs.split(1024):
        t = teacher(x.to(device)).double()
        s = model(x.to(device)).double()
        error = (s - t).square().sum(-1)
        tn, sn = t.norm(dim=-1), s.norm(dim=-1)
        valid = (tn >= 1e-8) & (sn >= 1e-8)
        cosine = (s * t).sum(-1) / (tn * sn).clamp_min(1e-300)
        rows.append(torch.stack((error, tn.square(), sn.square(), cosine, valid.double()), -1).cpu())
    a = torch.cat(rows).numpy()
    if output:
        np.savez_compressed(output, statistics=a)
    return summarize(a, inputs.shape[-1])


def summarize(a, width):
    valid = a[:, 4].astype(bool)
    return {
        "n": len(a), "mse": float(a[:, 0].mean() / width),
        "nmse": float(a[:, 0].sum() / (a[:, 1].sum() + 1e-12)),
        "cosine": float(a[valid, 3].mean()) if valid.any() else None,
        "cosine_excluded": int((~valid).sum()),
        "teacher_norm_mean": float(np.sqrt(a[:, 1]).mean()),
        "student_norm_mean": float(np.sqrt(a[:, 2]).mean()),
        "energy_norm_ratio": float(np.sqrt(a[:, 2].sum() / (a[:, 1].sum() + 1e-12))),
        "squared_error_quantiles": dict(zip(["p50", "p90", "p99"], np.quantile(a[:, 0], [.5, .9, .99]).tolist())),
    }
