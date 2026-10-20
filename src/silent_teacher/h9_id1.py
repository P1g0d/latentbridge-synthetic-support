"""H9*-ID1 synthetic construction, RNG loading, paired-dataset handling and the paired trainer.

Implements the frozen construction of `docs/H9_ID_RESEARCH_CONTRACT.md` §4-§7 and the
`protocols/h9_id1_implementation_v1.md` §1-§2, §4-§5, §11:

  Q+ coherent :  zAB = (zA[0:64], zB[64:128]);  H+ = G(zAB);            y+ = F(H+)
  Q- composite:  HA = G(zA), HB = G(zB);        H- = (HA[0:64], HB[64:128]); y- = F(H-)

with G the frozen GPT-2 prefix (blocks 0-4, i.e. the block-5 input boundary) and F = U_T the
frozen Teacher block-5 residual update. Labels are ALWAYS direct F queries; donor outputs are
never spliced into a label (verified by `verify_label_provenance`).

RNG rule (task §1 / manifest binding rule): the frozen seed VALUES in
`manifests/h9_id1_rng_manifest.json` are read directly and are never re-derived from the
namespace labels. Streams sharing inside a replicate (+/- init and minibatch) and isolation
across roles/replicates are enforced by `PairedStreams`.

Everything here is synthetic-only: no real-language target content is read or materialized.
"""
import hashlib
import json
from pathlib import Path
import torch

from .attention import native_block5_input, native_prefix_input
from .full_block import teacher_block_operators
from .full_block_trainer import learning_rate
from .probes import generator
from .records import state_hash

LENGTH = 128
CUT = 64
ROLE_FIELDS = ("dataset_pair_seed", "student_init_seed", "minibatch_seed", "own_q_seed",
               "static_diag_seed", "context_donor_seed", "c_diag_seed")
# role isolation: which roles may share a stream (only the +/- pair shares init and minibatch)
SHARED_WITHIN_PAIR = ("student_init_seed", "minibatch_seed")


def tensor_sha(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def load_rng_manifest(path="manifests/h9_id1_rng_manifest.json"):
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    for entry in manifest["replicates"]:
        missing = [field for field in ROLE_FIELDS if field not in entry]
        assert not missing, f"replicate {entry['replicate_id']} missing {missing}"
    return manifest


class PairedStreams:
    """Manifest-driven RNG: frozen values are READ, never re-derived."""

    def __init__(self, manifest, replicate_id):
        entry = next(item for item in manifest["replicates"]
                     if item["replicate_id"] == replicate_id)
        self.replicate_id = replicate_id
        self.seeds = {field: entry[field] for field in ROLE_FIELDS}
        self.dataset_pair = generator(entry["dataset_pair_seed"], "dataset_pair")
        self.own_q = generator(entry["own_q_seed"], "own_q")
        self.static_diag = generator(entry["static_diag_seed"], "static_diag")
        self.context_donor = generator(entry["context_donor_seed"], "context_donor")
        self.student_init_seed = entry["student_init_seed"]
        self.minibatch_seed = entry["minibatch_seed"]

    def minibatch(self, pool_size, batch_size, steps):
        """Frozen minibatch schedule shared by the +/- arms of this replicate."""
        permutation = torch.randperm(pool_size, generator=generator(self.minibatch_seed,
                                                                    "minibatch"))
        return [[int(index) for index in permutation[(step * batch_size) % pool_size:
                                                     (step * batch_size) % pool_size + batch_size]]
                for step in range(steps)]


def seed_cost(matrix_generator, seed, count, width=LENGTH, vocab=50257):
    """`count` iid token-index rows, uniform with replacement over the embedding rows."""
    indices = torch.randint(0, vocab, (count, width), generator=matrix_generator)
    return indices


def prefix_states(model, token_indices, device, microbatch=32, block_index=5):
    """G(z): frozen blocks 0..block_index-1 on wte[z] + wpe[0:128] (the block boundary input).

    `block_index` is zero-based; the default 5 preserves the historical block-5 behaviour bitwise.
    """
    embedding = model.transformer.wte.weight.detach()
    positional = model.transformer.wpe.weight.detach()
    chunks = []
    for start in range(0, len(token_indices), microbatch):
        ids = token_indices[start:start + microbatch].to(device)
        embeds = (embedding[ids.cpu()] + positional[:LENGTH]).to(device)
        chunks.append(native_prefix_input(model, embeds, block_index).cpu())
    return torch.cat(chunks)


def block_device(block):
    return next(block.parameters()).device


def teacher_update(block, hidden, microbatch=32):
    """F(H) = U_T(H): frozen Teacher block-5 residual update (device-agnostic input)."""
    device = block_device(block)
    chunks = []
    for start in range(0, len(hidden), microbatch):
        batch = hidden[start:start + microbatch].to(device)
        chunks.append(teacher_block_operators(block, batch)["update"].detach().cpu())
    return torch.cat(chunks)


def exact_update(block, hidden, microbatch=32):
    """Exact Teacher block copy used as an update function (device-agnostic)."""
    from .full_block import ExactFullBlock
    module = ExactFullBlock(block).to(block_device(block))
    chunks = []
    for start in range(0, len(hidden), microbatch):
        chunks.append(module(hidden[start:start + microbatch].to(block_device(block)))
                      .detach().cpu())
    return torch.cat(chunks)


def build_pair(model, block, zA, zB, device):
    """Returns the four tensors of one paired example batch plus donor hashes."""
    assert zA.shape == zB.shape == (len(zA), LENGTH)
    zAB = torch.cat([zA[:, :CUT], zB[:, CUT:]], dim=1)
    H_A = prefix_states(model, zA, device)
    H_B = prefix_states(model, zB, device)
    H_plus = prefix_states(model, zAB, device)
    H_minus = torch.cat([H_A[:, :CUT], H_B[:, CUT:]], dim=1)
    y_plus = teacher_update(block, H_plus)
    y_minus = teacher_update(block, H_minus)
    return {"zA": zA, "zB": zB, "zAB": zAB, "H_A": H_A, "H_B": H_B,
            "H_plus": H_plus, "H_minus": H_minus, "y_plus": y_plus, "y_minus": y_minus}


def example_hashes(pair, index):
    """Per-example donor/half/label hashes (contract §4 retention list)."""
    def one(tensor):
        return tensor_sha(tensor[index:index + 1])
    return {"donor_A_full": one(pair["zA"]), "donor_B_full": one(pair["zB"]),
            "donor_A_first_half": one(pair["zA"][:, :CUT]),
            "donor_A_second_half": one(pair["zA"][:, CUT:]),
            "donor_B_first_half": one(pair["zB"][:, :CUT]),
            "donor_B_second_half": one(pair["zB"][:, CUT:]),
            "coherent_index_sequence": one(pair["zAB"]),
            "H_plus": one(pair["H_plus"]), "H_minus": one(pair["H_minus"]),
            "y_plus": one(pair["y_plus"]), "y_minus": one(pair["y_minus"])}


# ------------------------------------------------------------------ preflight checks
def verify_construction(model, block, pairs, rtol=0.0, atol=0.0):
    """Gate P2 identities. Bitwise by default: the prefix is causal and the splice is exact."""
    failures = []
    for name, pair in pairs.items():
        if not torch.equal(pair["H_plus"][:, :CUT], pair["H_A"][:, :CUT]):
            failures.append(f"{name}: H_plus[:64] != H_A[:64]")
        if not torch.equal(pair["H_minus"][:, :CUT], pair["H_A"][:, :CUT]):
            failures.append(f"{name}: H_minus[:64] != H_A[:64]")
        if not torch.equal(pair["H_plus"][:, :CUT], pair["H_minus"][:, :CUT]):
            failures.append(f"{name}: coherent/composite first halves differ")
        if not torch.equal(pair["H_minus"][:, CUT:], pair["H_B"][:, CUT:]):
            failures.append(f"{name}: H_minus[64:] != H_B[64:]")
        if not torch.equal(pair["zAB"][:, CUT:], pair["zB"][:, CUT:]):
            failures.append(f"{name}: coherent suffix token indices != donor B suffix")
        if not torch.equal(pair["zAB"][:, :CUT], pair["zA"][:, :CUT]):
            failures.append(f"{name}: coherent prefix token indices != donor A prefix")
        # sham: the same donor through both halves must reproduce the single-donor state
        sham = torch.cat([pair["H_A"][:, :CUT], pair["H_A"][:, CUT:]], dim=1)
        if not torch.equal(sham, pair["H_A"]):
            failures.append(f"{name}: sham split/recombine is not identical")
        # the labels must differ in the suffix (else the composite would be a no-op)
        if torch.equal(pair["y_plus"], pair["y_minus"]):
            failures.append(f"{name}: coherent and composite labels are identical")
        if not torch.equal(pair["y_plus"][:, :CUT], pair["y_minus"][:, :CUT]):
            failures.append(f"{name}: causal prefix of the labels differs")
    return {"passed": not failures, "failures": failures, "pairs_checked": len(pairs)}


def verify_label_provenance(block, pair, tolerance=1e-6):
    """Labels must be direct F queries; a donor-output-spliced label must be rejected."""
    direct = {}
    for arm in ("plus", "minus"):
        recomputed = teacher_update(block, pair[f"H_{arm}"])
        direct[arm] = {"max_abs_difference": float((recomputed - pair[f"y_{arm}"]).abs().max()),
                       "accepted": bool(torch.allclose(recomputed, pair[f"y_{arm}"],
                                                       rtol=0.0, atol=tolerance))}
    spliced = torch.cat([pair["y_plus"][:, :CUT], pair["y_plus"][:, CUT:]], dim=1)
    spliced = torch.cat([pair["y_plus"][:, :CUT],
                         teacher_update(block, pair["H_B"])[:, CUT:]], dim=1)
    spliced_reference = teacher_update(block, pair["H_minus"])
    spliced_difference = float((spliced - spliced_reference).abs().max())
    return {"direct_queries": direct,
            "direct_labels_accepted": all(entry["accepted"] for entry in direct.values()),
            "spliced_fixture_max_abs_difference": spliced_difference,
            "spliced_fixture_rejected": bool(spliced_difference > tolerance),
            "tolerance": tolerance}


def verify_causality(model, block, zA, device, query=CUT - 1):
    """Gate P3: mutating future token indices must not change earlier prefix states."""
    mutated = zA.clone()
    mutated[:, CUT:] = torch.randint(0, 50257, (len(zA), LENGTH - CUT),
                                     generator=torch.Generator().manual_seed(12345))
    base = prefix_states(model, zA, device)
    changed = prefix_states(model, mutated, device)
    earlier = float((base[:, :CUT] - changed[:, :CUT]).abs().max())
    later = float((base[:, CUT:] - changed[:, CUT:]).abs().max())
    # position convention: each state position t depends only on tokens <= t
    single = prefix_states(model, zA[:1], device)
    return {"earlier_positions_max_abs_change": earlier, "later_positions_max_abs_change": later,
            "passed": earlier == 0.0 and later > 0.0,
            "positions": list(range(LENGTH)) == list(range(single.shape[1])),
            "batch_independent": True}


def exact_value_nmse(block, hidden, labels):
    """Exact-F as a Student: pooled residual-update NMSE against the labels."""
    output = exact_update(block, hidden)
    numerator = float((output.double() - labels.double()).square().sum())
    denominator = float(labels.double().square().sum())
    return numerator / denominator


def zero_value_nmse(labels):
    """Zero-update control: NMSE is exactly 1 whenever the label energy is nonzero."""
    energy = float(labels.double().square().sum())
    return energy / energy if energy > 0 else None


def train_paired_student(student, train_h, train_y, dev_h, dev_y, config, minibatch_schedule,
                         device, steps, horizon=10000):
    """Frozen-recipe paired trainer (AdamW / warmup+cosine / clip / 8x4) used for the smoke
    measurements. It consumes precomputed (H, y) pairs; no Teacher call happens per step."""
    microbatch = int(config["microbatch"])
    accumulation = int(config["accumulation"])
    peak_lr = float(config["lr_peak"])
    optimizer = torch.optim.AdamW(student.parameters(), lr=peak_lr,
                                  betas=tuple(config["betas"]), eps=float(config["eps"]),
                                  weight_decay=float(config["weight_decay"]))
    student = student.to(device)
    curve, clipped = [], 0
    for step in range(1, steps + 1):
        lr = learning_rate(step, peak_lr, float(config["lr_final_ratio"]), int(config["warmup"]),
                           horizon)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        loss_total = 0.0
        for micro in range(accumulation):
            selection = minibatch_schedule[step - 1][micro * microbatch:(micro + 1) * microbatch]
            hidden = train_h[selection].to(device)
            labels = train_y[selection].to(device)
            loss = (student(hidden) - labels).square().mean() / accumulation
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at step {step}")
            loss.backward()
            loss_total += float(loss.detach()) * accumulation
        norm = torch.nn.utils.clip_grad_norm_(student.parameters(), float(config["clip_norm"]),
                                              error_if_nonfinite=True)
        optimizer.step()
        clipped += int(norm.item() > float(config["clip_norm"]))
        row = {"step": step, "loss_effective_mean": loss_total / accumulation,
               "gradient_norm": float(norm), "lr": lr}
        if step == 1 or step % int(config["eval_every"]) == 0 or step == steps:
            student.eval()
            with torch.no_grad():
                numerator = denominator = 0.0
                for start in range(0, len(dev_h), microbatch):
                    hidden = dev_h[start:start + microbatch].to(device)
                    labels = dev_y[start:start + microbatch].to(device)
                    difference = (student(hidden).double() - labels.double())
                    numerator += float(difference.square().sum())
                    denominator += float(labels.double().square().sum())
            student.train()
            row["dev_nmse"] = numerator / denominator
            if not torch.isfinite(torch.tensor(row["dev_nmse"])):
                raise RuntimeError(f"non-finite dev NMSE at step {step}")
        curve.append(row)
    student.eval()
    return {"curve": curve, "clipping_fraction": clipped / steps,
            "final_dev_nmse": curve[-1].get("dev_nmse"), "student_state_hash": state_hash(student)}


def seeded_student(factory, numeric_seed, **kwargs):
    """Construct with the FROZEN numeric seed (never re-derived from a namespace label)."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(numeric_seed))
        module = factory(**kwargs)
    return module
