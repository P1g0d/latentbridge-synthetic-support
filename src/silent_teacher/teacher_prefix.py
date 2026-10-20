"""Teacher-parameter-derived probe generation for Stage A v3a (G2a / G2b).

Everything here depends only on frozen Teacher parameters, frozen Teacher
computation and synthetic randomness. No tokenizer, no text, no token
frequency, no language data, no validation-derived statistics. The target
boundary is the GPT-2 `transformer.h[5].ln_2` output (post-attention residual),
identical to Language Gate 1; the MLP itself is never executed by the prefix.
"""
import torch

from .probes import generator


def teacher_prefix(model, inputs_embeds):
    """Frozen prefix: dropout, blocks 0-4, block 5 attention + residual, ln_2.

    Returns the MLP input boundary states. The target MLP is deliberately not
    executed here.
    """
    with torch.no_grad():
        hidden = model.transformer.drop(inputs_embeds)
        for index in range(5):
            hidden = model.transformer.h[index](hidden, attention_mask=None, use_cache=False)[0]
        block = model.transformer.h[5]
        attention = block.attn(block.ln_1(hidden), attention_mask=None, layer_past=None,
                               head_mask=None, use_cache=False)
        attention = attention[0] if isinstance(attention, tuple) else attention
        return block.ln_2(hidden + attention)


def teacher_prefix_skip(model, inputs_embeds):
    """Frozen skip prefix (Stage A v3c ablation): blocks 0-4 are skipped; the
    sequence enters block 5 directly through its native attention path —
    ln_1, attention, residual addition — followed by ln_2. Positional
    embeddings must already be present in `inputs_embeds` and are added exactly
    once. The target MLP is never executed."""
    with torch.no_grad():
        hidden = model.transformer.drop(inputs_embeds)
        block = model.transformer.h[5]
        query = block.ln_1(hidden)
        attention = block.attn(query, attention_mask=None, layer_past=None,
                               head_mask=None, use_cache=False)
        attention = attention[0] if isinstance(attention, tuple) else attention
        return block.ln_2(hidden + attention)


def teacher_prefix_by_kind(model, inputs_embeds, kind):
    if kind == "full":
        return teacher_prefix(model, inputs_embeds)
    if kind == "skip":
        return teacher_prefix_skip(model, inputs_embeds)
    raise ValueError(f"Unknown prefix kind: {kind}")


def embedding_statistics(embedding_weight):
    """mu_E and a covariance factor L_E computed ONLY from the frozen Teacher
    embedding parameter matrix (float64). Falls back to an eigen factor if the
    covariance is not positive definite; the fallback flag is returned."""
    weight = embedding_weight.detach().double()
    mu = weight.mean(0)
    centered = weight - mu
    covariance = centered.T @ centered / (weight.shape[0] - 1)
    try:
        factor = torch.linalg.cholesky(covariance)
        fallback = False
    except Exception:
        eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
        factor = eigenvectors * eigenvalues.clamp_min(0.0).sqrt()
        fallback = True
    return mu, factor, {"fallback_used": fallback,
                        "covariance_trace": float(torch.diagonal(covariance).sum())}


def g2_states(model, family, sequences, length, rng, device, embedding_weight, positional_weight,
              mu=None, factor=None, prefix="full"):
    """Generate `sequences * length` MLP-input states for family G2a or G2b."""
    width = embedding_weight.shape[1]
    prefix_fn = teacher_prefix if prefix == "full" else teacher_prefix_skip
    chunks = []
    for _ in range(sequences):
        if family == "G2a":
            indices = torch.randint(0, embedding_weight.shape[0], (length,), generator=rng)
            sequence = embedding_weight[indices] + positional_weight[:length]
        elif family == "G2b":
            noise = torch.randn(length, width, generator=rng, dtype=torch.float64)
            sequence = (mu + noise @ factor.T).float() + positional_weight[:length]
        else:
            raise ValueError(f"Unknown G2 family: {family}")
        states = prefix_fn(model, sequence.unsqueeze(0).to(device))
        chunks.append(states.squeeze(0).cpu())
    return torch.cat(chunks)


def batch_permutation(seed, stream, pool_size):
    """Deterministic batch-order permutation for an arbitrary (seed, stream)."""
    return torch.randperm(pool_size, generator=generator(seed, stream))


def permutation_for(seed, family, pool_size):
    """Deterministic, init-independent batch order for a v3a family."""
    return batch_permutation(seed, f"v3a_{family}_perm", pool_size)


def batch_indices(permutation, batch_size, step):
    """Contiguous slice of the permutation for a 1-based training step."""
    batches = len(permutation) // batch_size
    index = (step - 1) % batches
    return permutation[index * batch_size:(index + 1) * batch_size]


def copy_initialized_student(teacher, width, intermediate):
    """Student whose parameters are the frozen norm-product copy (all trainable)."""
    from .modules import MLP, weight_copy
    copied = weight_copy(teacher, intermediate)
    student = MLP(width, intermediate)
    student.load_state_dict(copied.state_dict())
    return student
