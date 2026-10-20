import hashlib
import torch


def stream_seed(seed, name):
    return int.from_bytes(hashlib.sha256(f"{seed}:{name}".encode()).digest()[:8], "little") % (2**63 - 1)


def generator(seed, name):
    return torch.Generator(device="cpu").manual_seed(stream_seed(seed, name))


def gaussian(count, width, rng):
    return torch.randn(count, width, generator=rng, dtype=torch.float32)


def ln2_transform(weight, bias, eps):
    """Frozen ln_2,T probe transform: h = ln_2,T(z), z ~ N(0, I).

    Pure tensor function of the synthetic z and the frozen Teacher ln_2
    parameters only — no tokenizer, no text, no language-derived inputs.
    """
    def transform(z):
        return torch.nn.functional.layer_norm(z, (z.shape[-1],), weight, bias, eps)
    return transform


def development_probes(family, transform, seed, size, width):
    """Per-family development set from the shared (seed, "development") z-stream.

    G0 returns z itself; G1 returns transform(z) with the frozen ln_2,T — each
    family's checkpoint selection happens on its own training distribution.
    Regression guard for the dev-distribution mismatch fault: a G1 dev set is
    never the raw z stream.
    """
    z = gaussian(size, width, generator(seed, "development"))
    if family == "G1":
        if transform is None:
            raise ValueError("G1 development probes require the frozen ln_2 transform")
        return transform(z)
    if family != "G0" or transform is not None:
        raise ValueError(f"Inconsistent development family request: {family}")
    return z
