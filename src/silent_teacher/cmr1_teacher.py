"""CMR1 I2 — minimal wrapper around the frozen Pythia-160M zero-based block-5 operator.

Keeps five things distinct (protocol §2–§4, B1/B3 contracts):

1. synthetic / random accepted token ids (the Native sampling law, B2);
2. prefix blocks 0–4 (used **only** to construct Native support);
3. the block-5 boundary hidden state ``H``;
4. the block-5 residual update ``F(H) = U_T(H) = A_T(H) + B_T(H)`` (branch accumulation, not
   identity-plus-update);
5. the block-5 full output ``H + U_T(H)`` (verification only).

**Auxiliary semantics are part of the operator's definition, not incidental state.** The 4.56.2 reference
computes the position ids, the causal mask and the RoPE position embeddings *at the model level* and
passes them into every layer, so the wrapper reproduces exactly that prelude and passes
``use_cache=False`` with no ``layer_past``. The boundary-locality property is therefore **conditional**:

    Given the same frozen boundary tensor H and the same frozen auxiliary positional/mask/numerical
    semantics, block 5 must not depend on prefix-block internal execution history or hidden cache/state.

**Runtime.** The scientific reference runtime is ``transformers==4.56.2``. This module asserts the loaded
version and refuses to act as the scientific reference otherwise; the currently installed 5.15.1 may
never silently become the reference (plan §5.5).

Scope: no training, no target reader, no RNG generation, no materialization, no scientific outcome.
"""
from __future__ import annotations

import json
import weakref
from pathlib import Path

import torch
from torch import nn

#: Teacher scientific identity (Stage-A frozen, B1 contract). No mutable branch name is used.
PINNED_REPOSITORY = "EleutherAI/pythia-160m"
PINNED_REVISION = "50f5173d932e8e61f858120bcb800b97af589f46"
REFERENCE_TRANSFORMERS = "4.56.2"
REFERENCE_HUB = "0.36.2"
BLOCK_INDEX = 5
SEQUENCE_LENGTH = 128
VOCAB_SIZE = 50304

#: Frozen config facts recorded from the pinned revision (B1 contract, externally verified).
FROZEN_CONFIG_FACTS = {
    "hidden_size": 768,
    "num_hidden_layers": 12,
    "num_attention_heads": 12,
    "intermediate_size": 3072,
    "rotary_pct": 0.25,
    "rotary_emb_base": 10000,
    "layer_norm_eps": 1e-5,
    "initializer_range": 0.02,
    "hidden_act": "gelu",
    "use_parallel_residual": True,
    "vocab_size": VOCAB_SIZE,
    "bos_token_id": 0,
    "eos_token_id": 0,
}


#: Frozen weight-artifact identity (B1 record; externally verified from the official repository).
PINNED_WEIGHT_SHA256 = "29d2e457a664e41c12c735f20a36dc0956a665f614a54ce5db21a32e75965270"
PINNED_WEIGHT_BYTES = 374_998_696
PINNED_CONFIG_SHA256 = None      # config.json identity is determined by revision + path (B1 record)

_ASSET_FILES = ("config.json", "model.safetensors")
_PINNED_LOADED_MODELS: weakref.WeakSet = weakref.WeakSet()


class TeacherAssetsUnavailable(RuntimeError):
    """Raised when the pinned Teacher artifacts are not present in the local asset cache.

    **Timing (corrected in I1–I3.1).** The real pinned assets must be *available, identity-verified and
    loaded through the production loader* **before** the qualification/calibration identity freeze and
    therefore **before** calibration and qualification — they are a pre-qualification implementation
    prerequisite, **not** a Stage-B retrieval obligation. The corrected ordering is:

        Stage A frozen → implementation/preflight (this round) → qualification identity freeze →
        calibration/tolerance freeze → Q1/Q2/Q3 → Stage B execution-control freeze.
    """


class TeacherWeightIdentityMismatch(RuntimeError):
    """Raised when a local weight artifact does not match the frozen content identity."""


class TeacherConfigIdentityMismatch(RuntimeError):
    """Raised when a local config artifact disagrees with the frozen configuration facts."""


class ReferenceRuntimeMismatch(RuntimeError):
    """Raised when the loaded transformers version is not the frozen scientific reference."""


def assert_reference_runtime() -> str:
    """Assert that the loaded ``transformers`` is the frozen scientific reference; return its version."""
    import transformers

    version = transformers.__version__
    if version != REFERENCE_TRANSFORMERS:
        raise ReferenceRuntimeMismatch(
            f"loaded transformers=={version}; the scientific reference is =={REFERENCE_TRANSFORMERS}. "
            "Any governed reference comparison must run under the frozen runtime (plan §5.5).")
    return version


# --------------------------------------------------------------------------------- builders
def build_reference_teacher(seed: int = 0, device: str = "cpu", dtype: torch.dtype = torch.float32):
    """Architecture-identical reference model with **NON-GOVERNING synthetic weights**.

    Uses a ``GPTNeoXConfig`` carrying the frozen config facts and the framework's own initialization.
    This is sufficient for every structural/numerical wrapper property (boundary identity, auxiliary
    semantics, conditional independence, no-cache, determinism) because those are architecture
    properties, not weight properties. It is **not** the scientific Teacher.

    **Ordering (corrected in the I1–I3 terminal closure).** Weight-level identity is not a later
    Stage-B retrieval step. The frozen Teacher asset identity was already verified during I1–I3
    preflight, and Teacher availability/load identity is required **before qualification** — it is a
    pre-qualification implementation prerequisite. Stage B is the later *execution-control* freeze; it
    does not retrieve or re-establish Teacher weight identity. See :class:`TeacherAssetsUnavailable`.
    """
    assert_reference_runtime()
    from transformers import GPTNeoXConfig, GPTNeoXForCausalLM

    config = GPTNeoXConfig(**FROZEN_CONFIG_FACTS)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        model = GPTNeoXForCausalLM(config)
    model = model.to(device=device, dtype=dtype).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.weights_provenance = "NON-GOVERNING SYNTHETIC WEIGHTS — architecture tests only"
    return model, model.gpt_neox.layers[BLOCK_INDEX]


def sha256_of_file(path: str | Path, chunk: int = 1 << 22) -> str:
    """Streaming SHA-256 (never loads the whole artifact into memory)."""
    import hashlib

    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def build_pinned_teacher(local_root: str | Path, device: str = "cpu",
                         dtype: torch.dtype = torch.float32):
    """Load the **pinned** Teacher through the official 4.56.2 loading semantics, fail closed.

    Steps, each of which must hold:

    1. the asset cache contains the required files;
    2. ``model.safetensors`` hashes to the **frozen** weight identity (content-addressed — the cache path
       is not scientific identity);
    3. ``config.json`` agrees with every frozen configuration fact;
    4. ``GPTNeoXForCausalLM.from_pretrained(..., local_files_only=True)`` loads the checkpoint under the
       official key namespace (``gpt_neox.*`` / ``embed_out.weight``) — **no hand-written renaming**;
    5. ``missing_keys`` is empty and every ``unexpected_keys`` entry matches a pattern the model itself
       declares ignorable (``attention.bias`` / ``attention.masked_bias`` — the legacy causal-mask
       buffers present in the released checkpoint); ``mismatched_keys`` is empty;
    6. the model is eval-only, parameters frozen, and loaded in the requested dtype (frozen regime: FP32).

    Raises :class:`TeacherAssetsUnavailable`, :class:`TeacherWeightIdentityMismatch`,
    :class:`TeacherConfigIdentityMismatch` or :class:`ReferenceRuntimeMismatch`.  It never falls back to
    synthetic weights.
    """
    assert_reference_runtime()
    from transformers import GPTNeoXForCausalLM

    root = Path(local_root)
    absent = [name for name in _ASSET_FILES if not (root / name).exists()]
    if absent:
        raise TeacherAssetsUnavailable(
            f"pinned Teacher assets missing from {root.as_posix()}: {absent}. Populate the cache with the "
            f"pinned revision {PINNED_REVISION} artifacts (config.json + model.safetensors, the latter "
            f"hash-verified against {PINNED_WEIGHT_SHA256}). These assets are a PRE-QUALIFICATION "
            "implementation prerequisite, not a Stage-B retrieval obligation.")

    weights_path = root / "model.safetensors"
    digest = sha256_of_file(weights_path)
    if digest != PINNED_WEIGHT_SHA256:
        raise TeacherWeightIdentityMismatch(
            f"weight artifact {weights_path.as_posix()} hashes to {digest}; the frozen identity is "
            f"{PINNED_WEIGHT_SHA256}. Refusing to use it.")

    raw = json.loads((root / "config.json").read_text(encoding="utf-8"))
    mismatches = {key: (raw.get(key), value) for key, value in FROZEN_CONFIG_FACTS.items()
                  if raw.get(key) != value}
    if mismatches:
        raise TeacherConfigIdentityMismatch(
            f"pinned config disagrees with the frozen facts: {mismatches}")

    model, info = GPTNeoXForCausalLM.from_pretrained(
        root.as_posix(), local_files_only=True, dtype=dtype, output_loading_info=True)
    missing = list(info.get("missing_keys", []))
    unexpected = list(info.get("unexpected_keys", []))
    mismatched = list(info.get("mismatched_keys", []))
    ignorable = tuple(GPTNeoXForCausalLM._keys_to_ignore_on_load_unexpected or ())
    import re as _re

    def _officially_ignorable(key: str) -> bool:
        return any(_re.search(pattern, key) for pattern in ignorable)

    unexplained = [key for key in unexpected if not _officially_ignorable(key)]
    if missing or mismatched or unexplained:
        raise TeacherWeightIdentityMismatch(
            f"official load reported missing={missing[:4]} mismatched={mismatched[:4]} "
            f"unexplained_unexpected={unexplained[:4]}; officially ignored keys were "
            f"{[k for k in unexpected if _officially_ignorable(k)][:3]}")

    model = model.to(device=device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.weights_provenance = {
        "repository": PINNED_REPOSITORY,
        "revision": PINNED_REVISION,
        "weight_sha256": digest,
        "weight_bytes": weights_path.stat().st_size,
        "loaded_dtype": str(next(model.parameters()).dtype),
        "checkpoint_dtype": raw.get("torch_dtype"),
        "cache_path": root.as_posix(),
        "note": "cache path is transport provenance, not scientific identity",
        "officially_ignored_keys": [k for k in unexpected if _officially_ignorable(k)],
    }
    _PINNED_LOADED_MODELS.add(model)
    return model, model.gpt_neox.layers[BLOCK_INDEX]


def assert_pinned_teacher_wrapper(wrapper) -> dict:
    """Governed path accepts only a model actually loaded by the pinned-asset loader."""
    from transformers import GPTNeoXForCausalLM
    if type(wrapper) is not BlockFiveWrapper or type(wrapper.model) is not GPTNeoXForCausalLM:
        raise TeacherWeightIdentityMismatch("governed Teacher must be the production pinned wrapper/model")
    model = wrapper.model
    provenance = getattr(model, "weights_provenance", None)
    if model not in _PINNED_LOADED_MODELS or not isinstance(provenance, dict):
        raise TeacherWeightIdentityMismatch("Teacher was not produced by the pinned asset loader")
    expected = {"repository": PINNED_REPOSITORY, "revision": PINNED_REVISION,
                "weight_sha256": PINNED_WEIGHT_SHA256, "weight_bytes": PINNED_WEIGHT_BYTES,
                "loaded_dtype": "torch.float32"}
    if any(provenance.get(key) != value for key, value in expected.items()):
        raise TeacherWeightIdentityMismatch("loaded Teacher provenance differs from the frozen identity")
    if any(getattr(model.config, key) != value for key, value in FROZEN_CONFIG_FACTS.items()):
        raise TeacherConfigIdentityMismatch("loaded Teacher configuration differs from the frozen facts")
    if any(name in wrapper.__dict__ for name in (
            "prefix_states", "capture_boundary", "residual_update", "layer_output", "auxiliary")):
        raise TeacherWeightIdentityMismatch("governed Teacher wrapper methods were replaced on the instance")
    if (wrapper.block is not model.gpt_neox.layers[BLOCK_INDEX]
            or wrapper.block.use_parallel_residual is not True or model.training
            or any(p.requires_grad or p.dtype != torch.float32 for p in model.parameters())):
        raise TeacherWeightIdentityMismatch("the governed Teacher is not the frozen FP32 eval block-5 path")
    assert_reference_runtime()
    return {key: provenance[key] for key in expected}


# --------------------------------------------------------------------------------- wrapper
class BlockFiveWrapper:
    """The frozen block-5 operator, with the model-level auxiliary semantics reproduced explicitly.

    **The block index is not a parameter** (I1–I3.1): production always wraps ``gpt_neox.layers[5]``.
    Passing a block index is a ``TypeError`` at the call site, and the wrapped layer is asserted to *be*
    that exact layer object — so the production path cannot silently become block 4, 6 or any other.
    """

    def __init__(self, model) -> None:
        self.model = model
        self.block_index = BLOCK_INDEX
        self.block = model.gpt_neox.layers[BLOCK_INDEX]
        # identity assertion (C7 convention): the wrapper's layer is THE block-5 layer object
        assert self.block is model.gpt_neox.layers[BLOCK_INDEX], "block identity assertion failed"
        assert len(model.gpt_neox.layers) > BLOCK_INDEX, "model does not have a block 5"
        assert self.block.use_parallel_residual is True, "frozen Teacher uses parallel residuals"
        self.config = model.config
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    # ------------------------------------------------------------------ auxiliary semantics
    def auxiliary(self, hidden: torch.Tensor, position_ids: torch.Tensor | None = None):
        """Reproduce the model-level prelude EXACTLY as the 4.56.2 reference computes it.

        Reference prelude (``GPTNeoXModel.forward``): with no cache,
        ``cache_position = arange(0, T)``; ``position_ids = cache_position.unsqueeze(0)`` — shape
        ``[1, T]``, deliberately **not** batch-expanded; the causal mask is created with
        ``input_embeds`` (used only to infer batch/length/dtype) and that ``cache_position``; the RoPE
        position embeddings are computed once and passed to every layer.
        """
        from transformers.masking_utils import create_causal_mask

        if position_ids is None:
            cache_position = torch.arange(hidden.shape[1], device=hidden.device)
            position_ids = cache_position.unsqueeze(0)          # [1, T], as in the reference
        else:
            cache_position = position_ids[0]
        causal_mask = create_causal_mask(config=self.config, input_embeds=hidden, attention_mask=None,
                                         cache_position=cache_position, past_key_values=None,
                                         position_ids=position_ids)
        position_embeddings = self.model.gpt_neox.rotary_emb(hidden, position_ids=position_ids)
        return position_ids, causal_mask, position_embeddings

    # ------------------------------------------------------------------ operator quantities
    def layer_output(self, hidden: torch.Tensor, position_ids: torch.Tensor | None = None,
                     **aux) -> torch.Tensor:
        """The authoritative layer call: ``H + U_T(H)`` (verification quantity)."""
        position_ids, causal_mask, position_embeddings = self.auxiliary(hidden, position_ids)
        with torch.no_grad():
            out = self.block(hidden, attention_mask=causal_mask, position_ids=position_ids,
                             layer_past=None, use_cache=False,
                             position_embeddings=position_embeddings)
        # the 4.56.2 layer returns a tuple; the model itself takes `outputs[0]` (hidden states)
        return out[0] if isinstance(out, tuple) else out

    def residual_update(self, hidden: torch.Tensor, position_ids: torch.Tensor | None = None,
                        **aux) -> torch.Tensor:
        """``U_T(H) = A_T(H) + B_T(H)``, computed in the layer's own branch-accumulation order.

        ``GPTNeoXLayer.forward`` evaluates ``mlp_output + attn_output + hidden_states``, i.e.
        ``(mlp + attn) + H``; computing ``mlp + attn`` directly here is therefore **bitwise** the same
        intermediate the layer produces, and ``layer_output(H) == residual_update(H) + H`` is asserted.
        """
        position_ids, causal_mask, position_embeddings = self.auxiliary(hidden, position_ids)
        with torch.no_grad():
            attention = self.block.attention(
                self.block.input_layernorm(hidden), attention_mask=causal_mask,
                position_ids=position_ids, layer_past=None, use_cache=False,
                position_embeddings=position_embeddings)[0]
            attention = self.block.post_attention_dropout(attention)
            mlp = self.block.post_mlp_dropout(
                self.block.mlp(self.block.post_attention_layernorm(hidden)))
        return mlp + attention

    # ------------------------------------------------------------------ boundary quantities
    def prefix_states(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Run embeddings + blocks 0–4 **only** (the Native support construction path)."""
        assert token_ids.dim() == 2, "token_ids must be [B, L]"
        assert token_ids.shape[1] == SEQUENCE_LENGTH, f"frozen L={SEQUENCE_LENGTH}"
        with torch.no_grad():
            hidden = self.model.gpt_neox.embed_in(token_ids)
            position_ids, causal_mask, position_embeddings = self.auxiliary(hidden)
            for layer in self.model.gpt_neox.layers[: self.block_index]:
                out = layer(hidden, attention_mask=causal_mask, position_ids=position_ids,
                            layer_past=None, use_cache=False,
                            position_embeddings=position_embeddings)
                hidden = out[0] if isinstance(out, tuple) else out   # 4.56.2 returns a tuple
        return hidden

    def capture_boundary(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Capture exact model-forward block-5 input, stopping before the target layer/logits."""
        captured: dict[str, torch.Tensor] = {}

        class _BoundaryCaptured(Exception):
            pass

        def hook(_module, inputs):
            captured["hidden"] = inputs[0].detach().clone()
            raise _BoundaryCaptured()

        handle = self.block.register_forward_pre_hook(hook)
        try:
            with torch.no_grad():
                self.model(input_ids=token_ids, use_cache=False)
        except _BoundaryCaptured:
            pass
        finally:
            handle.remove()
        if "hidden" not in captured:
            raise RuntimeError("block-5 boundary capture failed")
        return captured["hidden"]

    # ------------------------------------------------------------------ structural checks
    def assert_boundary_semantics(self, hidden: torch.Tensor, position_ids: torch.Tensor | None = None):
        """Assert the frozen auxiliary semantics and the branch-accumulation identity."""
        position_ids, causal_mask, position_embeddings = self.auxiliary(hidden, position_ids)
        assert self.block.use_parallel_residual is True
        assert self.block.training is False and self.model.training is False
        assert self.block.post_attention_dropout.p == 0.0 and self.block.post_mlp_dropout.p == 0.0
        assert self.block.attention.attention_dropout == 0.0
        output = self.layer_output(hidden, position_ids)
        update = self.residual_update(hidden, position_ids)
        return {
            "layer_output_is_update_plus_hidden": bool(torch.equal(output, update + hidden)),
            "position_ids": tuple(position_ids.shape),
            "causal_mask_is_present": causal_mask is not None,
            "position_embeddings": tuple(t.dim() for t in position_embeddings),
            "dropout_values": (self.block.post_attention_dropout.p, self.block.post_mlp_dropout.p),
        }


def count_teacher_block_parameters(block: nn.Module) -> int:
    return sum(parameter.numel() for parameter in block.parameters())


__all__ = [
    "BLOCK_INDEX", "FROZEN_CONFIG_FACTS", "PINNED_REPOSITORY", "PINNED_REVISION",
    "REFERENCE_HUB", "REFERENCE_TRANSFORMERS", "SEQUENCE_LENGTH", "BlockFiveWrapper",
    "ReferenceRuntimeMismatch", "TeacherAssetsUnavailable", "assert_reference_runtime",
    "build_pinned_teacher", "build_reference_teacher", "count_teacher_block_parameters",
]
