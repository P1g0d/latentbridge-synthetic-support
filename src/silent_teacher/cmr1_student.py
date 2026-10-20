"""CMR1 I1 — the frozen Pythia-family compressed Student block.

Implements the Stage-A frozen geometry (semantic freeze commit
``c6791ee2f70e01a0ac61de22d6ba93075413c42f``; protocol §3.2/§3.5, B3 and B15 contracts):

* residual width 768;
* attention inner width 384 — **6 heads x 64**;
* Q/K/V 768 -> 384 each (SPLIT projections: the canonical production representation, see
  ``CANONICAL_QKV_REPRESENTATION``);
* output projection 384 -> 768;
* FFN 768 -> 1536 -> 768;
* two independent pre-branch LayerNorms, ``eps = 1e-5``, weight 1 / bias 0, trainable, **both reading the
  same residual input**;
* Pythia **parallel** residual: ``U_S(H) = A_S(LN1(H)) + MLP_S(LN2(H))``;
* attention dropout 0, hidden dropout 0 (no stochastic module exists at all);
* attention bias present (Q/K/V and output);
* RoPE: fraction 0.25 -> rotary dimension **16** per head, **first 16** coordinates rotated, NeoX
  **half-split** pairing, base 10000, **8** inverse frequencies;
* exact erf GELU;
* frozen initialization: projection weights ``N(0, 0.02**2)``, all biases zero, LayerNorm weight 1 /
  bias 0, **no** output-projection or depth scaling;
* the primary scientific forward quantity is the **residual update** ``U_S(H)``; the full block output is
  provided separately as ``H + U_S(H)``.

Scope: this module contains no training loop, no target reader, no RNG, no data access and no scientific
outcome. It is I1 implementation code; verification lives in ``tests/cmr1/``.
"""
from __future__ import annotations

import torch
from torch import nn

# ------------------------------------------------------------------ frozen constants (no freedom)
WIDTH = 768
HEADS = 6
HEAD_DIM = 64
INNER_WIDTH = HEADS * HEAD_DIM            # 384
FFN_WIDTH = 1536
LAYERNORM_EPS = 1e-5
ROTARY_FRACTION = 0.25
ROTARY_DIM = int(HEAD_DIM * ROTARY_FRACTION)   # 16
ROPE_BASE = 10000.0
INIT_STD = 0.02
ATTENTION_DROPOUT = 0.0
HIDDEN_DROPOUT = 0.0
ATTENTION_BIAS = True

EXPECTED_PARAMETER_COUNT = 3_546_240

#: Canonical production representation (I1 decision, recorded). SPLIT is canonical because our inner
#: width (384) differs from the hidden width (768), so the reference model's fused layout convention
#: cannot be copied verbatim; three explicit projections remove the fused-interleaving question and make
#: rotated-Q/K extraction a direct module access. The fused spelling was **removed** in I1–I3.1: it
#: existed only as a cross-check, and keeping it reachable risked a silent production switch. Its
#: parameter-count identity was verified in Stage A (B3) and remains on record there.
CANONICAL_QKV_REPRESENTATION = "split"

#: Module-creation / initialization ordering that a formal ``student_init`` seed will govern.
#: ``CMR1StudentBlock.__init__`` builds: ln1 -> ln2 -> attention(q_proj, k_proj, v_proj, o_proj, rotary)
#: -> mlp(fc, act, proj).  Each ``nn.Linear`` performs PyTorch's default constructor initialization,
#: which **consumes CPU RNG**; ``reset_parameters()`` then overwrites every projection weight with the
#: frozen ``N(0, 0.02**2)`` and every bias with zero, and sets the LayerNorm parameters to weight 1 /
#: bias 0.  The constructor draws therefore affect the RNG *position* but not the initialized values:
#: the frozen rule fully determines the weights.  This module applies the frozen rule ONCE, in
#: ``__init__``; it never reapplies it and never draws a seed itself.
INITIALIZATION_ORDER = (
    "ln1", "ln2", "attention.q_proj", "attention.k_proj", "attention.v_proj", "attention.o_proj",
    "attention.rotary(buffer, no init call)", "mlp.fc", "mlp.proj",
)


class StudentParameterIdentityMismatch(RuntimeError):
    """Raised when the instantiated Student block does not have the frozen parameter count."""


class StudentGeometryIdentityMismatch(RuntimeError):
    """Raised when a production Student does not realize the frozen geometry (I1–I3.1)."""


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """NeoX half-split rotation: pairs ``(i, i + n/2)`` inside the rotated slice."""
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


class StudentRotaryEmbedding(nn.Module):
    """RoPE for the compressed Student: rotary dimension 16, base 10000, 8 inverse frequencies.

    The inverse-frequency construction and the cosine/sine layout follow the frozen 4.56.2 reference
    arithmetic exactly (``1 / base ** (arange(0, dim, 2) / dim)``; the frequency axis is **duplicated**
    to length ``rotary_dim``, never interleaved).
    """

    def __init__(self, head_dim: int = HEAD_DIM, fraction: float = ROTARY_FRACTION,
                 base: float = ROPE_BASE) -> None:
        super().__init__()
        self.head_dim = head_dim
        self.rotary_dim = int(head_dim * fraction)
        self.base = base
        inv_freq = 1.0 / (base ** (torch.arange(0, self.rotary_dim, 2).float() / self.rotary_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)   # 8 frequencies

    def cos_sin(self, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(cos, sin)`` of shape ``[B, 1, T, rotary_dim]`` for the given positions."""
        freqs = self.inv_freq[None, :, None].float() @ position_ids[:, None, :].float()
        emb = torch.cat((freqs, freqs), dim=1)                       # duplicate the FREQUENCY axis
        return emb.cos().transpose(1, 2).unsqueeze(1), emb.sin().transpose(1, 2).unsqueeze(1)

    def apply(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """Rotate the **first ``rotary_dim``** coordinates of ``x``; leave the rest untouched."""
        rot, passthrough = x[..., : self.rotary_dim], x[..., self.rotary_dim:]
        rotated = (rot * cos) + (rotate_half(rot) * sin)
        return torch.cat((rotated, passthrough), dim=-1)


class NarrowAttention(nn.Module):
    """Compressed attention: Q/K/V 768 -> 384 each, 6 heads x 64, O 384 -> 768, causal, RoPE-rotated Q/K."""

    def __init__(self, width: int = WIDTH, heads: int = HEADS, head_dim: int = HEAD_DIM,
                 bias: bool = ATTENTION_BIAS) -> None:
        super().__init__()
        self.width, self.heads, self.head_dim = width, heads, head_dim
        self.inner_width = heads * head_dim
        self.attention_dropout = ATTENTION_DROPOUT      # frozen value; no dropout module exists
        self.scaling = head_dim ** -0.5                 # 0.125
        self.q_proj = nn.Linear(width, self.inner_width, bias=bias)
        self.k_proj = nn.Linear(width, self.inner_width, bias=bias)
        self.v_proj = nn.Linear(width, self.inner_width, bias=bias)
        self.o_proj = nn.Linear(self.inner_width, width, bias=bias)
        self.rotary = StudentRotaryEmbedding(head_dim)

    # ------------------------------------------------------------------ projections
    def project_qkv(self, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return per-head ``q, k, v`` of shape ``[B, heads, T, head_dim]`` from the SPLIT projections."""
        batch, length, _ = hidden.shape
        shape = (batch, length, self.heads, self.head_dim)
        return (self.q_proj(hidden).view(*shape).transpose(1, 2),
                self.k_proj(hidden).view(*shape).transpose(1, 2),
                self.v_proj(hidden).view(*shape).transpose(1, 2))

    def rotated_qk(self, hidden: torch.Tensor, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """NON-SCIENTIFIC-PATH DEBUG/TEST HOOK: the RoPE-rotated query and key tensors.

        Exposed so that the future Q2-4 calibration can extract ``TS-Q2-4`` (rotated QUERY + rotated KEY
        only) without reaching into private state. It is not part of the scientific forward output.
        """
        query, key, _ = self.project_qkv(hidden)
        cos, sin = self.rotary.cos_sin(position_ids)
        return self.rotary.apply(query, cos, sin), self.rotary.apply(key, cos, sin)

    def forward(self, hidden: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        batch, length, _ = hidden.shape
        query, key, value = self.project_qkv(hidden)
        cos, sin = self.rotary.cos_sin(position_ids)
        query = self.rotary.apply(query, cos, sin)
        key = self.rotary.apply(key, cos, sin)

        scores = (query @ key.transpose(-1, -2)) * self.scaling
        causal = torch.triu(torch.ones(length, length, dtype=torch.bool, device=hidden.device),
                            diagonal=1)
        scores = scores.masked_fill(causal, float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        context = (weights @ value).transpose(1, 2).reshape(batch, length, self.inner_width)
        return self.o_proj(context)


class StudentMLP(nn.Module):
    """FFN 768 -> 1536 -> 768 with exact erf GELU; pointwise over positions."""

    def __init__(self, width: int = WIDTH, hidden_width: int = FFN_WIDTH) -> None:
        super().__init__()
        self.hidden_dropout = HIDDEN_DROPOUT          # frozen value; no dropout module exists
        self.fc = nn.Linear(width, hidden_width)
        self.act = nn.GELU()                          # exact erf GELU (approximate="none")
        self.proj = nn.Linear(hidden_width, width)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.proj(self.act(self.fc(hidden)))


class CMR1StudentBlock(nn.Module):
    """The frozen compressed Student block.

    ``forward`` (and ``update``) return the **residual update** ``U_S(H)``; ``block_output`` returns
    ``H + U_S(H)``. The MLP branch reads the residual input ``H`` through ``ln2`` — never ``H + A``.
    """

    def __init__(self, width: int = WIDTH, heads: int = HEADS, head_dim: int = HEAD_DIM,
                 ffn_width: int = FFN_WIDTH, eps: float = LAYERNORM_EPS) -> None:
        super().__init__()
        self.width, self.heads, self.head_dim, self.ffn_width = width, heads, head_dim, ffn_width
        self.layernorm_eps = eps
        self.ln1 = nn.LayerNorm(width, eps=eps)          # pre-attention, reads H
        self.ln2 = nn.LayerNorm(width, eps=eps)          # pre-MLP, reads the SAME residual input H
        self.attention = NarrowAttention(width, heads, head_dim)
        self.mlp = StudentMLP(width, ffn_width)
        self.reset_parameters()

    # ------------------------------------------------------------------ frozen initialization
    def reset_parameters(self) -> None:
        """N(0, 0.02**2) for every projection weight; zero biases; LayerNorm weight 1 / bias 0.

        No output-projection scaling and no depth scaling is applied (the GPT-2 ``c_proj`` rule is
        explicitly not imported — B15).
        """
        projections = [m for m in self.modules() if isinstance(m, nn.Linear)]
        for layer in projections:
            nn.init.normal_(layer.weight, mean=0.0, std=INIT_STD)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
        for norm in (self.ln1, self.ln2):
            nn.init.ones_(norm.weight)
            nn.init.zeros_(norm.bias)

    # ------------------------------------------------------------------ forward
    def update(self, hidden: torch.Tensor, position_ids: torch.Tensor | None = None) -> torch.Tensor:
        """``U_S(H) = A_S(LN1(H)) + MLP_S(LN2(H))`` — the primary scientific quantity."""
        if position_ids is None:
            position_ids = torch.arange(hidden.shape[1], device=hidden.device).unsqueeze(0)
            position_ids = position_ids.expand(hidden.shape[0], -1)
        attention = self.attention(self.ln1(hidden), position_ids)
        mlp = self.mlp(self.ln2(hidden))          # NOTE: reads `hidden`, NOT `hidden + attention`
        return attention + mlp

    def forward(self, hidden: torch.Tensor, position_ids: torch.Tensor | None = None) -> torch.Tensor:
        return self.update(hidden, position_ids)

    def block_output(self, hidden: torch.Tensor, position_ids: torch.Tensor | None = None) -> torch.Tensor:
        """``S_block(H) = H + U_S(H)`` — provided separately; never the training target."""
        return hidden + self.update(hidden, position_ids)

    # ------------------------------------------------------------------ introspection
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def assert_frozen_parameter_identity(self) -> int:
        count = self.parameter_count()
        if count != EXPECTED_PARAMETER_COUNT:
            raise StudentParameterIdentityMismatch(
                f"Student block has {count} parameters; frozen identity requires "
                f"{EXPECTED_PARAMETER_COUNT}. HARD STOP — STUDENT_PARAMETER_IDENTITY_MISMATCH.")
        return count

    def parameter_breakdown(self) -> dict[str, int]:
        """Per-module breakdown of the frozen parameter budget (for audit)."""
        return {
            "ln1": sum(p.numel() for p in self.ln1.parameters()),
            "ln2": sum(p.numel() for p in self.ln2.parameters()),
            "attention": sum(p.numel() for p in self.attention.parameters()),
            "mlp": sum(p.numel() for p in self.mlp.parameters()),
            "total": self.parameter_count(),
        }

    def assert_frozen_geometry(self) -> dict:
        """Verify every frozen geometry constant on the instantiated module (I1–I3.1).

        Any deviation raises :class:`StudentGeometryIdentityMismatch` — the production path can neither
        be given nor drift into a non-frozen geometry.
        """
        checks: dict[str, tuple] = {
            "residual_width": (self.width, WIDTH),
            "heads": (self.heads, HEADS),
            "head_dim": (self.head_dim, HEAD_DIM),
            "inner_width": (self.attention.inner_width, INNER_WIDTH),
            "ffn_width": (self.mlp.fc.out_features, FFN_WIDTH),
            "layernorm_eps": (self.ln1.eps, LAYERNORM_EPS),
            "layernorm_eps_2": (self.ln2.eps, LAYERNORM_EPS),
            "rotary_dim": (self.attention.rotary.rotary_dim, ROTARY_DIM),
            "rope_base": (self.attention.rotary.base, ROPE_BASE),
            "inverse_frequencies": (self.attention.rotary.inv_freq.numel(), ROTARY_DIM // 2),
            "attention_scaling": (self.attention.scaling, HEAD_DIM ** -0.5),
            "attention_dropout": (self.attention.attention_dropout, ATTENTION_DROPOUT),
            "hidden_dropout": (self.mlp.hidden_dropout, HIDDEN_DROPOUT),
            "split_qkv": (hasattr(self.attention, "q_proj") and hasattr(self.attention, "k_proj")
                          and hasattr(self.attention, "v_proj")
                          and not hasattr(self.attention, "qkv_proj"), True),
            "no_dropout_module": (not any(isinstance(m, nn.Dropout) for m in self.modules()), True),
            "biases_present": (all(m.bias is not None for m in (self.attention.q_proj,
                                                                self.attention.k_proj,
                                                                self.attention.v_proj,
                                                                self.attention.o_proj,
                                                                self.mlp.fc, self.mlp.proj)), True),
        }
        failures = {name: found for name, (found, expected) in checks.items() if found != expected}
        if failures:
            raise StudentGeometryIdentityMismatch(
                f"production Student geometry deviates from the frozen identity: {failures}")
        return {name: found for name, (found, _expected) in checks.items()}


def build_student(seed: int | None = None) -> CMR1StudentBlock:
    """**Production** construction of the CMR1 Student block.

    Takes **no** geometry, epsilon, rotary or QKV-representation argument: the frozen identity is
    hard-wired (I1–I3.1). Any attempt to pass such an argument is a ``TypeError`` at the call site, and a
    block that somehow deviates from the frozen constants is rejected by
    :meth:`CMR1StudentBlock.assert_frozen_geometry` and
    :meth:`CMR1StudentBlock.assert_frozen_parameter_identity`.

    ``seed`` is an **engineering-only** convenience for tests (NON-GOVERNING); the scientific paired
    initialization identity is derived from the B9 RNG role schema and is not generated here.
    """
    if seed is None:
        block = CMR1StudentBlock()
    else:
        with torch.random.fork_rng(devices=[]):      # no side effect on the surrounding CPU RNG
            torch.manual_seed(int(seed))
            block = CMR1StudentBlock()
    block.assert_frozen_geometry()
    block.assert_frozen_parameter_identity()
    return block


def build_student_architecture() -> CMR1StudentBlock:
    """Allocate the fixed architecture without invoking any parameter initializer or RNG.

    Only persisted-state consumers use this path. Empty parameters are not an initialization;
    they must all be replaced by a strict, verified frozen state before any forward call.
    The ordinary construction/initialization path above remains unchanged.
    """
    def module(cls, **attributes):
        value = cls.__new__(cls)
        nn.Module.__init__(value)
        for name, attribute in attributes.items():
            setattr(value, name, attribute)
        return value

    def linear(inputs, outputs):
        value = module(nn.Linear, in_features=inputs, out_features=outputs)
        value.weight = nn.Parameter(torch.empty(outputs, inputs, dtype=torch.float32))
        value.bias = nn.Parameter(torch.empty(outputs, dtype=torch.float32))
        return value

    def norm():
        value = module(nn.LayerNorm, normalized_shape=(WIDTH,), eps=LAYERNORM_EPS,
                       elementwise_affine=True)
        value.weight = nn.Parameter(torch.empty(WIDTH, dtype=torch.float32))
        value.bias = nn.Parameter(torch.empty(WIDTH, dtype=torch.float32))
        return value

    block = module(CMR1StudentBlock, width=WIDTH, heads=HEADS, head_dim=HEAD_DIM,
                   ffn_width=FFN_WIDTH, layernorm_eps=LAYERNORM_EPS)
    block.ln1, block.ln2 = norm(), norm()
    block.attention = module(NarrowAttention, width=WIDTH, heads=HEADS, head_dim=HEAD_DIM,
                             inner_width=INNER_WIDTH, attention_dropout=ATTENTION_DROPOUT,
                             scaling=HEAD_DIM ** -0.5)
    for name in ('q_proj', 'k_proj', 'v_proj'):
        setattr(block.attention, name, linear(WIDTH, INNER_WIDTH))
    block.attention.o_proj = linear(INNER_WIDTH, WIDTH)
    # Nonpersistent RoPE frequencies are architecture constants, not sampled parameters.
    block.attention.rotary = StudentRotaryEmbedding()
    block.mlp = module(StudentMLP, hidden_dropout=HIDDEN_DROPOUT)
    block.mlp.fc = linear(WIDTH, FFN_WIDTH)
    block.mlp.act = nn.GELU()
    block.mlp.proj = linear(FFN_WIDTH, WIDTH)
    block.assert_frozen_geometry()
    block.assert_frozen_parameter_identity()
    return block


__all__ = [
    "CANONICAL_QKV_REPRESENTATION", "EXPECTED_PARAMETER_COUNT", "INITIALIZATION_ORDER",
    "CMR1StudentBlock", "NarrowAttention", "StudentMLP", "StudentRotaryEmbedding",
    "StudentGeometryIdentityMismatch", "StudentParameterIdentityMismatch", "build_student",
    "rotate_half", "build_student_architecture",
]
