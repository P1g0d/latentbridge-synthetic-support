"""Stage B0 attention-branch modules: compressed Student, parameter-only controls and the
frozen Teacher target path.

Boundary discipline (protocols/stage_b0_attention_feasibility_v1.md): the input is the
block-5 input residual sequence H; the target is Teacher LN1 + attention (output projection
and bias included), with no residual addition, no ln_2 and no MLP.
"""
import math
import torch
from torch import nn

EPS = 1e-5


def causal_mask(length, device):
    """Boolean mask, True where attention is allowed (diagonal visible)."""
    return torch.tril(torch.ones(length, length, dtype=torch.bool, device=device))


def block_geometry(teacher_block):
    """Width, head count and head_dim inferred from the Teacher attention module."""
    heads = int(teacher_block.attn.num_heads)
    head_dim = int(teacher_block.attn.head_dim)
    width = int(teacher_block.ln_1.weight.shape[0])
    return width, heads, head_dim


def attention_math(qkv, out_weight, out_bias, heads, head_dim, length, bias_qkv=None):
    """QKV-in-one attention with causal masking; Conv1D-style out projection."""
    batch = qkv.shape[0]
    query, key, value = qkv.split(heads * head_dim, dim=2)
    query = query.view(batch, length, heads, head_dim).transpose(1, 2)
    key = key.view(batch, length, heads, head_dim).transpose(1, 2)
    value = value.view(batch, length, heads, head_dim).transpose(1, 2)
    scores = query @ key.transpose(-2, -1) / math.sqrt(head_dim)
    mask = causal_mask(length, qkv.device)
    scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
    weights = torch.softmax(scores, dim=-1)
    output = (weights @ value).transpose(1, 2).reshape(batch, length, heads * head_dim)
    return output @ out_weight + out_bias


class CompressedAttention(nn.Module):
    """Student: LN1 + 6-head / internal-width-384 attention branch (768 -> 384 -> 768)."""

    def __init__(self, width=768, heads=6, head_dim=64):
        super().__init__()
        self.width, self.heads, self.head_dim = width, heads, head_dim
        internal = heads * head_dim
        self.ln = nn.LayerNorm(width, eps=EPS)
        self.qkv = nn.Linear(width, 3 * internal)
        self.out = nn.Linear(internal, width)
        for layer in (self.qkv, self.out):
            nn.init.normal_(layer.weight, std=0.02)
            nn.init.zeros_(layer.bias)
        nn.init.ones_(self.ln.weight)
        nn.init.zeros_(self.ln.bias)

    def forward(self, hidden):
        length = hidden.shape[1]
        projected = self.qkv(self.ln(hidden))
        return attention_math(projected, self.out.weight.T, self.out.bias, self.heads, self.head_dim, length)


class ExactTeacherAttention(nn.Module):
    """Parameter-only copy of Teacher LN1 + 12-head attention (the Exact control)."""

    def __init__(self, teacher_block):
        super().__init__()
        self.width, self.heads, self.head_dim = block_geometry(teacher_block)
        self.ln_weight = nn.Parameter(teacher_block.ln_1.weight.detach().clone(), requires_grad=False)
        self.ln_bias = nn.Parameter(teacher_block.ln_1.bias.detach().clone(), requires_grad=False)
        self.qkv_weight = nn.Parameter(teacher_block.attn.c_attn.weight.detach().clone(), requires_grad=False)
        self.qkv_bias = nn.Parameter(teacher_block.attn.c_attn.bias.detach().clone(), requires_grad=False)
        self.out_weight = nn.Parameter(teacher_block.attn.c_proj.weight.detach().clone(), requires_grad=False)
        self.out_bias = nn.Parameter(teacher_block.attn.c_proj.bias.detach().clone(), requires_grad=False)

    def forward(self, hidden):
        length = hidden.shape[1]
        normalised = torch.nn.functional.layer_norm(hidden, (hidden.shape[-1],), self.ln_weight, self.ln_bias, EPS)
        projected = normalised @ self.qkv_weight + self.qkv_bias
        return attention_math(projected, self.out_weight, self.out_bias, self.heads, self.head_dim, length)


def head_scores(teacher_block):
    """s_r = ||[WQ_r, WK_r, WV_r]||_F * ||WO_r||_F for every Teacher head (Conv1D layout)."""
    width, heads, head_dim = block_geometry(teacher_block)
    qkv_weight = teacher_block.attn.c_attn.weight.detach()
    out_weight = teacher_block.attn.c_proj.weight.detach()
    scores = []
    # Conv1D weight layout: (in_features, out_features) = (width, 3*width); Q|K|V blocks lie along dim 1.
    assert qkv_weight.shape == (width, 3 * width)
    for head in range(heads):
        offset = head * head_dim
        q = qkv_weight[:, offset:offset + head_dim]
        k = qkv_weight[:, width + offset:width + offset + head_dim]
        v = qkv_weight[:, 2 * width + offset:2 * width + offset + head_dim]
        o = out_weight[offset:offset + head_dim, :]
        scores.append(float(torch.cat([q.flatten(), k.flatten(), v.flatten()]).norm() * o.norm()))
    return scores


def top_heads(teacher_block, count=6):
    """Top-`count` head indices by the frozen score rule, ties by ascending index."""
    _, heads, _ = block_geometry(teacher_block)
    scores = head_scores(teacher_block)
    order = sorted(range(heads), key=lambda head: (-scores[head], head))
    return order[:count], scores


class HeadCopyAttention(nn.Module):
    """Parameter-only 6-head baseline: Teacher LN1 + top-6 heads copied by the frozen rule."""

    def __init__(self, teacher_block, heads=6):
        super().__init__()
        width, teacher_heads, head_dim = block_geometry(teacher_block)
        assert teacher_heads >= heads, "teacher has fewer heads than the copy needs"
        self.width, self.heads, self.head_dim = width, heads, head_dim
        internal = heads * self.head_dim
        selected, scores = top_heads(teacher_block, heads)
        self.selected_heads = selected
        self.head_scores = scores
        qkv_weight = teacher_block.attn.c_attn.weight.detach()
        qkv_bias = teacher_block.attn.c_attn.bias.detach()
        out_weight = teacher_block.attn.c_proj.weight.detach()
        self.ln_weight = nn.Parameter(teacher_block.ln_1.weight.detach().clone(), requires_grad=False)
        self.ln_bias = nn.Parameter(teacher_block.ln_1.bias.detach().clone(), requires_grad=False)
        parts_weight, parts_bias = [], []
        for section in (0, width, 2 * width):        # Q, K, V blocks along dim 1 (Conv1D layout)
            for head in selected:
                columns = slice(section + head * head_dim, section + (head + 1) * head_dim)
                parts_weight.append(qkv_weight[:, columns])
                parts_bias.append(qkv_bias[columns])
        self.qkv_weight = nn.Parameter(torch.cat(parts_weight, dim=1), requires_grad=False)
        self.qkv_bias = nn.Parameter(torch.cat(parts_bias), requires_grad=False)
        rows = torch.cat([torch.arange(head * head_dim, (head + 1) * head_dim) for head in selected])
        self.out_weight = nn.Parameter(out_weight[rows, :].clone(), requires_grad=False)
        self.out_bias = nn.Parameter(teacher_block.attn.c_proj.bias.detach().clone(), requires_grad=False)
        assert self.qkv_weight.shape == (width, 3 * internal)

    def forward(self, hidden):
        length = hidden.shape[1]
        normalised = torch.nn.functional.layer_norm(hidden, (hidden.shape[-1],), self.ln_weight, self.ln_bias, EPS)
        projected = normalised @ self.qkv_weight + self.qkv_bias
        return attention_math(projected, self.out_weight, self.out_bias, self.heads, self.head_dim, length)


class FullAttention(nn.Module):
    """Independent 12-head attention Student (conditional capacity control), same interfaces."""

    def __init__(self, width=768, heads=12, head_dim=64):
        super().__init__()
        self.heads, self.head_dim = heads, head_dim
        internal = heads * head_dim
        self.ln = nn.LayerNorm(width, eps=EPS)
        self.qkv = nn.Linear(width, 3 * internal)
        self.out = nn.Linear(internal, width)
        for layer in (self.qkv, self.out):
            nn.init.normal_(layer.weight, std=0.02)
            nn.init.zeros_(layer.bias)
        nn.init.ones_(self.ln.weight)
        nn.init.zeros_(self.ln.bias)

    def forward(self, hidden):
        length = hidden.shape[1]
        projected = self.qkv(self.ln(hidden))
        return attention_math(projected, self.out.weight.T, self.out.bias, self.heads, self.head_dim, length)


class TokenWiseControl(nn.Module):
    """Position-independent control: LN -> Linear -> gelu_new -> Linear (≈1.183M parameters)."""

    def __init__(self, width=768):
        super().__init__()
        self.ln = nn.LayerNorm(width, eps=EPS)
        self.first = nn.Linear(width, width)
        self.second = nn.Linear(width, width)
        self.act = nn.GELU(approximate="tanh")
        for layer in (self.first, self.second):
            nn.init.normal_(layer.weight, std=0.02)
            nn.init.zeros_(layer.bias)
        nn.init.ones_(self.ln.weight)
        nn.init.zeros_(self.ln.bias)

    def forward(self, hidden):
        return self.second(self.act(self.first(self.ln(hidden))))


class ZeroControl(nn.Module):
    def forward(self, hidden):
        return torch.zeros_like(hidden)


class ConstantControl(nn.Module):
    def __init__(self, vector):
        super().__init__()
        self.vector = nn.Parameter(vector.detach().clone(), requires_grad=False)

    def forward(self, hidden):
        return self.vector.view(1, 1, -1).expand_as(hidden)


def native_prefix_input(model, inputs_embeds, block_index):
    """Frozen Teacher blocks 0..block_index-1 only: produces the block-`block_index` input residual.

    `block_index` is a zero-based block index (5 == the historical block-5 boundary). The
    arithmetic, the block order and the no-cache setting are identical to the historical
    block-5-only implementation, so `block_index=5` reproduces it bitwise.
    """
    with torch.no_grad():
        hidden = model.transformer.drop(inputs_embeds)
        for index in range(block_index):
            hidden = model.transformer.h[index](hidden, attention_mask=None, use_cache=False)[0]
    return hidden


def native_block5_input(model, inputs_embeds):
    """Frozen Teacher blocks 0-4 only: produces the block-5 input residual sequence.

    Deliberately never touches any h[5] submodule — that is the property the B-Native
    generator must satisfy (protocol engineering gate).

    Preserved entry point: it is exactly `native_prefix_input` at `block_index=5`.
    """
    return native_prefix_input(model, inputs_embeds, 5)


def teacher_target(teacher_block, hidden):
    """Frozen Teacher target path: LN1_T then attention (projection + bias). No residual/ln_2/MLP."""
    with torch.no_grad():
        normalised = teacher_block.ln_1(hidden)
        output = teacher_block.attn(normalised, attention_mask=None, layer_past=None, head_mask=None,
                                    use_cache=False)
    return output[0] if isinstance(output, tuple) else output


def parameter_count(module):
    return sum(parameter.numel() for parameter in module.parameters())
