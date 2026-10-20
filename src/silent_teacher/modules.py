import torch
from torch import nn
from transformers.activations import ACT2FN


class MLP(nn.Module):
    def __init__(self, width=768, intermediate=1536):
        super().__init__()
        self.fc = nn.Linear(width, intermediate)
        self.proj = nn.Linear(intermediate, width)
        self.act = ACT2FN["gelu_new"]
        for layer in (self.fc, self.proj):
            nn.init.normal_(layer.weight, std=0.02)
            nn.init.zeros_(layer.bias)

    def forward(self, h):
        return self.proj(self.act(self.fc(h)))

    @classmethod
    def from_hf(cls, source):
        result = cls(source.c_fc.weight.shape[0], source.c_fc.weight.shape[1])
        with torch.no_grad():
            result.fc.weight.copy_(source.c_fc.weight.T)
            result.fc.bias.copy_(source.c_fc.bias)
            result.proj.weight.copy_(source.c_proj.weight.T)
            result.proj.bias.copy_(source.c_proj.bias)
        return result.eval()


def freeze(model):
    return model.eval().requires_grad_(False)


def prefix_copy(teacher, units):
    """Prefix-copy diagnostic baseline (kept, not the formal weight-copy
    baseline): the frozen Teacher MLP restricted to its FIRST `units` hidden
    neurons, weights copied verbatim, full output bias kept."""
    result = MLP(teacher.fc.in_features, units)
    with torch.no_grad():
        result.fc.weight.copy_(teacher.fc.weight[:units])
        result.fc.bias.copy_(teacher.fc.bias[:units])
        result.proj.weight.copy_(teacher.proj.weight[:, :units])
        result.proj.bias.copy_(teacher.proj.bias)
    return freeze(result)


def weight_copy(teacher, units):
    """Formal weight-copy baseline (norm-product neuron selection).

    Select the `units` hidden neurons with the largest
    q_j = ||W1[:, j]||_2 * ||W2[j, :]||_2, sorted descending with ties broken
    deterministically by ascending neuron index; copy the corresponding input
    weights, hidden bias and output weights, and keep the full Teacher output
    bias. Untrained; selection is a fixed deterministic function of the frozen
    Teacher parameters.
    """
    w1, w2 = teacher.fc.weight, teacher.proj.weight  # [hidden, in], [out, hidden]
    q = w1.norm(dim=1) * w2.norm(dim=0)
    selected = torch.argsort(-q, stable=True)[:units]
    result = MLP(teacher.fc.in_features, units)
    with torch.no_grad():
        result.fc.weight.copy_(w1[selected])
        result.fc.bias.copy_(teacher.fc.bias[selected])
        result.proj.weight.copy_(w2[:, selected])
        result.proj.bias.copy_(teacher.proj.bias)
    return freeze(result)
