"""Stage B full-block engineering module (pre-protocol; no main experiment this round).

Boundary: input H = block-5 input residual sequence [B, L, 768]. The Stage B main target is
the block UPDATE:

    A_T(H) = Attn_T(LN1_T(H), causal)          (output projection and bias included)
    R_T(H) = H + A_T(H)
    B_T(H) = MLP_T(LN2_T(R_T(H)))
    U_T(H) = A_T(H) + B_T(H)                   <- main target (branch accumulation)
    F_T(H) = H + U_T(H)                        (= Teacher block output; identity checked)

The update is accumulated directly from the branch outputs — never as a large subtraction
F - H (that identity is only used in tests, within FP32 tolerance).

Hard constraint: the Student MLP must read the Student's own residual R_S = H + A_S(H).
No Teacher intermediate state, adapters or hidden auxiliary paths.
"""
import hashlib

import torch
from torch import nn

from .attention import attention_math
from .records import file_hash
from .b0_phase1 import pooled_nmse, splice_future, splice_history

EPS = 1e-5
MLP_NEURONS = 1536
TEACHER_BLOCK_PARAMS = 7_087_872
STUDENT_BLOCK_PARAMS = 3_546_240
TOKENWISE_BLOCK_PARAMS = 3_545_856


def teacher_block_operators(block, hidden):
    """Full Teacher block-5 operator decomposition, branch-accumulated (no_grad by caller)."""
    attention = block.attn(block.ln_1(hidden), attention_mask=None, layer_past=None, head_mask=None,
                           use_cache=False)[0]
    residual = hidden + attention
    mlp = block.mlp(block.ln_2(residual))
    update = attention + mlp
    full = residual + mlp
    return {"attention": attention, "mlp": mlp, "update": update, "full": full, "residual": residual}


class FullBlockStudent(nn.Module):
    """The single Stage B main Student: LN1 + 6x64 causal attention + LN2 + 768-1536-768 MLP,
    full-block topology with the MLP reading the Student's own residual."""

    def __init__(self, width=768, heads=6, head_dim=64, mlp_hidden=1536):
        super().__init__()
        self.width, self.heads, self.head_dim = width, heads, head_dim
        internal = heads * head_dim
        self.ln = nn.LayerNorm(width, eps=EPS)
        self.qkv = nn.Linear(width, 3 * internal)
        self.out = nn.Linear(internal, width)
        self.ln2 = nn.LayerNorm(width, eps=EPS)
        self.fc = nn.Linear(width, mlp_hidden)
        self.proj = nn.Linear(mlp_hidden, width)
        self.act = nn.GELU(approximate="tanh")
        for layer in (self.qkv, self.out, self.fc, self.proj):
            nn.init.normal_(layer.weight, std=0.02)
            nn.init.zeros_(layer.bias)
        for norm in (self.ln, self.ln2):
            nn.init.ones_(norm.weight)
            nn.init.zeros_(norm.bias)

    def attention_update(self, hidden):
        projected = self.qkv(self.ln(hidden))
        return attention_math(projected, self.out.weight.T, self.out.bias, self.heads, self.head_dim,
                              hidden.shape[1])

    def mlp_update(self, residual):
        return self.proj(self.act(self.fc(self.ln2(residual))))

    def forward(self, hidden):
        attention = self.attention_update(hidden)
        residual = hidden + attention                     # the Student's OWN residual
        mlp = self.mlp_update(residual)                   # Student MLP reads Student R_S
        update = attention + mlp
        return update

    def full_output(self, hidden):
        return hidden + self.forward(hidden)


class TokenWiseFullBlock(nn.Module):
    """Parameter-matched position-independent full-block control: per position i,
    R_i = H_i + P1(LN1(H_i)); U_i = P1_out_i + P2(LN2(R_i)). Cannot read other positions."""

    def __init__(self, width=768, mlp_hidden=1536):
        super().__init__()
        self.ln = nn.LayerNorm(width, eps=EPS)
        self.p1_fc = nn.Linear(width, width)
        self.p1_proj = nn.Linear(width, width)
        self.ln2 = nn.LayerNorm(width, eps=EPS)
        self.fc = nn.Linear(width, mlp_hidden)
        self.proj = nn.Linear(mlp_hidden, width)
        self.act = nn.GELU(approximate="tanh")
        for layer in (self.p1_fc, self.p1_proj, self.fc, self.proj):
            nn.init.normal_(layer.weight, std=0.02)
            nn.init.zeros_(layer.bias)
        for norm in (self.ln, self.ln2):
            nn.init.ones_(norm.weight)
            nn.init.zeros_(norm.bias)

    def forward(self, hidden):
        shape = hidden.shape
        flat = hidden.reshape(-1, shape[-1])              # positions fully independent
        p1 = self.p1_proj(self.act(self.p1_fc(self.ln(flat))))
        residual = flat + p1
        mlp = self.proj(self.act(self.fc(self.ln2(residual))))
        update = (p1 + mlp).reshape(shape)
        return update

    def full_output(self, hidden):
        return hidden + self.forward(hidden)


def mlp_neuron_scores(block):
    """q_j = ||c_fc.weight[:, j]||_2 * ||c_proj.weight[j, :]||_2 for every Teacher MLP neuron."""
    fc_weight = block.mlp.c_fc.weight.detach()
    proj_weight = block.mlp.c_proj.weight.detach()
    # structural layout: c_fc (width, neurons) and c_proj (neurons, width) share the neuron axis
    assert fc_weight.shape[0] == proj_weight.shape[1] and fc_weight.shape[1] == proj_weight.shape[0]
    return (fc_weight.norm(dim=0) * proj_weight.norm(dim=1))


def select_mlp_neurons(block, count=None):
    """Top-`count` neurons by q_j descending; ties broken by ascending neuron index.
    Defaults to the frozen 1536 for GPT-2 Small (capped by the available neuron count)."""
    scores = mlp_neuron_scores(block)
    limit = count or MLP_NEURONS
    limit = min(limit, scores.shape[0])
    order = sorted(range(scores.shape[0]), key=lambda j: (-float(scores[j]), j))
    return order[:limit]


class ExactFullBlock(nn.Module):
    """Parameter-only copy of the full Teacher block (12-head attention + MLP). The Exact
    baseline: update output must reproduce the Teacher update, full output the Teacher output."""

    def __init__(self, teacher_block):
        super().__init__()
        self.heads = int(teacher_block.attn.num_heads)
        self.head_dim = int(teacher_block.attn.head_dim)
        for name, tensor in (("ln1_weight", teacher_block.ln_1.weight), ("ln1_bias", teacher_block.ln_1.bias),
                             ("qkv_weight", teacher_block.attn.c_attn.weight),
                             ("qkv_bias", teacher_block.attn.c_attn.bias),
                             ("out_weight", teacher_block.attn.c_proj.weight),
                             ("out_bias", teacher_block.attn.c_proj.bias),
                             ("ln2_weight", teacher_block.ln_2.weight), ("ln2_bias", teacher_block.ln_2.bias),
                             ("fc_weight", teacher_block.mlp.c_fc.weight),
                             ("fc_bias", teacher_block.mlp.c_fc.bias),
                             ("mlp_proj_weight", teacher_block.mlp.c_proj.weight),
                             ("mlp_proj_bias", teacher_block.mlp.c_proj.bias)):
            self.register_buffer(name, tensor.detach().clone())
        self.act = nn.GELU(approximate="tanh")

    def attention_update(self, hidden):
        normalised = torch.nn.functional.layer_norm(hidden, (hidden.shape[-1],), self.ln1_weight,
                                                    self.ln1_bias, EPS)
        projected = normalised @ self.qkv_weight + self.qkv_bias
        return attention_math(projected, self.out_weight, self.out_bias, self.heads, self.head_dim,
                              hidden.shape[1])

    def mlp_update(self, residual):
        normalised = torch.nn.functional.layer_norm(residual, (residual.shape[-1],), self.ln2_weight,
                                                    self.ln2_bias, EPS)
        return self.act(normalised @ self.fc_weight + self.fc_bias) @ self.mlp_proj_weight + self.mlp_proj_bias

    def forward(self, hidden):
        attention = self.attention_update(hidden)
        residual = hidden + attention
        mlp = self.mlp_update(residual)
        return attention + mlp

    def full_output(self, hidden):
        return hidden + self.forward(hidden)


class ZeroUpdateControl(nn.Module):
    """Identity / zero-update baseline: U = 0, so the block is an exact identity (F = H)."""

    def forward(self, hidden):
        return torch.zeros_like(hidden)

    def full_output(self, hidden):
        return hidden


class FullBlockParameterCopy(nn.Module):
    """Heuristic parameter-copy baseline (NOT optimal pruning): attention = B0's frozen top-6
    head norm-product rule (audited Conv1D orientation); MLP = top-1536 neurons by
    q_j = ||c_fc[:, j]|| * ||c_proj[j, :]||, ties by ascending index; copies LN1 and LN2.
    Full self-contained forward: copied attention residual -> copied LN2 -> copied MLP.
    No Teacher intermediate state is injected."""

    def __init__(self, teacher_block):
        super().__init__()
        self.heads = 6
        self.head_dim = int(teacher_block.attn.head_dim)
        width = teacher_block.ln_1.weight.shape[0]
        qkv_weight = teacher_block.attn.c_attn.weight.detach()
        qkv_bias = teacher_block.attn.c_attn.bias.detach()
        out_weight = teacher_block.attn.c_proj.weight.detach()
        self.register_buffer("ln1_weight", teacher_block.ln_1.weight.detach().clone())
        self.register_buffer("ln1_bias", teacher_block.ln_1.bias.detach().clone())
        head_scores = []
        for head in range(teacher_block.attn.num_heads):
            offset = head * self.head_dim
            q = qkv_weight[:, offset:offset + self.head_dim]
            k = qkv_weight[:, width + offset:width + offset + self.head_dim]
            v = qkv_weight[:, 2 * width + offset:2 * width + offset + self.head_dim]
            o = out_weight[offset:offset + self.head_dim, :]
            head_scores.append((-(float(torch.cat([q.flatten(), k.flatten(), v.flatten()]).norm() * o.norm())), head))
        selected = [head for _, head in sorted(head_scores)[:self.heads]]
        parts_weight, parts_bias = [], []
        for section in (0, width, 2 * width):
            for head in selected:
                columns = slice(section + head * self.head_dim, section + (head + 1) * self.head_dim)
                parts_weight.append(qkv_weight[:, columns])
                parts_bias.append(qkv_bias[columns])
        self.register_buffer("qkv_weight", torch.cat(parts_weight, dim=1))
        self.register_buffer("qkv_bias", torch.cat(parts_bias))
        rows = torch.cat([torch.arange(head * self.head_dim, (head + 1) * self.head_dim) for head in selected])
        self.register_buffer("out_weight", out_weight[rows, :].clone())
        self.register_buffer("out_bias", teacher_block.attn.c_proj.bias.detach().clone())
        neurons = select_mlp_neurons(teacher_block)
        self.register_buffer("ln2_weight", teacher_block.ln_2.weight.detach().clone())
        self.register_buffer("ln2_bias", teacher_block.ln_2.bias.detach().clone())
        self.register_buffer("fc_weight", teacher_block.mlp.c_fc.weight.detach()[:, neurons].clone())
        self.register_buffer("fc_bias", teacher_block.mlp.c_fc.bias.detach()[neurons].clone())
        self.register_buffer("mlp_proj_weight", teacher_block.mlp.c_proj.weight.detach()[neurons, :].clone())
        self.register_buffer("mlp_proj_bias", teacher_block.mlp.c_proj.bias.detach().clone())
        self.act = nn.GELU(approximate="tanh")
        self.selected_heads = selected
        self.selected_neurons = neurons

    def attention_update(self, hidden):
        normalised = torch.nn.functional.layer_norm(hidden, (hidden.shape[-1],), self.ln1_weight,
                                                    self.ln1_bias, EPS)
        projected = normalised @ self.qkv_weight + self.qkv_bias
        return attention_math(projected, self.out_weight, self.out_bias, self.heads, self.head_dim,
                              hidden.shape[1])

    def mlp_update(self, residual):
        normalised = torch.nn.functional.layer_norm(residual, (residual.shape[-1],), self.ln2_weight,
                                                    self.ln2_bias, EPS)
        return self.act(normalised @ self.fc_weight + self.fc_bias) @ self.mlp_proj_weight + self.mlp_proj_bias

    def forward(self, hidden):
        attention = self.attention_update(hidden)
        residual = hidden + attention
        mlp = self.mlp_update(residual)
        return attention + mlp

    def full_output(self, hidden):
        return hidden + self.forward(hidden)


class ExplicitBlockTarget:
    """Explicit, counted Stage B target callable.

    The trainer receives this object and NEVER reaches into a model for its target. Call
    accounting separates VERIFIER bookkeeping probes from SCIENTIFIC training target calls:
    `counters["verification_probe_calls"]` (identity/output checks) and
    `counters["training_target_calls"]` (loss evaluations). The proxy diagnostic requires
    the scientific full-Teacher training calls to be 0; the verifier's own teacher probes
    are bookkeeping and are reported separately."""

    def __init__(self, name, module, forward_fn, role="unspecified"):
        self.name = name
        self.role = role
        self.module = module
        self.forward_fn = forward_fn
        self.parameter_state_hash = state_hash_of(module)
        self.counters = {"verification_probe_calls": 0, "training_target_calls": 0}
        self._mode = "verification"

    def set_mode(self, mode):
        assert mode in ("verification", "training")
        self._mode = mode

    def __call__(self, hidden):
        self.counters[f"{self._mode}_probe_calls" if self._mode == "verification"
                      else "training_target_calls"] += 1
        return self.forward_fn(hidden)

    @property
    def call_count(self):
        return sum(self.counters.values())

    def output_sha256(self, hidden):
        with torch.no_grad():
            output = self.forward_fn(hidden).detach()
        return hashlib.sha256(output.cpu().contiguous().numpy().tobytes()).hexdigest()


def state_hash_of(module):
    digest = hashlib.sha256()
    for name, tensor in module.state_dict().items():
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def teacher_update_target_for_block(model, block_index):
    """Generalized Teacher residual-update target factory for a zero-based block index.

    `block_index=5` reproduces the historical block-5 target exactly.
    """
    block = model.transformer.h[block_index]
    return ExplicitBlockTarget(f"teacher_full_block_update_b{block_index}", block,
                               lambda hidden: teacher_block_operators(block, hidden)["update"],
                               role="main_stage_b_training_target")


def teacher_full_block_update_target(model):
    """Preserved entry point: the historical block-5 Teacher residual-update target."""
    return teacher_update_target_for_block(model, 5)


def proxy_full_block_update_target(proxy_module):
    return ExplicitBlockTarget("representable_proxy_full_block_update", proxy_module,
                               lambda hidden: proxy_module(hidden),
                               role="qualification_diagnostic_target")


def verify_target_wiring(record, claimed_proxy, teacher_block, probe, teacher_module_calls):
    """Hard wiring verifier (audit requirement): a trainer that CLAIMS the representable-proxy
    target but is actually wired to the full Teacher block must FAIL here — not just in logs.

    Checks: callable/name identity, parameter-state identity against the Teacher block,
    output-provenance identity against a recomputed Teacher probe output, a nonzero call
    count for the claimed target, and zero observed calls into the Teacher block-5 modules.
    """
    failures = []
    if record["target_name"] != claimed_proxy:
        failures.append(f"target name mismatch: {record['target_name']!r} != {claimed_proxy!r}")
    teacher_hash = state_hash_of(teacher_block)
    if record["parameter_state_hash"] == teacher_hash:
        failures.append("parameter_state_hash equals the full Teacher block (wrong target wired)")
    if record["call_count"] <= 0:
        failures.append("claimed target was never called")
    probe = probe.to(teacher_block.ln_1.weight.device)   # device-agnostic verification
    with torch.no_grad():
        teacher_probe = teacher_block_operators(teacher_block, probe)["update"]
    teacher_probe_sha = hashlib.sha256(teacher_probe.detach().cpu().contiguous()
                                       .numpy().tobytes()).hexdigest()
    if record["probe_output_sha256"] == teacher_probe_sha:
        failures.append("probe output hash equals the Teacher update output (wrong target wired)")
    if any(count > 0 for count in teacher_module_calls.values()):
        observed = {key: value for key, value in teacher_module_calls.items() if value > 0}
        failures.append(f"teacher block5 module calls observed during claimed-proxy training: {observed}")
    return {"passed": not failures, "failures": failures}


def load_parameters_into_student(proxy, student):
    """Copy a Student-isomorphic proxy's frozen parameters into a FullBlockStudent.

    Proxy buffers store Conv1D-style weights (in, out) applied as x @ W; nn.Linear applies
    x @ W.T, so every weight is transposed on load. Returns the student for chaining."""
    with torch.no_grad():
        student.ln.weight.copy_(proxy.ln1_weight)
        student.ln.bias.copy_(proxy.ln1_bias)
        student.qkv.weight.copy_(proxy.qkv_weight.T)
        student.qkv.bias.copy_(proxy.qkv_bias)
        student.out.weight.copy_(proxy.out_weight.T)
        student.out.bias.copy_(proxy.out_bias)
        student.ln2.weight.copy_(proxy.ln2_weight)
        student.ln2.bias.copy_(proxy.ln2_bias)
        student.fc.weight.copy_(proxy.fc_weight.T)
        student.fc.bias.copy_(proxy.fc_bias)
        student.proj.weight.copy_(proxy.mlp_proj_weight.T)
        student.proj.bias.copy_(proxy.mlp_proj_bias)
    return student


def verify_exact_load(proxy, student, probe, tolerance=1e-4):
    """An exact-loaded Student must reproduce the proxy outputs (an exact representable
    solution exists). Uses the documented cross-path FP32 tolerance."""
    with torch.no_grad():
        proxy_out = proxy(probe)
        student_out = student(probe)
    difference = float((proxy_out - student_out).abs().max())
    scale = float(proxy_out.abs().max())
    return {"max_abs_difference": difference, "scale": scale, "relative": difference / scale,
            "tolerance": tolerance,
            "passed": bool(difference <= tolerance and difference / scale <= 1e-5)}


@torch.no_grad()
def full_block_context_evaluation(update_target_fn, update_student_fn, pool, donor, queries=(63, 127),
                                  microbatch=8, uninformative_threshold=1e-4):
    """U-space context evaluation (frozen Phase 2 semantics applied to full-block updates).

    Returns per-query and pooled ContextNMSE, delta cosine, response squared-energy ratio,
    and the Teacher intervention energy ratio with its uninformative guard."""
    teacher_deltas, teacher_base = {}, {}
    for query in queries:
        spliced = splice_history(pool, donor, query)
        assert torch.equal(spliced[:, query], pool[:, query]), "query position must stay bit-identical"
        assert torch.equal(spliced[:, query + 1:], pool[:, query + 1:]), "future positions must be unchanged"
        base_chunks, delta_chunks = [], []
        for start in range(0, len(pool), microbatch):
            chunk = pool[start:start + microbatch]
            base_chunks.append(update_target_fn(chunk)[:, query].cpu())
            delta_chunks.append(update_target_fn(spliced[start:start + microbatch])[:, query].cpu())
        base = torch.cat(base_chunks)
        teacher_deltas[query] = torch.cat(delta_chunks) - base
        teacher_base[query] = base
    pooled_teacher = torch.cat([teacher_deltas[q] for q in queries])
    pooled_base = torch.cat([teacher_base[q] for q in queries])
    teacher_energy = float(pooled_teacher.double().square().sum() / pooled_base.double().square().sum())
    context = {"teacher_intervention_energy_ratio": teacher_energy,
               "uninformative": bool(teacher_energy < uninformative_threshold), "objects": {}}
    for name, student_fn in update_student_fn.items():
        student_deltas = {}
        for query in queries:
            spliced = splice_history(pool, donor, query)
            base_chunks, spliced_chunks = [], []
            for start in range(0, len(pool), microbatch):
                base_chunks.append(student_fn(pool[start:start + microbatch])[:, query].cpu())
                spliced_chunks.append(student_fn(spliced[start:start + microbatch])[:, query].cpu())
            student_deltas[query] = torch.cat(spliced_chunks) - torch.cat(base_chunks)
        all_student = torch.cat([student_deltas[q] for q in queries])
        all_teacher = torch.cat([teacher_deltas[q] for q in queries])
        metrics = pooled_nmse(all_student, all_teacher)
        flat_s, flat_t = all_student.double().flatten(), all_teacher.double().flatten()
        metrics["delta_cosine"] = (float(flat_s @ flat_t / (flat_s.norm() * flat_t.norm()))
                                   if float(flat_s.norm()) > 0 and float(flat_t.norm()) > 0 else None)
        for query in queries:
            metrics[f"t{query}"] = pooled_nmse(student_deltas[query], teacher_deltas[query])
        context["objects"][name] = metrics
    return context


@torch.no_grad()
def full_block_future_causality(update_fns, pool, donor, query=63, tolerance=1e-6, microbatch=8):
    """Future causality negative control: only [query+1, L) is replaced (must actually change
    the input) and every causal update function's output at `query` must be unchanged."""
    spliced = splice_future(pool, donor, query)
    assert torch.equal(spliced[:, :query + 1], pool[:, :query + 1]), "history must be unchanged"
    assert not torch.equal(spliced[:, query + 1:], pool[:, query + 1:]), "future input must actually change"
    worst = {}
    for name, fn in update_fns.items():
        maximum = 0.0
        for start in range(0, len(pool), microbatch):
            base = fn(pool[start:start + microbatch])[:, query]
            changed = fn(spliced[start:start + microbatch])[:, query]
            maximum = max(maximum, float((base - changed).abs().max()))
        worst[name] = maximum
    return {"query": query, "max_abs_change": worst, "tolerance": tolerance,
            "passed": bool(max(worst.values()) <= tolerance), "future_input_changed": True,
            "note": "t = L-1 replaces an empty slice and is a no-op; it is never evidence"}


def verify_audit_separation(record):
    """Hard guard for the audit-timing discipline (pre-protocol §23). `record` must certify:
    audit namespaces/rules frozen before training, fresh audit tensors materialized after the
    terminal checkpoints (or provably hash-only and trainer-inaccessible), and zero audit
    metric access before the evaluation phase."""
    failures = []
    if not record.get("audit_namespaces_frozen_before_training"):
        failures.append("audit namespace/rule was not frozen before training")
    if not record.get("audit_materialized_after_terminal_checkpoints") and \
            not record.get("audit_hash_only_and_trainer_inaccessible"):
        failures.append("audit tensors materialized early without a proven inaccessibility guarantee")
    if record.get("audit_accessible_during_training"):
        failures.append("audit tensors were accessible to the trainer")
    if record.get("audit_metric_access"):
        failures.append("audit metrics were computed or consulted before the evaluation phase")
    return {"passed": not failures, "failures": failures}


def target_identity_artifact(name, role, module, probe_input, source_provenance,
                             creation_code_path=None, teacher_revision=None):
    """Frozen target-identity artifact, created OUTSIDE the trainer (audit requirement).

    Records role/name, architecture type, parameter SHA-256, the fixed probe input hash and
    the probe output hash, intended source/provenance and the creation code hash. The
    trainer receives this artifact and independently verifies that the ACTUAL target
    matches it — the expectation never comes from the live target itself."""
    # ExplicitBlockTarget wrappers expose the underlying module; FullBlockParameterCopy holds
    # its frozen weights as BUFFERS (no parameters), so the device is taken from whichever
    # exists first.
    device_module = module.module if hasattr(module, "module") else module
    params = list(device_module.parameters())
    buffers = list(device_module.buffers())
    device = params[0].device if params else (buffers[0].device if buffers else probe_input.device)
    probe_input = probe_input.to(device)      # device-agnostic artifact creation
    with torch.no_grad():
        if hasattr(module, "forward_fn"):          # ExplicitBlockTarget wrapper
            probe_output = module.forward_fn(probe_input)
        else:
            probe_output = module(probe_input)
            if isinstance(probe_output, tuple):    # GPT-2 blocks return (hidden, presents)
                probe_output = probe_output[0]
    identity_module = module.module if hasattr(module, "module") else module
    return {
        "role": role, "name": name, "architecture": type(identity_module).__name__,
        "parameter_sha256": state_hash_of(identity_module) if hasattr(identity_module, "state_dict") else None,
        "probe_input_sha256": hashlib.sha256(probe_input.detach().cpu().contiguous()
                                             .numpy().tobytes()).hexdigest(),
        "probe_output_sha256": hashlib.sha256(probe_output.detach().cpu().contiguous()
                                              .numpy().tobytes()).hexdigest(),
        "source_provenance": source_provenance,
        "creation_code_sha256": (file_hash(creation_code_path) if creation_code_path else None),
        "teacher_revision": teacher_revision,
    }


def verify_target_identity(actual_target, expected_artifact, probe_input):
    """Independent verification: the ACTUAL target must match the FROZEN expected artifact.

    Positive identity matching (actual == frozen expected) — not merely 'not the Teacher'.
    Returns {"passed", "failures", "verification_probe_calls"}; a verifier probe increments
    the actual target's verification counter, never its training counter."""
    failures = []
    if hasattr(actual_target, "module") and hasattr(actual_target.module, "parameters"):
        try:
            device = next(actual_target.module.parameters()).device
            probe_input = probe_input.to(device)   # device-agnostic verification
        except StopIteration:
            pass
    with torch.no_grad():
        if hasattr(actual_target, "module"):
            actual_output = actual_target.forward_fn(probe_input)
            actual_param_hash = actual_target.parameter_state_hash
            actual_name = actual_target.name
        else:
            actual_output = actual_target(probe_input)
            actual_param_hash = state_hash_of(actual_target)
            actual_name = type(actual_target).__name__
    if actual_name != expected_artifact["name"]:
        failures.append(f"actual name {actual_name!r} != frozen expected {expected_artifact['name']!r}")
    if expected_artifact.get("parameter_sha256") is not None and             actual_param_hash != expected_artifact["parameter_sha256"]:
        failures.append("actual parameter SHA256 != frozen expected parameter SHA256")
    probe_input_sha = hashlib.sha256(probe_input.detach().cpu().contiguous()
                                     .numpy().tobytes()).hexdigest()
    if probe_input_sha != expected_artifact["probe_input_sha256"]:
        failures.append("probe input hash != frozen expected probe input hash "
                        "(expected/actual probe definitions must be identical)")
    probe_output_sha = hashlib.sha256(actual_output.detach().cpu().contiguous()
                                      .numpy().tobytes()).hexdigest()
    if probe_output_sha != expected_artifact["probe_output_sha256"]:
        failures.append("actual probe output SHA256 != frozen expected probe output SHA256")
    return {"passed": not failures, "failures": failures,
            "verification_probe_calls": 1}