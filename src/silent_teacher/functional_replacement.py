"""Stage B functional replacement evaluator (pre-protocol; synthetic smoke tests only).

Replaces Teacher GPT-2 block 5 with a replacement module that outputs the FULL block output
(F = H + U), then continues through the frozen blocks 6-11, ln_f and the LM head. No Teacher
hidden state is re-injected after block 5, no calibration, no logits adapter.

Scoring (frozen now): for an L=128 token window, logits positions 0..126 predict tokens
1..127 — 127 scored tokens per window (future corrected PG-19 validation: 1600 windows ->
203,200 scored tokens; never computed in this pre-protocol round).

Metric definitions (frozen now):
- Primary Teacher-behavior fidelity: KL(p_T || p_S), full vocabulary, temperature 1, same
  scoring positions, aggregated as the mean over scored tokens.
- Top-1 agreement; Top-5 set overlap |Top5_T ∩ Top5_S| / 5 (NOT "top-5 accuracy").
- Language-model consequence: NLL, PPL = exp(global aggregate NLL), ΔNLL = NLL_S - NLL_T.
  NLL_S < NLL_T means the replacement is a better LANGUAGE MODEL, not that it is "closer to
  the Teacher" — KL and NLL are different evidence axes.
"""
import torch
import torch.nn.functional as F

SCORED_TOKENS_PER_WINDOW = 127


def scoring_positions(length=128):
    """(logit_positions, target_positions): 0..length-2 predict 1..length-1."""
    positions = torch.arange(length - 1)
    return positions, positions + 1


@torch.no_grad()
def run_teacher(model, token_ids):
    return model(token_ids).logits


@torch.no_grad()
def run_with_replacement(model, block5_replacement, token_ids, capture=None, block_index=5):
    """blocks 0..b-1 -> replacement full block output -> blocks b+1..N-1 -> ln_f -> lm_head.

    `block_index` is zero-based; the default 5 preserves the historical block-5 behaviour bitwise.
    `capture` (optional dict) records the replaced block's input/output pair, proving the pipeline
    consumed the replacement's own output (no Teacher hidden re-injection)."""
    hidden = model.transformer.wte(token_ids) + model.transformer.wpe(
        torch.arange(token_ids.shape[1], device=token_ids.device))
    hidden = model.transformer.drop(hidden)
    for index in range(block_index):
        hidden = model.transformer.h[index](hidden, attention_mask=None, use_cache=False)[0]
    block_input = hidden
    hidden = block5_replacement.full_output(hidden)
    if capture is not None:
        capture["block5_input"] = block_input.cpu().clone()
        capture["block5_output"] = hidden.cpu().clone()
    for index in range(block_index + 1, len(model.transformer.h)):
        hidden = model.transformer.h[index](hidden, attention_mask=None, use_cache=False)[0]
    hidden = model.transformer.ln_f(hidden)
    return model.lm_head(hidden)


@torch.no_grad()
def functional_metrics(teacher_logits, replaced_logits, target_token_ids):
    """Aggregate functional metrics over the shifted scoring positions.

    `target_token_ids`: [B, L] token ids (the original window); targets are tokens 1..L-1.
    Returns per-window-count aggregates; callers sum the `*_sum` fields across windows and
    divide by the total scored token count (global aggregation, never window averaging).
    """
    positions, targets = scoring_positions(teacher_logits.shape[1])
    t_logits = teacher_logits[:, positions, :].double()
    s_logits = replaced_logits[:, positions, :].double()
    target_ids = target_token_ids[:, targets]
    log_p_t = F.log_softmax(t_logits, dim=-1)
    log_p_s = F.log_softmax(s_logits, dim=-1)
    p_t = log_p_t.exp()
    kl_sum_per_token = (p_t * (log_p_t - log_p_s)).sum(dim=-1)          # KL(p_T || p_S)
    top1_agreement = (t_logits.argmax(dim=-1) == s_logits.argmax(dim=-1)).to(torch.float64)
    top5_t = t_logits.topk(5, dim=-1).indices
    top5_s = s_logits.topk(5, dim=-1).indices
    matches = (top5_t.unsqueeze(-1) == top5_s.unsqueeze(-2)).any(dim=-1)   # [B, P, 5]
    overlaps = matches.to(torch.float64).sum(dim=-1) / 5.0                 # [B, P] in {0, .., 1}
    token_log_p_t = torch.gather(log_p_t, -1, target_ids.unsqueeze(-1)).squeeze(-1)
    token_log_p_s = torch.gather(log_p_s, -1, target_ids.unsqueeze(-1)).squeeze(-1)
    nll_t_sum = float((-token_log_p_t).sum())
    nll_s_sum = float((-token_log_p_s).sum())
    scored = int(target_ids.numel())
    return {"scored_tokens": scored,
            "kl_sum": float(kl_sum_per_token.sum()), "kl_mean_over_tokens": float(kl_sum_per_token.mean()),
            "top1_agreement_sum": float(top1_agreement.sum()),
            "top5_overlap_sum": float(overlaps.sum()),
            "nll_teacher_sum": nll_t_sum, "nll_replaced_sum": nll_s_sum,
            "delta_nll_mean": (nll_s_sum - nll_t_sum) / scored,
            "aggregation": "sums over scored tokens; callers divide by total scored tokens "
                           "(PPL = exp(NLL_sum / scored_tokens)); KL is the mean over scored tokens"}


def aggregate_functional_results(per_window):
    """Global aggregation across windows: never an average of per-window PPLs."""
    scored = sum(entry["scored_tokens"] for entry in per_window)
    kl_sum = sum(entry["kl_sum"] for entry in per_window)
    nll_teacher_sum = sum(entry["nll_teacher_sum"] for entry in per_window)
    nll_replaced_sum = sum(entry["nll_replaced_sum"] for entry in per_window)
    top1_sum = sum(entry["top1_agreement_sum"] for entry in per_window)
    top5_sum = sum(entry["top5_overlap_sum"] for entry in per_window)
    nll_teacher = nll_teacher_sum / scored
    nll_replaced = nll_replaced_sum / scored
    return {"scored_tokens": scored, "kl_mean_over_tokens": kl_sum / scored,
            "top1_agreement": top1_sum / scored, "top5_set_overlap": top5_sum / scored,
            "nll_teacher": nll_teacher, "nll_replaced": nll_replaced,
            "delta_nll": nll_replaced - nll_teacher,
            "ppl_teacher": float(torch.exp(torch.tensor(nll_teacher))),
            "ppl_replaced": float(torch.exp(torch.tensor(nll_replaced))),
            "note": "KL(p_T||p_S) is the Teacher-behavior fidelity axis; NLL_S < NLL_T would mean "
                    "a better language model, not a closer Teacher match"}


@torch.no_grad()
def run_replacement_with_stages(model, block5_replacement, token_ids, block_index=5):
    """Like `run_with_replacement`, but captures per-stage hidden states for engineering
    equivalence checks: the replaced block's input, the replacement's output, the outputs after
    each later block, the ln_f hidden and the final logits.

    `block_index` is zero-based; the default 5 preserves the historical block-5 behaviour bitwise.
    Stage labels are `block{b}_input` / `block{b}_output` for the replaced block and
    `block{i}_output` for each later block."""
    stages = {}
    hidden = model.transformer.wte(token_ids) + model.transformer.wpe(
        torch.arange(token_ids.shape[1], device=token_ids.device))
    hidden = model.transformer.drop(hidden)
    for index in range(block_index):
        hidden = model.transformer.h[index](hidden, attention_mask=None, use_cache=False)[0]
    stages[f"block{block_index}_input"] = hidden.clone()
    hidden = block5_replacement.full_output(hidden)
    stages[f"block{block_index}_output"] = hidden.clone()
    for index in range(block_index + 1, len(model.transformer.h)):
        hidden = model.transformer.h[index](hidden, attention_mask=None, use_cache=False)[0]
        stages[f"block{index}_output"] = hidden.clone()
    hidden = model.transformer.ln_f(hidden)
    stages["ln_f"] = hidden.clone()
    stages["logits"] = model.lm_head(hidden)
    return stages


@torch.no_grad()
def run_teacher_with_stages(model, token_ids):
    """Native Teacher path with the same per-stage captures (blocks 0-4 -> native block 5 ->
    blocks 6-11 -> ln_f -> logits)."""
    stages = {}
    hidden = model.transformer.wte(token_ids) + model.transformer.wpe(
        torch.arange(token_ids.shape[1], device=token_ids.device))
    hidden = model.transformer.drop(hidden)
    for index in range(5):
        hidden = model.transformer.h[index](hidden, attention_mask=None, use_cache=False)[0]
    stages["block5_input"] = hidden.clone()
    hidden = model.transformer.h[5](hidden, attention_mask=None, use_cache=False)[0]
    stages["block5_output"] = hidden.clone()
    for index in range(6, 12):
        hidden = model.transformer.h[index](hidden, attention_mask=None, use_cache=False)[0]
        stages[f"block{index}_output"] = hidden.clone()
    hidden = model.transformer.ln_f(hidden)
    stages["ln_f"] = hidden.clone()
    stages["logits"] = model.lm_head(hidden)
    return stages


def stage_differences(teacher_stages, replacement_stages):
    """Per-stage max-abs and RMS differences (frozen FP32 engineering comparison)."""
    differences = {}
    for name, teacher_tensor in teacher_stages.items():
        difference = (teacher_tensor.double() - replacement_stages[name].double())
        differences[name] = {"max_abs": float(difference.abs().max()),
                             "rms": float(difference.square().mean().sqrt()),
                             "scale_max_abs": float(teacher_tensor.double().abs().max())}
    return differences
