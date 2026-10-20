"""H9*-ID2 frozen analysis: co-primary endpoints, noninferiority, material loss, reference guards
and the four-state terminal classification.

Implements exactly protocol v0.4 sections 4-8 as accepted:

  C_k(S)      = (ContextNMSE_{k,63} + ContextNMSE_{k,127}) / 2      (equal-query mean)
  C_P(S)      = equal-environment mean over C4 / peS2o / StackExchange
  D_value     = R_P(S_pos) - R_P(S_nat)
  D_context   = C_P(S_pos) - C_P(S_nat)
  NI_e        iff mean(D_e) + t(0.95, 47) * SE(D_e)  <  delta_e     (STRICT)
  ML_e        iff mean(D_e) - t(0.975, 47) * SE(D_e) >  delta_e     (STRICT, Bonferroni over 2)
  GUARD_value iff U95(native value)   <  a_value   + 0.020
  GUARD_ctx   iff U95(native context) <  a_context + 0.015
  precedence  : NOT_ESTIMABLE > MATERIAL_LOSS > BOTH_AGGREGATE_ENDPOINTS_PRESERVED > INCONCLUSIVE

Nothing here runs on real ID2 outcomes in this phase.
"""
import math

M = 48
DF = M - 1
DELTA_VALUE = 0.020
DELTA_CONTEXT = 0.015
A_VALUE = 0.6543170287510748
A_CONTEXT = 0.706632475320574
ENVIRONMENTS = ("c4", "pes2o", "se")

TERMINAL_STATES = ("NOT_ESTIMABLE", "MATERIAL_LOSS", "BOTH_AGGREGATE_ENDPOINTS_PRESERVED",
                   "INCONCLUSIVE")
INCONCLUSIVE_REASONS = ("PRECISION", "REFERENCE_NOT_REESTABLISHED", "MIXED_ENDPOINTS")


# --------------------------------------------------------------------------------- t quantiles
def _betacf(a, b, x, itmax=300, eps=3e-16):
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > 1e-30 else 1e-30)
    h = d
    for m in range(1, itmax + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        c = 1.0 + aa / c
        d = 1.0 / (d if abs(d) > 1e-30 else 1e-30)
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        c = 1.0 + aa / c
        d = 1.0 / (d if abs(d) > 1e-30 else 1e-30)
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def _betai(a, b, x):
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    front = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
                     + a * math.log(x) + b * math.log(1.0 - x))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def t_cdf(t, df):
    tail = 0.5 * _betai(df / 2.0, 0.5, df / (df + t * t))
    return 1.0 - tail if t > 0 else tail


def t_quantile(p, df):
    lo, hi = 0.0, 1000.0
    for _ in range(300):
        mid = 0.5 * (lo + hi)
        if t_cdf(mid, df) < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


T95 = t_quantile(0.95, DF)
T975 = t_quantile(0.975, DF)


# ---------------------------------------------------------------------------------- endpoints
def context_endpoint(per_environment_queries):
    """`C_P = mean_k (ContextNMSE_{k,63} + ContextNMSE_{k,127}) / 2` over the frozen environments."""
    missing = [k for k in ENVIRONMENTS if k not in per_environment_queries]
    assert not missing, f"missing environments: {missing}"
    per_env = [(per_environment_queries[k]["t63"] + per_environment_queries[k]["t127"]) / 2.0
               for k in ENVIRONMENTS]
    return math.fsum(per_env) / len(per_env)


def value_endpoint(per_environment_nmse):
    missing = [k for k in ENVIRONMENTS if k not in per_environment_nmse]
    assert not missing, f"missing environments: {missing}"
    return math.fsum(per_environment_nmse[k] for k in ENVIRONMENTS) / len(ENVIRONMENTS)


def paired_differences(pos_endpoints, nat_endpoints):
    assert len(pos_endpoints) == len(nat_endpoints) == M, \
        f"expected {M} paired values, got {len(pos_endpoints)}/{len(nat_endpoints)}"
    return [pos_endpoints[i] - nat_endpoints[i] for i in range(M)]


def _finite(values):
    return all(isinstance(v, (int, float)) and math.isfinite(v) for v in values)


def statistics_of(values, t_crit=T95, df=DF):
    n = len(values)
    mean = math.fsum(values) / n
    sd = math.sqrt(math.fsum((v - mean) ** 2 for v in values) / (n - 1)) if n > 1 else 0.0
    se = sd / math.sqrt(n)
    return {"m": n, "df": df, "mean": mean, "sd": sd, "se": se, "t_crit": t_crit,
            "L95": mean - t_crit * se, "U95": mean + t_crit * se}


def ni_passes(u95, delta_e):
    """STRICT `<`: equality at the margin does not pass."""
    return bool(u95 < delta_e)


def ml_passes(l95, delta_e):
    """STRICT `>`: equality at the margin does not pass."""
    return bool(l95 > delta_e)


def noninferiority(deltas, delta_e):
    stats = statistics_of(deltas)
    return {**stats, "delta": delta_e, "passes": ni_passes(stats["U95"], delta_e)}


def material_loss(deltas, delta_e):
    stats = statistics_of(deltas, t_crit=T975)
    return {**stats, "delta": delta_e, "passes": ml_passes(stats["L95"], delta_e)}


def reference_guard(native_values, a_reference, margin):
    """`U95(native) < a_reference + margin`, one-sided 95% upper bound over the 48 native values."""
    stats = statistics_of(native_values)
    return {**stats, "a_reference": a_reference, "margin": margin,
            "threshold": a_reference + margin, "passes": bool(stats["U95"] < a_reference + margin)}


# ------------------------------------------------------------------------------ classification
def classify(deltas_value=None, deltas_context=None, native_value=None,
             native_context=None, unavailable_reason=None):
    """The frozen four-state classification with strict inequalities and fixed precedence."""
    if unavailable_reason is not None:
        return {"state": "NOT_ESTIMABLE", "reason": unavailable_reason,
                "detail": "confirmatory paired analysis cannot be completed"}
    for name, series in (("D_value", deltas_value), ("D_context", deltas_context),
                         ("native_value", native_value), ("native_context", native_context)):
        if series is None:
            return {"state": "NOT_ESTIMABLE", "reason": f"{name} missing",
                    "detail": "a required paired endpoint is unavailable"}
        if len(series) != M:
            return {"state": "NOT_ESTIMABLE", "reason": f"{name} has {len(series)} values, need {M}",
                    "detail": "a required paired endpoint is incomplete"}
        if not _finite(series):
            return {"state": "NOT_ESTIMABLE", "reason": f"{name} non-finite",
                    "detail": "a required paired endpoint is non-finite"}

    ni_value = noninferiority(deltas_value, DELTA_VALUE)
    ni_context = noninferiority(deltas_context, DELTA_CONTEXT)
    ml_value = material_loss(deltas_value, DELTA_VALUE)
    ml_context = material_loss(deltas_context, DELTA_CONTEXT)
    guard_value = reference_guard(native_value, A_VALUE, DELTA_VALUE)
    guard_context = reference_guard(native_context, A_CONTEXT, DELTA_CONTEXT)

    evidence = {"noninferiority": {"value": ni_value, "context": ni_context},
                "material_loss": {"value": ml_value, "context": ml_context},
                "reference_guards": {"value": guard_value, "context": guard_context}}

    if ml_value["passes"] or ml_context["passes"]:
        profile = ("both" if (ml_value["passes"] and ml_context["passes"])
                   else "value" if ml_value["passes"] else "context")
        return {"state": "MATERIAL_LOSS", "endpoint_profile": profile, "reason": None,
                "evidence": evidence}
    if ni_value["passes"] and ni_context["passes"] and guard_value["passes"] \
            and guard_context["passes"]:
        return {"state": "BOTH_AGGREGATE_ENDPOINTS_PRESERVED", "reason": None, "evidence": evidence}
    if not (guard_value["passes"] and guard_context["passes"]):
        reason = "REFERENCE_NOT_REESTABLISHED"
    elif ni_value["passes"] != ni_context["passes"]:
        reason = "MIXED_ENDPOINTS"
    else:
        reason = "PRECISION"
    return {"state": "INCONCLUSIVE", "reason": reason, "evidence": evidence}


# ----------------------------------------------------------------------------------- fixtures
def _series(mean, sd, n=M, seed=0):
    """Deterministic, dependency-free pseudo-sample with an exact mean and sample SD.

    The offset pattern is antisymmetric (`+v` for each `-v`), so `fsum` cancels exactly and the
    sample mean is exactly `mean`; the scale is solved so the sample SD is exactly `sd` up to a
    single float rounding. No RNG is used, so the fixtures are reproducible by construction.
    """
    if n % 2:
        raise ValueError("n must be even")
    half = [(k + 1) / (n // 2) for k in range(n // 2)]
    offsets = half + [-v for v in half]
    scale = sd / math.sqrt(math.fsum(o * o for o in offsets) / (n - 1))
    return [mean + scale * o for o in offsets]


def analysis_fixtures():
    """The accepted synthetic classifier fixtures. No real ID2 outcome is involved."""
    fixtures = {}

    def add(name, expected_state, **kwargs):
        fixtures[name] = {**kwargs, "expected_state": expected_state}

    def _placeholder():        # keeps the helper signature explicit for reviewers
        return None

    add("F1_both_preserved", "BOTH_AGGREGATE_ENDPOINTS_PRESERVED", expected_profile=None,
        deltas_value=_series(0.000, 0.005), deltas_context=_series(0.000, 0.012),
        native_value=_series(A_VALUE, 0.0122), native_context=_series(A_CONTEXT, 0.0299))
    add("F2_material_loss_value_only", "MATERIAL_LOSS", expected_profile="value",
        deltas_value=_series(0.050, 0.002), deltas_context=_series(0.000, 0.010),
        native_value=_series(A_VALUE, 0.0122), native_context=_series(A_CONTEXT, 0.0299))
    add("F3_material_loss_context_only", "MATERIAL_LOSS", expected_profile="context",
        deltas_value=_series(0.000, 0.005), deltas_context=_series(0.040, 0.004),
        native_value=_series(A_VALUE, 0.0122), native_context=_series(A_CONTEXT, 0.0299))
    add("F4_material_loss_both", "MATERIAL_LOSS", expected_profile="both",
        deltas_value=_series(0.050, 0.002), deltas_context=_series(0.040, 0.004),
        native_value=_series(A_VALUE, 0.0122), native_context=_series(A_CONTEXT, 0.0299))
    add("F5_inconclusive_precision", "INCONCLUSIVE", expected_reason="PRECISION",
        deltas_value=_series(0.000, 0.30), deltas_context=_series(0.000, 0.30),
        native_value=_series(A_VALUE, 0.0122), native_context=_series(A_CONTEXT, 0.0299))
    add("F6_inconclusive_reference_not_reestablished", "INCONCLUSIVE",
        expected_reason="REFERENCE_NOT_REESTABLISHED",
        deltas_value=_series(0.000, 0.005), deltas_context=_series(0.000, 0.012),
        native_value=_series(A_VALUE + 0.20, 0.0122), native_context=_series(A_CONTEXT, 0.0299))
    # context delta centred exactly on its margin: NI fails (U95 > delta), ML fails (L95 < delta)
    add("F7_inconclusive_mixed_endpoints", "INCONCLUSIVE", expected_reason="MIXED_ENDPOINTS",
        deltas_value=_series(0.000, 0.005), deltas_context=_series(0.015, 0.002),
        native_value=_series(A_VALUE, 0.0122), native_context=_series(A_CONTEXT, 0.0299))
    add("F8_not_estimable_nonfinite", "NOT_ESTIMABLE", expected_reason_contains="non-finite",
        deltas_value=_series(0.000, 0.005), deltas_context=_series(0.000, 0.012),
        native_value=_series(A_VALUE, 0.0122), native_context=_series(A_CONTEXT, 0.0299),
        inject_nonfinite=True)
    add("F9_not_estimable_missing_arm", "NOT_ESTIMABLE", expected_reason_contains="terminal",
        unavailable_reason="one Student in the pair has no terminal status")
    return fixtures


def boundary_fixtures():
    """Strict-inequality boundary fixtures plus the precedence case."""
    out = {}

    # U95 exactly equal to the margin -> NI must NOT pass (strict <)
    se_target = 0.001
    delta = DELTA_VALUE
    mean = delta - T95 * se_target
    out["F10_ni_boundary_equal"] = {
        "mean": mean, "se": se_target, "delta": delta,
        "expected_ni": False,
        "note": "U95 == delta exactly; NI_e requires U95 < delta, so it must not pass"}

    # L95 exactly equal to the margin -> ML must NOT pass (strict >)
    mean_ml = delta + T975 * se_target
    out["F11_ml_boundary_equal"] = {
        "mean": mean_ml, "se": se_target, "delta": delta,
        "expected_ml": False,
        "note": "L95 == delta exactly; ML_e requires L95 > delta, so it must not pass"}
    return out


def run_fixtures():
    """Run every classifier fixture. Single source of truth for P5 and the evaluator's inspection
    mode, so the two can never report different states for the same fixture.
    """
    results, failures = {}, []
    for name, fixture in analysis_fixtures().items():
        kwargs = {k: v for k, v in fixture.items()
                  if k not in ("expected_state", "expected_reason", "expected_reason_contains",
                               "expected_profile")}
        if kwargs.pop("inject_nonfinite", False) and kwargs.get("deltas_value") is not None:
            kwargs["deltas_value"] = list(kwargs["deltas_value"])
            kwargs["deltas_value"][0] = float("nan")
        outcome = classify(**kwargs)
        entry = {"state": outcome["state"], "reason": outcome.get("reason"),
                 "profile": outcome.get("endpoint_profile"),
                 "expected": fixture["expected_state"]}
        checks = [outcome["state"] == fixture["expected_state"]]
        if fixture.get("expected_reason"):
            checks.append(outcome.get("reason") == fixture["expected_reason"])
        if fixture.get("expected_reason_contains"):
            checks.append(fixture["expected_reason_contains"] in (outcome.get("reason") or ""))
        if fixture.get("expected_profile"):
            checks.append(outcome.get("endpoint_profile") == fixture["expected_profile"])
        entry["passed"] = all(checks)
        results[name] = entry
        if not entry["passed"]:
            failures.append(name)
    return {"fixtures": results, "failed": failures, "passed": not failures}


def _synth_from_mean_se(mean, se, n=M):
    """A series with exactly the requested mean and SE (hence the requested bounds)."""
    sd = se * math.sqrt(n)
    return _series(mean, sd, n=n)
