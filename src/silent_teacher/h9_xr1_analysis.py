"""H9*-XR1 frozen analysis: df = 23 endpoints, the Native qualification guard, material loss,
preservation and the four-state terminal classification.

Implements `protocols/h9_xr1_block9_replication_v0_2.md` §6-§9 exactly:

    N_i,e = Native NMSE          Q_i,e = Q_pos NMSE
    G_i,e = 1 - N_i,e            D_i,e = Q_i,e - N_i,e
    Z_i,e = D_i,e - 0.05*G_i,e = Q_i,e - 0.95*N_i,e - 0.05

    Native qualification : L95(mean(G_e)) > 0.20  <=>  U95(mean(N_e)) < 0.80   (SINGLE-ARM variance)
    Material loss        : L97.5(mean(Z_e)) > 0                                (either endpoint)
    Preservation         : U95(mean(Z_e)) < 0                                  (both endpoints)
    precedence           : NATIVE_NOT_QUALIFIED > MATERIAL_LOSS >
                           BOTH_ENDPOINTS_PRESERVED > INCONCLUSIVE

All bounds are STRICT. No historical block-5 anchor and no ID2 absolute margin enters this module:
`theta_e = E[D_e]/E[G_e]` is a ratio of expectations, and the only fixed constant beyond df is the
materiality tolerance rho = 0.05 and the preregistered 0.20 Native operating criterion.

Only the pure quantile mathematics and the two endpoint aggregators are imported from the frozen
H9*-ID2 analysis module. `M`, `DF`, `DELTA_*` and `A_*` are deliberately NOT imported.
"""
import math

from .h9_id2_analysis import (ENVIRONMENTS, _betacf, _betai,  # noqa: F401  (reused unchanged)
                              context_endpoint, t_cdf, t_quantile, value_endpoint)

# ---------------------------------------------------------------------------------- constants
M = 24
DF = M - 1
RHO = 0.05                       # frozen materiality tolerance (protocol §6)
NATIVE_MIN_RECOVERY = 0.20       # preregistered Native operating criterion (protocol §7)
NATIVE_MAX_NMSE = 1.0 - NATIVE_MIN_RECOVERY
SQRT_M = math.sqrt(M)

#: computed from the frozen quantile machinery rather than typed from the protocol text
T95 = t_quantile(0.95, DF)       # 1.7138715277470475  (protocol states ...048)
T975 = t_quantile(0.975, DF)     # 2.0686576104190477  (protocol states ...0486)
MULT_U95 = T95 / SQRT_M          # 0.3498425606387107  (protocol states 0.349843)
MULT_L975 = T975 / SQRT_M        # 0.4222629665043181  (protocol states 0.422263)
PROTOCOL_CONSTANTS = {"t_0.95_23": 1.713871527747048, "t_0.975_23": 2.0686576104190486,
                      "multiplier_u95": 0.349843, "multiplier_l975": 0.422263}

TERMINAL_STATES = ("NATIVE_NOT_QUALIFIED", "MATERIAL_LOSS", "BOTH_ENDPOINTS_PRESERVED",
                   "INCONCLUSIVE")
ENDPOINTS = ("value", "context")


def constants_identity(tolerance=1e-12):
    """P1: the df=23 constants must equal the frozen protocol values within a stated tolerance.

    Not bitwise: the published constants come from an independent implementation, and the frozen
    `t_quantile` bisection agrees to 15 significant figures.
    """
    checks = {
        "t_0.95_23": abs(T95 - PROTOCOL_CONSTANTS["t_0.95_23"]) <= tolerance,
        "t_0.975_23": abs(T975 - PROTOCOL_CONSTANTS["t_0.975_23"]) <= tolerance,
        "multiplier_u95": round(MULT_U95, 6) == PROTOCOL_CONSTANTS["multiplier_u95"],
        "multiplier_l975": round(MULT_L975, 6) == PROTOCOL_CONSTANTS["multiplier_l975"],
        "cdf_at_t95": abs(t_cdf(T95, DF) - 0.95) < 1e-12,
        "cdf_at_t975": abs(t_cdf(T975, DF) - 0.975) < 1e-12,
    }
    return {"passed": all(checks.values()), "checks": checks, "tolerance": tolerance,
            "computed": {"t_0.95_23": T95, "t_0.975_23": T975, "multiplier_u95": MULT_U95,
                         "multiplier_l975": MULT_L975, "m": M, "df": DF}}


# ---------------------------------------------------------------------------------- statistics
def statistics_of(values, t_crit=T95, df=DF):
    """Sample statistics with an XR1 default df.

    The frozen H9*-ID2 `statistics_of` defaults to df = 47, so this wrapper exists to make that
    impossible to inherit by accident: XR1 always passes df = 23.
    """
    n = len(values)
    mean = math.fsum(values) / n
    sd = math.sqrt(math.fsum((v - mean) ** 2 for v in values) / (n - 1)) if n > 1 else 0.0
    se = sd / math.sqrt(n)
    return {"m": n, "df": df, "mean": mean, "sd": sd, "se": se, "t_crit": t_crit,
            "L95": mean - t_crit * se, "U95": mean + t_crit * se}


def paired_quantities(native, pos):
    """N, Q, G, D, Z for one endpoint, from the 24 paired replicate values."""
    assert len(native) == len(pos) == M, f"expected {M} paired values, got {len(native)}/{len(pos)}"
    n = list(native)
    q = list(pos)
    g = [1.0 - value for value in n]
    d = [q[i] - n[i] for i in range(M)]
    z = [d[i] - RHO * g[i] for i in range(M)]
    return {"N": n, "Q": q, "G": g, "D": d, "Z": z}


def native_qualification(native_values, g_values=None):
    """`L95(mean(G)) > 0.20`, equivalently `U95(mean(N)) < 0.80`, using the SINGLE-ARM variance."""
    g = g_values if g_values is not None else [1.0 - value for value in native_values]
    stats_g = statistics_of(g)
    stats_n = statistics_of(native_values)
    return {"G": stats_g, "N": stats_n,
            "L95_G": stats_g["L95"], "U95_N": stats_n["U95"],
            "threshold_G": NATIVE_MIN_RECOVERY, "threshold_N": NATIVE_MAX_NMSE,
            "passes": bool(stats_g["L95"] > NATIVE_MIN_RECOVERY),
            "consistent_with_N": bool((stats_g["L95"] > NATIVE_MIN_RECOVERY)
                                      == (stats_n["U95"] < NATIVE_MAX_NMSE))}


def material_loss(z_values):
    """`L97.5(mean(Z)) > 0` — STRICT; equality at zero does not pass."""
    stats = statistics_of(z_values, t_crit=T975)
    return {**stats, "passes": bool(stats["L95"] > 0.0), "rule": "L97.5(mean(Z)) > 0, strict"}


def preservation(z_values):
    """`U95(mean(Z)) < 0` — STRICT; equality at zero does not pass."""
    stats = statistics_of(z_values)
    return {**stats, "passes": bool(stats["U95"] < 0.0), "rule": "U95(mean(Z)) < 0, strict"}


# ------------------------------------------------------------------------------- classification
def _finite(values):
    return all(isinstance(v, (int, float)) and math.isfinite(v) for v in values)


def classify(native=None, pos=None, unavailable_reason=None):
    """The frozen four-state classification, strict inequalities, fixed precedence.

    `native` / `pos` are `{endpoint: [24 values]}`. Unlike the H9*-ID2 classifier there is no
    NOT_ESTIMABLE state; an incomplete input raises rather than inventing a fifth class.
    """
    if unavailable_reason is not None:
        raise ValueError(f"XR1 has no NOT_ESTIMABLE state: {unavailable_reason}")
    for endpoint in ENDPOINTS:
        if not native or endpoint not in native or not pos or endpoint not in pos:
            raise ValueError(f"endpoint {endpoint!r} is missing")
        for label, series in (("native", native[endpoint]), ("pos", pos[endpoint])):
            if len(series) != M:
                raise ValueError(f"{label}/{endpoint} has {len(series)} values, need {M}")
            if not _finite(series):
                raise ValueError(f"{label}/{endpoint} is non-finite")

    quantities = {endpoint: paired_quantities(native[endpoint], pos[endpoint])
                  for endpoint in ENDPOINTS}
    guard = {endpoint: native_qualification(native[endpoint], quantities[endpoint]["G"])
             for endpoint in ENDPOINTS}
    loss = {endpoint: material_loss(quantities[endpoint]["Z"]) for endpoint in ENDPOINTS}
    keep = {endpoint: preservation(quantities[endpoint]["Z"]) for endpoint in ENDPOINTS}
    evidence = {"quantities": quantities, "native_qualification": guard,
                "material_loss": loss, "preservation": keep}

    if not (guard["value"]["passes"] and guard["context"]["passes"]):
        failed = [e for e in ENDPOINTS if not guard[e]["passes"]]
        return {"state": "NATIVE_NOT_QUALIFIED", "reason": f"guard failed on {', '.join(failed)}",
                "endpoint_profile": None, "evidence": evidence}
    if loss["value"]["passes"] or loss["context"]["passes"]:
        profile = ("both" if (loss["value"]["passes"] and loss["context"]["passes"])
                   else "value" if loss["value"]["passes"] else "context")
        return {"state": "MATERIAL_LOSS", "reason": None, "endpoint_profile": profile,
                "evidence": evidence}
    if keep["value"]["passes"] and keep["context"]["passes"]:
        return {"state": "BOTH_ENDPOINTS_PRESERVED", "reason": None, "endpoint_profile": None,
                "evidence": evidence}
    return {"state": "INCONCLUSIVE", "reason": "neither rule established", "endpoint_profile": None,
            "evidence": evidence}


# ------------------------------------------------------------------------------------ fixtures
def _series(mean, sd, n=M):
    """Deterministic antisymmetric-offset series with an EXACT sample mean and sample SD.

    No RNG, so a bound can be placed exactly on a boundary. Offsets are symmetric (`+v` for each
    `-v`), so `fsum` cancels exactly.
    """
    if n % 2:
        raise ValueError("n must be even")
    half = [(k + 1) / (n // 2) for k in range(n // 2)]
    offsets = half + [-v for v in half]
    scale = sd / math.sqrt(math.fsum(o * o for o in offsets) / (n - 1))
    return [mean + scale * o for o in offsets]


def _from_bound(mean, se, n=M):
    return _series(mean, se * math.sqrt(n), n=n)


def _native_at(u95_target, sd=0.02):
    """A Native series whose U95(mean(N)) equals `u95_target` exactly."""
    mean = u95_target - T95 * (sd / SQRT_M)
    return _series(mean, sd)


def _z_at(bound, target, sd=0.02):
    """A Z series whose U95 equals `bound` (target='u95') or whose L97.5 equals it ('l975')."""
    se = sd / SQRT_M
    mean = target - T95 * se if bound == "u95" else target + T975 * se
    return _series(mean, sd)


def analysis_fixtures():
    """X1-X12: every terminal class, both endpoint-specific losses, and all strict boundaries."""
    good_native = {"value": _native_at(0.60), "context": _native_at(0.60)}     # U95(N) = 0.60 < 0.80
    qualified = {"value": _native_at(0.60), "context": _native_at(0.60)}
    fixtures = {}

    def add(name, native, pos, expected, profile=None, note=""):
        fixtures[name] = {"native": native, "pos": pos, "expected_state": expected,
                          "expected_profile": profile, "note": note}

    # Native plainly qualified helpers
    def q_native():
        return {"value": list(qualified["value"]), "context": list(qualified["context"])}

    def with_z(value_z, context_z):
        """Build pos = native + Z - rho*(1-native) ... solved directly: Q = Z + 0.95*N + 0.05."""
        native = q_native()
        pos = {}
        for endpoint, z in (("value", value_z), ("context", context_z)):
            pos[endpoint] = [z[i] + 0.95 * native[endpoint][i] + RHO for i in range(M)]
        return native, pos

    # X1 preservation: U95(Z) < 0 on both endpoints
    native, pos = with_z(_z_at("u95", -0.005), _z_at("u95", -0.005))
    add("X1_both_endpoints_preserved", native, pos, "BOTH_ENDPOINTS_PRESERVED")

    # X2/X3/X4 material loss, per endpoint profile
    native, pos = with_z(_z_at("l975", +0.005), _z_at("u95", -0.005))
    add("X2_material_loss_value_only", native, pos, "MATERIAL_LOSS", profile="value")
    native, pos = with_z(_z_at("u95", -0.005), _z_at("l975", +0.005))
    add("X3_material_loss_context_only", native, pos, "MATERIAL_LOSS", profile="context")
    native, pos = with_z(_z_at("l975", +0.005), _z_at("l975", +0.005))
    add("X4_material_loss_both", native, pos, "MATERIAL_LOSS", profile="both")

    # X5 inconclusive: neither bound established on either endpoint
    native, pos = with_z(_series(0.0, 0.02), _series(0.0, 0.02))
    add("X5_inconclusive", native, pos, "INCONCLUSIVE")

    # X6/X7 Native not qualified, on one endpoint each
    n6 = q_native(); n6["value"] = _native_at(0.85)          # U95(N) = 0.85 > 0.80
    native, pos = with_z(_z_at("l975", +0.005), _z_at("l975", +0.005))
    add("X6_native_not_qualified_value", {"value": n6["value"], "context": native["context"]}, pos,
        "NATIVE_NOT_QUALIFIED")
    n7 = q_native(); n7["context"] = _native_at(0.85)
    add("X7_native_not_qualified_context", {"value": native["value"], "context": n7["context"]},
        pos, "NATIVE_NOT_QUALIFIED")

    # X8 precedence: would be MATERIAL_LOSS but a guard fails -> NATIVE_NOT_QUALIFIED wins
    n8 = q_native(); n8["context"] = _native_at(0.85)
    add("X8_precedence_guard_beats_material_loss",
        {"value": n8["value"], "context": n8["context"]}, pos, "NATIVE_NOT_QUALIFIED",
        note="both endpoints materially lost, but the Native guard fails first")

    # X12 mixed: one endpoint preserves, the other is neither
    native, pos = with_z(_z_at("u95", -0.005), _series(0.0, 0.02))
    add("X12_mixed_endpoints", native, pos, "INCONCLUSIVE",
        note="value preserves, context is neither preserved nor lost")

    return fixtures


def boundary_fixtures():
    """X9/X10/X11: exact-boundary fixtures. Equality must never pass."""
    out = {}
    # X9 preservation boundary: U95(Z) == 0 exactly
    z = _z_at("u95", 0.0)
    out["X9_preservation_boundary_equal"] = {
        "z": z, "expected_preservation": False,
        "note": "U95(Z) == 0 exactly; preservation requires U95 < 0"}
    # X10 material-loss boundary: L97.5(Z) == 0 exactly
    z = _z_at("l975", 0.0)
    out["X10_loss_boundary_equal"] = {
        "z": z, "expected_material_loss": False,
        "note": "L97.5(Z) == 0 exactly; material loss requires L97.5 > 0"}
    # X11a Native boundary via U95(N) == 0.80 exactly
    n = _native_at(NATIVE_MAX_NMSE)
    out["X11a_native_boundary_u95_equal"] = {
        "native": n, "expected_qualification": False,
        "note": "U95(N) == 0.80 exactly; qualification requires U95(N) < 0.80"}
    # X11b Native boundary via L95(G) == 0.20 exactly (the equivalent form)
    g_mean = NATIVE_MIN_RECOVERY + T95 * (0.02 / SQRT_M)
    g = _series(g_mean, 0.02)
    out["X11b_native_boundary_l95_equal"] = {
        "native": [1.0 - value for value in g], "expected_qualification": False,
        "note": "L95(G) == 0.20 exactly; qualification requires L95(G) > 0.20"}
    return out


def run_fixtures():
    """Run every XR1 classifier fixture. Single source of truth for P6."""
    results, failures = {}, []
    for name, fixture in analysis_fixtures().items():
        outcome = classify(native=fixture["native"], pos=fixture["pos"])
        entry = {"state": outcome["state"], "profile": outcome.get("endpoint_profile"),
                 "expected": fixture["expected_state"], "note": fixture["note"]}
        checks = [outcome["state"] == fixture["expected_state"]]
        if fixture.get("expected_profile"):
            checks.append(outcome.get("endpoint_profile") == fixture["expected_profile"])
        entry["passed"] = all(checks)
        results[name] = entry
        if not entry["passed"]:
            failures.append(name)

    boundaries = {}
    b = boundary_fixtures()
    boundaries["X9_preservation_boundary_equal"] = {
        "passes": preservation(b["X9_preservation_boundary_equal"]["z"])["passes"],
        "expected": False}
    boundaries["X10_loss_boundary_equal"] = {
        "passes": material_loss(b["X10_loss_boundary_equal"]["z"])["passes"], "expected": False}
    boundaries["X11a_native_boundary_u95_equal"] = {
        "passes": native_qualification(b["X11a_native_boundary_u95_equal"]["native"])["passes"],
        "expected": False}
    boundaries["X11b_native_boundary_l95_equal"] = {
        "passes": native_qualification(b["X11b_native_boundary_l95_equal"]["native"])["passes"],
        "expected": False}
    for name, entry in boundaries.items():
        entry["passed"] = entry["passes"] is entry["expected"]
        if not entry["passed"]:
            failures.append(name)

    # an incomplete input must raise, never invent a class
    raised = False
    try:
        classify(native={"value": [0.5] * M}, pos={"value": [0.5] * M})
    except ValueError:
        raised = True
    if not raised:
        failures.append("X13_incomplete_input_must_raise")

    return {"fixtures": results, "boundaries": boundaries, "failed": failures,
            "passed": not failures,
            "state_space": list(TERMINAL_STATES),
            "states_covered": sorted({e["state"] for e in results.values()}),
            "profiles_covered": sorted({e["profile"] for e in results.values() if e["profile"]})}
