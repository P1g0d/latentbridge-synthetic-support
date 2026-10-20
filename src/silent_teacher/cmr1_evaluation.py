"""CMR1 target endpoints and paired analysis using the executed H9 ID2 primitives.

This module does not load targets or checkpoints. The formal driver supplies the
frozen inputs; the analysis consumes only persisted per-Student endpoint records.
"""
from __future__ import annotations

import math

from . import h9_id2_analysis as id2

REPLICATES = 48
ARMS = ("nat", "pos")
ENVIRONMENTS = ("c4", "pes2o", "stackexchange")
ID2_ENVIRONMENT_NAMES = {"c4": "c4", "pes2o": "pes2o", "stackexchange": "se"}
NATIVE_MIN_RECOVERY = 0.20


class EvaluationNotEstimable(RuntimeError):
    """A frozen primary endpoint cannot be computed; never impute or drop it."""


def _finite(value: object) -> bool:
    return type(value) in (float, int) and math.isfinite(value)


def aggregate_endpoints(per_environment: dict) -> dict:
    """Use ID2's frozen equal-environment and equal-query aggregators unchanged."""
    if set(per_environment) != set(ENVIRONMENTS):
        raise EvaluationNotEstimable("the frozen three-environment set is incomplete")
    values, contexts = {}, {}
    for cmr1_name, id2_name in ID2_ENVIRONMENT_NAMES.items():
        entry = per_environment[cmr1_name]
        value = entry["value_nmse"]
        context = entry["context"]
        if not _finite(value) or any(not _finite(context.get(key)) for key in ("t63", "t127")):
            raise EvaluationNotEstimable(f"{cmr1_name}: nonfinite or undefined primary endpoint")
        if (not _finite(context.get("teacher_intervention_energy_ratio")) or
                context["teacher_intervention_energy_ratio"] < 1e-4 or
                any(not _finite(context.get("teacher_response_denominator", {}).get(key)) or
                    context["teacher_response_denominator"][key] <= 0
                    for key in ("t63", "t127"))):
            raise EvaluationNotEstimable(f"{cmr1_name}: frozen Context denominator or energy guard")
        values[id2_name] = value
        contexts[id2_name] = {"t63": context["t63"], "t127": context["t127"]}
    return {"R_P": id2.value_endpoint(values), "C_P": id2.context_endpoint(contexts)}


def analyse_persisted(students: dict) -> dict:
    """Apply CMR1's frozen precedence using ID2's executed 48-pair t mathematics."""
    if set(students) != {str(i) for i in range(1, REPLICATES + 1)}:
        raise EvaluationNotEstimable("not all 48 paired replicates are persisted")
    series = {endpoint: {arm: [] for arm in ARMS} for endpoint in ("value", "context")}
    environments = {environment: {endpoint: {arm: [] for arm in ARMS}
                                  for endpoint in ("value", "context")}
                    for environment in ENVIRONMENTS}
    for i in range(1, REPLICATES + 1):
        pair = students[str(i)]
        if set(pair["arms"]) != set(ARMS):
            raise EvaluationNotEstimable(f"replicate {i}: paired arms incomplete")
        for arm in ARMS:
            record = pair["arms"][arm]
            if record.get("state") != "COMPLETE" or record.get("terminal_step") != 10000:
                raise EvaluationNotEstimable(f"replicate {i}/{arm}: endpoint not complete")
            metrics = record["metrics"]
            if any(not _finite(metrics.get(key)) for key in ("R_P", "C_P")):
                raise EvaluationNotEstimable(f"replicate {i}/{arm}: endpoint not finite")
            aggregate = aggregate_endpoints(metrics["environments"])
            if aggregate != {"R_P": metrics["R_P"], "C_P": metrics["C_P"]}:
                raise EvaluationNotEstimable(f"replicate {i}/{arm}: aggregate mismatch")
            series["value"][arm].append(metrics["R_P"])
            series["context"][arm].append(metrics["C_P"])
            for environment in ENVIRONMENTS:
                component = metrics["environments"][environment]
                environments[environment]["value"][arm].append(component["value_nmse"])
                environments[environment]["context"][arm].append(
                    (component["context"]["t63"] + component["context"]["t127"]) / 2.0)

    endpoints = {}
    for endpoint, margin in (("value", id2.DELTA_VALUE),
                             ("context", id2.DELTA_CONTEXT)):
        native = series[endpoint]["nat"]
        pos = series[endpoint]["pos"]
        difference = id2.paired_differences(pos, native)
        recovery = [1.0 - value for value in native]
        native_guard = id2.statistics_of(recovery)
        loss = id2.material_loss(difference, margin)
        preservation = id2.noninferiority(difference, margin)
        endpoints[endpoint] = {
            "native_recovery": {**native_guard,
                                "passes": native_guard["L95"] > NATIVE_MIN_RECOVERY,
                                "threshold": NATIVE_MIN_RECOVERY},
            "paired_difference": {**id2.statistics_of(difference),
                                  "positive_count": sum(value > 0 for value in difference)},
            "material_loss": loss, "preservation": preservation,
            "margin": margin,
            "native_mean": id2.statistics_of(native)["mean"],
            "q_pos_mean": id2.statistics_of(pos)["mean"],
            "theta": (id2.statistics_of(difference)["mean"] /
                      native_guard["mean"] if native_guard["mean"] != 0 else None),
        }

    guard = {name: entry["native_recovery"]["passes"] for name, entry in endpoints.items()}
    loss = {name: entry["material_loss"]["passes"] for name, entry in endpoints.items()}
    keep = {name: entry["preservation"]["passes"] for name, entry in endpoints.items()}
    if not all(guard.values()):
        terminal = "NATIVE_NOT_QUALIFIED"
    elif all(loss.values()):
        terminal = "FULL_REPLICATION"
    elif sum(loss.values()) == 1:
        terminal = "PARTIAL_REPLICATION"
    elif all(keep.values()):
        terminal = "QUALIFIED_PRESERVATION"
    else:
        terminal = "INCONCLUSIVE"
    other_endpoint = next((name for name in endpoints if not loss[name]), None)
    descriptives = {}
    for environment, components in environments.items():
        descriptives[environment] = {}
        for endpoint, arms in components.items():
            differences = id2.paired_differences(arms["pos"], arms["nat"])
            descriptives[environment][endpoint] = {
                "native": id2.statistics_of(arms["nat"]),
                "q_pos": id2.statistics_of(arms["pos"]),
                "paired_difference": {**id2.statistics_of(differences),
                                      "positive_count": sum(value > 0 for value in differences)}}
    return {"schema": "latentbridge.cmr1.paired_analysis.v1", "replicates": REPLICATES,
            "endpoints": endpoints, "terminal_class": terminal,
            "environment_descriptives": descriptives,
            "partial_other_endpoint": (None if terminal != "PARTIAL_REPLICATION" else
                                       {"name": other_endpoint,
                                        "status": "NON_INFERIOR" if keep[other_endpoint] else "UNRESOLVED"}),
            "rules": {"native_qualification": guard, "material_loss": loss,
                      "preservation": keep},
            "historical_primitives": ["h9_id2_analysis.value_endpoint",
                                      "h9_id2_analysis.context_endpoint",
                                      "h9_id2_analysis.paired_differences",
                                      "h9_id2_analysis.statistics_of",
                                      "h9_id2_analysis.material_loss",
                                      "h9_id2_analysis.noninferiority"]}
