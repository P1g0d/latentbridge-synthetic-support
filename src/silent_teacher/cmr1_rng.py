"""CMR1 B9 RNG-derivation semantics, qualification role registry and guards.

Encodes the authoritative contract in ``manifests/cmr1_rng_role_schema.json`` (B9) as amended by
the approved qualification-RNG amendment v0.4.  This module makes **no** scientific choice: every
value and rule below is transcribed from the frozen schema.

Two guards are load-bearing and deliberately closed in this phase:

``CANONICAL_QUALIFICATION_NAMESPACE``
    The concrete canonical qualification root namespace.  Deriving a root seed or a role seed from
    it is a **governed identity realization**, which is not authorized during implementation
    preflight.  :func:`qualification_root_seed` and :func:`qualification_role_seed` therefore raise
    :class:`GovernedQualificationNotAuthorized` unless an explicit authorization object is supplied.
    Capabilities come only from the committed authorization boundary; scratch records do not
    authorize governed realization.

``CAMPAIGN``
    Root and child derivation require an attested Stage-B context bound to ONE committed
    declaration and reservation. The implementation alone authorizes no campaign realization.
"""
from __future__ import annotations

import hashlib
from typing import Iterable, Mapping, Sequence

# --------------------------------------------------------------------------- frozen constants
MODULUS = 2**63 - 1
SCHEMA_ID = "latentbridge.cmr1.rng_role_schema.v1"
SERIALIZATION_RULE_ID = "B9-DERIVE-SER-1"
QUALIFICATION_SCHEMA_VERSION = "latentbridge/cmr1/qualification/v1"
STAGE_A_FREEZE_COMMIT = "c6791ee2f70e01a0ac61de22d6ba93075413c42f"
CANONICAL_QUALIFICATION_NAMESPACE = (
    f"latentbridge/cmr1/qualification/v1/stage-a-{STAGE_A_FREEZE_COMMIT}"
)
QUALIFICATION_TAG_RULE = "'stage-a-' || <full 40-hex-character Stage-A freeze commit id>"

#: The nine campaign roles (B9 ``roles``).  Unchanged by the amendment.  Derived at Stage B only.
CAMPAIGN_ROLES: tuple[str, ...] = (
    "donor_train", "donor_dev", "donor_audit",
    "permutation_train", "permutation_dev", "permutation_audit",
    "student_init", "minibatch_schedule", "context_donor",
)

#: The six qualification roles (B9 ``qualification_domain.role_registry``), in canonical order.
QUALIFICATION_ROLES: tuple[str, ...] = (
    "calibration/q2-1",
    "calibration/q2-4",
    "fixture/formal",
    "permutation/native-qpos",
    "init/q3",
    "schedule/q3",
)

#: B9 ``cross_domain_collision.domain_labels``.
DOMAINS: tuple[str, ...] = ("campaign", "qualification", "target_selection")

#: Marker that a namespace is explicitly non-governed.  Test namespaces must contain it.
TEST_ONLY_MARKER = "TEST-ONLY"


class RngSchemaError(RuntimeError):
    """A rule of the frozen B9 schema was violated."""


class GovernedQualificationNotAuthorized(RuntimeError):
    """Raised when a governed operation is attempted without a valid scoped capability.

    Governed identity realization is a separately-authorized act (B11 pre-exposure ordering, step 3).
    During implementation preflight no capability for the governed scope exists, so this fires.
    """


class CanonicalNamespaceGuardError(RuntimeError):
    """Raised when a TEST-ONLY code path is handed the canonical governed namespace."""


# The authorization BOUNDARY lives in :mod:`silent_teacher.cmr1_authorization`.  Its scopes, record type
# and capability type are re-exported here so existing callers keep working, but the ability to mint a
# governed capability requires a *committed* authorization artifact -- an in-memory record cannot
# authorize a governed scope (this is the repair of the boundary; see the incident record
# reports/cmr1_i4a_unauthorized_root_derivation_incident.md).
from silent_teacher.cmr1_authorization import (                                    # noqa: E402
    AUTHORIZATION_ARTIFACT_PATH,
    CAMPAIGN_ID,
    Capability,
    SCOPE_QUALIFICATION_IDENTITY_REALIZATION,
    SCOPE_STAGE_B_REALIZATION,
    SCOPE_TEST_ONLY,
    AuthorizationBoundaryError,
    AuthorizationRecord,
    KNOWN_PHASES,
    KNOWN_SCOPES,
    authorization_manifest,
    capability_authorizes,
    issue_capability,
    load_committed_authorization,
    test_only_record,
)

#: Backwards-compatible alias: the earlier draft called this an "error" on the record validator.
AuthorizationRecordError = AuthorizationBoundaryError


# --------------------------------------------------------------------------- seed rendering
def render_seed(seed: str | int) -> str:
    """Render a ``derive`` seed argument per ``B9-DERIVE-SER-1``.

    * ``str`` -- used **verbatim**: no quoting, trimming, Unicode normalization, path or case
      conversion.
    * ``int`` -- canonical unsigned base-10 ASCII decimal: digits ``0``-``9`` only, no leading
      ``+``, no leading zeros (zero renders as ``"0"``), no spaces, no separators, no scientific
      notation, no locale-dependent formatting.  **No zero padding** -- B9's two-digit
      ``zero_padding`` rule governs the replicate-id *name component*, not seed rendering.
    """
    if isinstance(seed, bool):                     # bool is an int subclass; reject explicitly
        raise RngSchemaError("bool is not a valid derive seed type")
    if isinstance(seed, str):
        return seed
    if isinstance(seed, int):
        if seed < 0:
            raise RngSchemaError("integer seeds must be non-negative (unsigned rendering)")
        return str(seed)                            # canonical decimal, locale-independent
    raise RngSchemaError(f"unsupported derive seed type: {type(seed).__name__}")


def derive(seed: str | int, name: str) -> int:
    """The shared frozen B9 primitive, unchanged.

    ``int.from_bytes(sha256(('<seed>:<name>').encode('utf-8')).digest()[:8], 'little') % (2**63-1)``
    """
    if not isinstance(name, str):
        raise RngSchemaError("derive name must be a string")
    payload = f"{render_seed(seed)}:{name}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % MODULUS


def _require_governed_authorization(namespace: str, capability: object | None) -> None:
    """Fail closed unless a **boundary-issued** capability for the governed scope is supplied.

    Both conditions must hold: the namespace must be the canonical one (a governed act is being
    requested), **and** the caller must present a capability issued from a committed authorization
    artifact carrying ``qualification_identity_realization``.  An in-memory record cannot produce such
    a capability, so ordinary library callers and scratch scripts cannot self-authorize.
    """
    if not isinstance(namespace, str) or not is_canonical_qualification_namespace(namespace):
        return                                    # non-governed namespaces never need a capability
    if not capability_authorizes(capability, SCOPE_QUALIFICATION_IDENTITY_REALIZATION):
        raise GovernedQualificationNotAuthorized(
            "deriving identities from the canonical qualification namespace is a governed identity "
            "realization and requires a capability issued from a committed authorization artifact for "
            f"scope {SCOPE_QUALIFICATION_IDENTITY_REALIZATION!r} (B11 step 3). No such capability "
            "exists in this phase. Use a TEST-ONLY namespace.")


def is_canonical_qualification_namespace(namespace: str) -> bool:
    return namespace == CANONICAL_QUALIFICATION_NAMESPACE


def require_test_only_namespace(namespace: str) -> str:
    """Assert a namespace is explicitly non-governed, so a test cannot mistake it for the real one."""
    if not isinstance(namespace, str):
        raise CanonicalNamespaceGuardError("an explicit TEST-ONLY namespace string is required")
    if is_canonical_qualification_namespace(namespace):
        raise CanonicalNamespaceGuardError(
            "the canonical governed qualification namespace may never be used on a TEST-ONLY path")
    if TEST_ONLY_MARKER not in namespace:
        raise CanonicalNamespaceGuardError(
            f"non-governed namespaces must contain {TEST_ONLY_MARKER!r}; got {namespace!r}")
    return namespace


def test_only_namespace(tag: str) -> str:
    """Build a clearly non-governed qualification-style namespace for tests."""
    if not tag or "/" in tag:
        raise CanonicalNamespaceGuardError("test namespace tag must be a non-empty single segment")
    return f"latentbridge/cmr1/qualification/{TEST_ONLY_MARKER}/{tag}"


# --------------------------------------------------------------------------- role validation
def validate_role_key(role: str, *, registry: Sequence[str] = QUALIFICATION_ROLES) -> str:
    """Return ``role`` if it is a canonical registry key, else fail closed."""
    if not isinstance(role, str) or role not in registry:
        raise RngSchemaError(
            f"HARD STOP - UNDECLARED QUALIFICATION ROLE: {role!r} is not in the canonical registry")
    return role


def qualification_role_name(namespace: str, role: str) -> str:
    """The step-Q2 ``name`` argument: the full path ``namespace + '/' + role``."""
    validate_role_key(role)
    if namespace.endswith("/"):
        raise RngSchemaError("namespace must not end with '/'")
    return f"{namespace}/{role}"


# --------------------------------------------------------------------------- derivation sequence
def qualification_root_seed(namespace: str,
                            capability: object | None = None) -> int:
    """Step **Q-1** (mandatory): ``derive(QUALIFICATION_ROOT_NAMESPACE, 'root')``."""
    _require_governed_authorization(namespace, capability)
    return derive(namespace, "root")


def qualification_role_seed(namespace: str, role: str,
                            capability: object | None = None) -> int:
    """Step **Q-2**: ``derive(QUALIFICATION_ROOT_SEED, namespace + '/' + role)``.

    The seed argument is always the **integer** root seed; the name argument is always the **full
    path** beginning with the namespace (amendment clauses (a) and (b)).
    """
    _require_governed_authorization(namespace, capability)
    root_seed = qualification_root_seed(namespace, capability)
    return derive(root_seed, qualification_role_name(namespace, role))


def qualification_role_seeds(namespace: str,
                             capability: object | None = None) -> dict[str, int]:
    """All six role seeds, keyed by canonical role string."""
    _require_governed_authorization(namespace, capability)
    root_seed = qualification_root_seed(namespace, capability)
    return {role: derive(root_seed, qualification_role_name(namespace, role))
            for role in QUALIFICATION_ROLES}


def validate_campaign_namespace(namespace: str) -> str:
    """Explicit B9 namespace only; no default, normalization or implicit root choice."""
    import re
    if not isinstance(namespace, str) or not re.fullmatch(
            r"latentbridge/cmr1/v[0-9]+/[^/\s:]+", namespace):
        raise RngSchemaError("HARD STOP - malformed explicit campaign namespace")
    return namespace


def _campaign_entry(namespace: str, context) -> None:
    validate_campaign_namespace(namespace)
    from silent_teacher.cmr1_stage_b import assert_stage_b_context
    assert_stage_b_context(context, namespace=namespace)


def campaign_root_seed(root_namespace: str, *, context=None) -> int:
    _campaign_entry(root_namespace, context)
    return derive(root_namespace, "root")


def campaign_child_seed(root_seed: int, root_namespace: str, replicate: int, role: str,
                        *, context=None) -> int:
    """Frozen B9 step_2; attested Stage-B context only, including explicit TEST-ONLY."""
    _campaign_entry(root_namespace, context)
    if type(replicate) is not int or not 1 <= replicate <= 48:
        raise RngSchemaError("HARD STOP - campaign replicate must be 1..48")
    validate_role_key(role, registry=CAMPAIGN_ROLES)
    if type(root_seed) is not int or root_seed != derive(root_namespace, "root"):
        raise RngSchemaError("HARD STOP - replacement or mismatched campaign root")
    return derive(root_seed, f"{root_namespace}/replicate-{replicate:02d}/{role}")


# --------------------------------------------------------------------------- collision semantics
class RoleRecord:
    """A consumed stochastic identity as a role-keyed record ``(domain, role_key, integer_seed)``."""

    __slots__ = ("domain", "role_key", "integer_seed")

    def __init__(self, domain: str, role_key: str, integer_seed: int) -> None:
        if domain not in DOMAINS:
            raise RngSchemaError(f"unknown collision domain {domain!r}; expected one of {DOMAINS}")
        if not isinstance(integer_seed, int) or isinstance(integer_seed, bool):
            raise RngSchemaError("integer_seed must be an int")
        if not 0 <= integer_seed < MODULUS:
            raise RngSchemaError("integer_seed must lie in [0, 2**63 - 1)")
        self.domain = domain
        self.role_key = role_key
        self.integer_seed = integer_seed

    def __repr__(self) -> str:                                              # pragma: no cover
        return f"RoleRecord({self.domain!r}, {self.role_key!r}, <seed>)"

    def __eq__(self, other: object) -> bool:
        return (isinstance(other, RoleRecord) and self.domain == other.domain
                and self.role_key == other.role_key
                and self.integer_seed == other.integer_seed)

    def __hash__(self) -> int:
        return hash((self.domain, self.role_key, self.integer_seed))


def validate_within_domain(records: Iterable[RoleRecord], domain: str, *,
                           declared_reuse_by_reference: Iterable[tuple[str, str]] = ()) -> dict:
    """B9 **Step A**: within-domain role-record validation.

    * every role key appears **exactly once**;
    * no two **distinct** role keys share an integer seed, unless the authoritative contract
      declares reuse-by-reference for that pair.

    For ``campaign`` this preserves B9's existing 432-child rule (the caller supplies the records).
    Returns a summary; raises :class:`RngSchemaError` on any violation.
    """
    records = list(records)
    if any(r.domain != domain for r in records):
        raise RngSchemaError("all records passed to validate_within_domain must share the domain")

    keys = [r.role_key for r in records]
    duplicates = sorted({k for k in keys if keys.count(k) > 1})
    if duplicates:
        raise RngSchemaError(f"HARD STOP - DUPLICATE QUALIFICATION ROLE: {duplicates}")

    declared = {frozenset(pair) for pair in declared_reuse_by_reference}
    by_seed: dict[int, list[str]] = {}
    for record in records:
        by_seed.setdefault(record.integer_seed, []).append(record.role_key)
    collisions = []
    for seed, owners in by_seed.items():
        if len(owners) > 1 and frozenset(owners) not in declared:
            collisions.append(sorted(owners))
    if collisions:
        raise RngSchemaError(
            f"HARD STOP - QUALIFICATION SEED COLLISION within {domain}: {collisions}")

    return {
        "domain": domain,
        "record_count": len(records),
        "distinct_role_keys": len(set(keys)),
        "distinct_seeds": len(by_seed),
        "declared_reuse_by_reference": sorted(sorted(p) for p in declared),
        "within_domain": "PASS",
    }


def validate_qualification_records(records: Iterable[RoleRecord]) -> dict:
    """Formal qualification-domain validation: EXACTLY the six canonical roles.

    Requires, fail-closed:

    * all six canonical role keys present;
    * **no extra role**;
    * each role appearing **exactly once**;
    * all six consumed integer seeds **pairwise distinct**.

    There is **no** qualification reuse-by-reference exception, and this function takes **no** reuse
    parameter: a caller cannot waive qualification pairwise distinctness, because there is nothing to
    pass.  (The approved campaign/target-selection reuse semantics are unaffected and remain available
    through :func:`validate_within_domain`.)
    """
    records = list(records)
    if any(r.domain != "qualification" for r in records):
        raise RngSchemaError("qualification validation requires qualification-domain records")

    keys = [r.role_key for r in records]
    expected = set(QUALIFICATION_ROLES)
    present = set(keys)

    missing = sorted(expected - present)
    extra = sorted(present - expected)
    if missing:
        raise RngSchemaError(f"HARD STOP - UNDECLARED QUALIFICATION ROLE: missing {missing}")
    if extra:
        raise RngSchemaError(f"HARD STOP - UNDECLARED QUALIFICATION ROLE: unexpected {extra}")

    duplicates = sorted({key for key in keys if keys.count(key) > 1})
    if duplicates:
        raise RngSchemaError(f"HARD STOP - DUPLICATE QUALIFICATION ROLE: {duplicates}")

    if len(records) != len(QUALIFICATION_ROLES):
        raise RngSchemaError(
            f"HARD STOP - DUPLICATE QUALIFICATION ROLE: expected exactly "
            f"{len(QUALIFICATION_ROLES)} records, got {len(records)}")

    seeds: dict[int, list[str]] = {}
    for record in records:
        seeds.setdefault(record.integer_seed, []).append(record.role_key)
    collisions = sorted(owners for owners in seeds.values() if len(owners) > 1)
    if collisions:
        raise RngSchemaError(
            f"HARD STOP - QUALIFICATION SEED COLLISION: distinct qualification roles share a seed: "
            f"{collisions}")

    return {"domain": "qualification", "record_count": len(records),
            "role_count": len(present), "distinct_seeds": len(seeds),
            "reuse_by_reference_exception": "NONE",
            "exact_canonical_set": True, "within_domain": "PASS"}


def validate_cross_domain(set_c: Iterable[int], set_q: Iterable[int],
                          set_t: Iterable[int] | None = None) -> dict:
    """B9 **Step B**: cross-domain intersection checks over consumed integer seeds.

    Requires ``intersection(SET_C, SET_Q) == empty`` and, when T exists, the C/T and Q/T forms.
    Root/intermediate values are excluded: they are derivation intermediates and are not consumed
    stochastic identities.  Category sizes are reported, never hard-coded.
    """
    c, q = set(set_c), set(set_q)
    result = {
        "SET_C_size": len(c),
        "SET_Q_size": len(q),
        "intersection(SET_C, SET_Q)": "empty" if not (c & q) else sorted(c & q),
        "total_expression": "|SET_C| + |SET_Q|" + (" + |SET_T|" if set_t is not None else ""),
    }
    if c & q:
        raise RngSchemaError(
            f"HARD STOP - QUALIFICATION SEED COLLISION: intersection(SET_C, SET_Q) = {sorted(c & q)}")
    if set_t is not None:
        t = set(set_t)
        result["SET_T_size"] = len(t)
        result["intersection(SET_C, SET_T)"] = "empty" if not (c & t) else sorted(c & t)
        result["intersection(SET_Q, SET_T)"] = "empty" if not (q & t) else sorted(q & t)
        if c & t:
            raise RngSchemaError(
                f"HARD STOP - QUALIFICATION CAMPAIGN NAMESPACE COLLISION: "
                f"intersection(SET_C, SET_T) = {sorted(c & t)}")
        if q & t:
            raise RngSchemaError(
                f"HARD STOP - QUALIFICATION SEED COLLISION: "
                f"intersection(SET_Q, SET_T) = {sorted(q & t)}")
    result["cross_domain"] = "PASS"
    return result


def qualification_records(namespace: str,
                          capability: object | None = None) -> list[RoleRecord]:
    """The six qualification role records.  Governed namespaces require a boundary capability."""
    seeds = qualification_role_seeds(namespace, capability)
    return [RoleRecord("qualification", role, seeds[role]) for role in QUALIFICATION_ROLES]


def qualification_registry_summary() -> Mapping[str, object]:
    """Static description of the registry, for identity records and documentation."""
    return {
        "schema_id": SCHEMA_ID,
        "serialization_rule_id": SERIALIZATION_RULE_ID,
        "qualification_schema_version": QUALIFICATION_SCHEMA_VERSION,
        "canonical_root_namespace": CANONICAL_QUALIFICATION_NAMESPACE,
        "tag_rule": QUALIFICATION_TAG_RULE,
        "domains": list(DOMAINS),
        "qualification_roles": list(QUALIFICATION_ROLES),
        "role_count": len(QUALIFICATION_ROLES),
        "campaign_roles": list(CAMPAIGN_ROLES),
        "campaign_role_count": len(CAMPAIGN_ROLES),
        "derivation": {
            "step_Q1": "QUALIFICATION_ROOT_SEED = derive(QUALIFICATION_ROOT_NAMESPACE, 'root')",
            "step_Q2": ("qual_seed(r) = derive(QUALIFICATION_ROOT_SEED, "
                        "QUALIFICATION_ROOT_NAMESPACE + '/' + r)"),
            "levels": "exactly TWO - role seeds derive DIRECTLY from the root seed",
            "root_step_mandatory": True,
        },
        "roots_excluded_from_collision": (
            "root/intermediate values are derivation intermediates and are not consumed stochastic "
            "identities; the exclusion is a prospective amendment clarification"),
        "governed_realization_authorized": False,
    }
