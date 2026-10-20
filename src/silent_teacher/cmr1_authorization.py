"""CMR1 governed-execution authorization **boundary** (execution-critical).

The previous design validated the *shape* of an in-memory :class:`AuthorizationRecord` and then issued a
capability that the governed entrypoints honoured.  That was a mechanism, not a boundary: any scratch
`python -c` invocation could construct a structurally valid record and self-authorize canonical authority,
which is exactly how the recorded unauthorized root derivation happened.

This module replaces it with a boundary built on an **existing repository trust primitive — git**.  A
governed capability can be issued only from an authorization record whose bytes are read out of the **git
object database** at a named revision, i.e. from a *committed* artifact.  The consequences are:

* ordinary library calls and scratch scripts **cannot mint** an authorizing record, because writing a file
  is not the same as committing it, and this module never reads the worktree;
* the decision is therefore externally recorded, attributable and permanently visible in history;
* **no** `force` flag, debug bypass, exported sentinel, magic boolean, or caller-controlled "approved"
  field exists anywhere in this module.

**Honest limitation (stated, not hidden).** This boundary is *auditability*, not cryptography. A party with
write access to the repository can `git commit` an authorization artifact and thereby authorize execution.
What the boundary guarantees is that doing so is a deliberate, recorded, reviewable act that cannot be
mistaken for an ordinary library call — not that it is impossible.

This module derives **no** seed and creates **no** governed object.
"""
from __future__ import annotations

import json
import subprocess
import weakref
from pathlib import Path

#: Exact scopes; Stage B cannot substitute for numerical or target-access phases.
SCOPE_QUALIFICATION_IDENTITY_REALIZATION = "qualification_identity_realization"
SCOPE_CALIBRATION_EXECUTION = "calibration_execution"
SCOPE_QUALIFICATION_EXECUTION = "qualification_execution"
SCOPE_STAGE_B_REALIZATION = "stage_b_execution_control"
SCOPE_TARGET_MATERIALIZATION = "target_materialization"
SCOPE_TEST_ONLY = "test_only"
KNOWN_SCOPES: tuple[str, ...] = (SCOPE_QUALIFICATION_IDENTITY_REALIZATION,
                                 SCOPE_CALIBRATION_EXECUTION,
                                 SCOPE_QUALIFICATION_EXECUTION, SCOPE_STAGE_B_REALIZATION,
                                 SCOPE_TARGET_MATERIALIZATION,
                                 SCOPE_TEST_ONLY)

#: The governed phase a record must name.  Exactly one phase per record.
KNOWN_PHASES: tuple[str, ...] = ("qualification_identity_realization",
                                  "calibration_execution", "qualification_execution",
                                  SCOPE_STAGE_B_REALIZATION, SCOPE_TARGET_MATERIALIZATION)

#: Canonical location of the authorization artifact (committed, never read from the worktree).
AUTHORIZATION_ARTIFACT_PATH = "manifests/cmr1_governed_execution_authorization.json"
MATERIALIZATION_AUTHORIZATION_ARTIFACT_PATH = (
    "manifests/cmr1_target_materialization_authorization.json")

CAMPAIGN_ID = "cmr1-pythia-cross-model-replication"

#: Fields a committed authorization artifact must carry in the FILE.  None may be absent or empty.
#: artifact_identity is deliberately NOT here: a file cannot contain its own SHA-256, so the
#: loader computes the artifact identity from the committed bytes and binds it into the record and the
#: capability.  If the file *does* declare an artifact_identity, it must equal that computed value
#: (a declared identity is checked, never trusted).
REQUIRED_FIELDS: tuple[str, ...] = (
    "schema",
    "scope",
    "campaign",
    "repository_commit",
    "decision_reference",
    "authorized_phase",
)


class AuthorizationBoundaryError(RuntimeError):
    """An authorization artifact is absent, malformed, out of scope, or not committed."""


# Loader attestation is held outside user-constructible fields.  Values are captured at load time so
# copying or mutating public record attributes cannot promote a scratch/test-only record.
_LOADED_RECORDS: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()
_ISSUED_CAPABILITIES: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


class AuthorizationRecord:
    """A validated authorization record.

    Instances are produced by :func:`load_committed_authorization` only.  Direct construction is possible
    for **TEST-ONLY** records, but a directly constructed record can never authorize a governed scope:
    :func:`issue_capability` refuses to mint a governed capability from one.
    """

    __slots__ = ("scope", "campaign", "repository_commit", "decision_reference",
                 "authorized_phase", "artifact_identity", "__weakref__")

    def __init__(self, *, scope: str, campaign: str, repository_commit: str,
                 decision_reference: str, authorized_phase: str,
                 artifact_identity: str | None = None) -> None:
        if scope not in KNOWN_SCOPES:
            raise AuthorizationBoundaryError(
                f"unknown scope {scope!r}; known scopes are {KNOWN_SCOPES}")
        for name, value in (("campaign", campaign),
                            ("repository_commit", repository_commit),
                            ("decision_reference", decision_reference),
                            ("authorized_phase", authorized_phase)):
            if not isinstance(value, str) or not value.strip():
                raise AuthorizationBoundaryError(f"{name} must be a non-empty string")
        if authorized_phase not in KNOWN_PHASES:
            raise AuthorizationBoundaryError(
                f"unknown authorized_phase {authorized_phase!r}; known phases are {KNOWN_PHASES}")
        self.scope = scope
        self.campaign = campaign
        self.repository_commit = repository_commit
        self.decision_reference = decision_reference
        self.authorized_phase = authorized_phase
        self.artifact_identity = artifact_identity

    @property
    def is_committed(self) -> bool:
        """True only for records read out of the git object database."""
        return self in _LOADED_RECORDS

    def to_dict(self) -> dict:
        return {"schema": "latentbridge.cmr1.governed_execution_authorization.v1",
                "scope": self.scope, "campaign": self.campaign,
                "repository_commit": self.repository_commit,
                "decision_reference": self.decision_reference,
                "authorized_phase": self.authorized_phase,
                "artifact_identity": self.artifact_identity}


class Capability:
    """An opaque capability issued from a **committed** authorization record."""

    __slots__ = ("_scope", "_authorized_phase", "_artifact_identity", "__weakref__")

    def __init__(self, record: AuthorizationRecord) -> None:
        if not isinstance(record, AuthorizationRecord) or record not in _LOADED_RECORDS:
            raise AuthorizationBoundaryError(
                "a capability may only be issued from a loader-attested committed record")
        if _LOADED_RECORDS[record] != _record_fields(record):
            raise AuthorizationBoundaryError("loaded authorization fields changed after verification")
        self._scope = record.scope
        self._authorized_phase = record.authorized_phase
        self._artifact_identity = record.artifact_identity
        _ISSUED_CAPABILITIES[self] = (self._scope, self._authorized_phase,
                                      self._artifact_identity)

    @property
    def scope(self) -> str:
        return self._scope

    @property
    def authorized_phase(self) -> str:
        return self._authorized_phase

    @property
    def artifact_identity(self) -> str | None:
        return self._artifact_identity

    def authorizes(self, scope: str) -> bool:
        return self._scope == scope

    def __repr__(self) -> str:                                              # pragma: no cover
        return f"Capability(scope={self._scope!r}, phase={self._authorized_phase!r})"


def _record_fields(record: AuthorizationRecord) -> tuple:
    return (record.scope, record.campaign, record.repository_commit,
            record.decision_reference, record.authorized_phase, record.artifact_identity)


def _git_show(repo_root: Path, revision: str, path: str) -> bytes:
    """Read ``<revision>:<path>`` from the git object database (never from the worktree)."""
    result = subprocess.run(["git", "show", f"{revision}:{path}"], cwd=str(repo_root),
                            capture_output=True)
    if result.returncode != 0:
        raise AuthorizationBoundaryError(
            f"no committed authorization artifact at {revision}:{path} "
            f"(git show failed with code {result.returncode}); an uncommitted file does not count")
    return result.stdout


def load_committed_authorization(*, repo_root: str | Path = ".",
                                 revision: str = "HEAD",
                                 path: str = AUTHORIZATION_ARTIFACT_PATH
                                 ) -> AuthorizationRecord:
    """Load and validate a **committed** authorization artifact.

    Validation binds, at minimum: scope; repository/commit identity; campaign identity; decision
    provenance; exactly one intended governed phase; and the artifact's own identity.
    """
    root = Path(repo_root)
    raw = _git_show(root, revision, path)
    try:
        document = json.loads(raw.decode("utf-8"))
    except Exception as exc:
        raise AuthorizationBoundaryError(f"authorization artifact is not valid JSON: {exc!r}") from exc
    if not isinstance(document, dict):
        raise AuthorizationBoundaryError("authorization artifact must be a JSON object")

    missing = [field for field in REQUIRED_FIELDS
               if field not in document or document[field] in (None, "")]
    if missing:
        raise AuthorizationBoundaryError(
            f"authorization artifact is missing required field(s): {missing}")
    if document["campaign"] != CAMPAIGN_ID:
        raise AuthorizationBoundaryError(
            f"authorization artifact names campaign {document['campaign']!r}, "
            f"expected {CAMPAIGN_ID!r}")
    if document["authorized_phase"] not in KNOWN_PHASES:
        raise AuthorizationBoundaryError(
            f"unknown authorized_phase {document['authorized_phase']!r}")
    if document["scope"] != SCOPE_TEST_ONLY and document["scope"] != document["authorized_phase"]:
        raise AuthorizationBoundaryError("governed scope and authorized phase must agree exactly")

    import hashlib
    identity = hashlib.sha256(raw).hexdigest()
    declared = document.get("artifact_identity")
    if declared not in (None, identity):
        raise AuthorizationBoundaryError(
            "authorization artifact identity mismatch: the declared artifact_identity does not equal "
            "the digest of the committed bytes")

    if document["scope"] != SCOPE_TEST_ONLY:
        authoritative_root = Path(__file__).resolve().parents[2]
        expected_path = (MATERIALIZATION_AUTHORIZATION_ARTIFACT_PATH
                         if document["scope"] == SCOPE_TARGET_MATERIALIZATION
                         else AUTHORIZATION_ARTIFACT_PATH)
        if root.resolve() != authoritative_root or path != expected_path:
            raise AuthorizationBoundaryError(
                "a governed authorization must be loaded from the canonical committed artifact "
                "in this repository; a disposable TEST-ONLY repo cannot authorize that namespace")
    record = AuthorizationRecord(scope=document["scope"], campaign=document["campaign"],
                                 repository_commit=document["repository_commit"],
                                 decision_reference=document["decision_reference"],
                                 authorized_phase=document["authorized_phase"],
                                 artifact_identity=identity)
    _LOADED_RECORDS[record] = _record_fields(record)
    return record


def test_only_record(*, scope: str = SCOPE_TEST_ONLY,
                     authorized_phase: str = "qualification_identity_realization") -> AuthorizationRecord:
    """A clearly non-governed record for unit tests.  Can never authorize a governed scope."""
    return AuthorizationRecord(scope=scope, campaign="TEST-ONLY", repository_commit="TEST-ONLY",
                              decision_reference="TEST-ONLY", authorized_phase=authorized_phase,
                              artifact_identity=None)


def issue_capability(record: AuthorizationRecord) -> Capability:
    """Issue a capability — **only** from a committed record.

    A directly constructed in-memory record (including a TEST-ONLY one) is refused here, which is the
    property that makes ordinary scratch callers unable to self-authorize governed realization.
    """
    if not isinstance(record, AuthorizationRecord):
        raise AuthorizationBoundaryError("issue_capability requires an AuthorizationRecord")
    if not record.is_committed:
        raise AuthorizationBoundaryError(
            "refusing to issue a capability from a non-committed record: an in-memory record cannot "
            "establish authorization. A governed capability requires an artifact read from the git "
            "object database.")
    return Capability(record)


def capability_authorizes(capability: object | None, scope: str) -> bool:
    """True only for a Capability whose scope matches exactly."""
    return (isinstance(capability, Capability)
            and _ISSUED_CAPABILITIES.get(capability) ==
            (scope, capability.authorized_phase, capability.artifact_identity)
            and capability.authorizes(scope))


def authorization_manifest() -> dict:
    return {
        "artifact_path": AUTHORIZATION_ARTIFACT_PATH,
        "target_materialization_artifact_path": MATERIALIZATION_AUTHORIZATION_ARTIFACT_PATH,
        "read_from": "git object database at a named revision (never the worktree)",
        "required_fields": list(REQUIRED_FIELDS),
        "known_scopes": list(KNOWN_SCOPES),
        "known_phases": list(KNOWN_PHASES),
        "campaign_id": CAMPAIGN_ID,
        "boundary_kind": "auditability, not cryptography",
        "self_authorization_possible": "no — writing a file is not committing it, and this module "
                                       "never reads the worktree",
        "residual_limitation": "a party with repository write access can commit an authorization; the "
                               "guarantee is that this is a deliberate, recorded, reviewable act",
        "derives_governed_seed": False,
        "in_memory_record_authorizes_governed_scope": False,
    }
