"""H9*-ID2 treatment construction, nine-role RNG interface and the fixed training contract.

Implements exactly the accepted implementation design v0.2:

  - finite position-wise empirical-marginal-preserving permutation intervention;
  - three independent donor pools per replicate (train 4096, own-Q dev 256, audit 512);
  - Native arm  H_nat[n, t] = H[n, t]
  - Q_pos arm   H_pos[n, t] = H[pi_t(n), t]     with pi_t independent plain permutations;
  - labels recomputed from the FINAL constructed tensors, never spliced;
  - a fixed 10,000-step terminal training endpoint with no extension and no selection.

No derangement, no fixed-point removal, no rejection and no resampling exist anywhere in the
treatment. `derangement()` is used only for the evaluation-only `context_donor` object (see
`context_donor_derangement`), which never touches the treatment.

This module reads no target content, creates no seed values for the formal campaign, and trains
nothing by itself.
"""
import errno
import hashlib
from pathlib import Path

import numpy as np
import torch

from .full_block_trainer import learning_rate
from .h9_id1 import prefix_states, seed_cost, seeded_student, teacher_update
from .probes import generator, stream_seed

LENGTH = 128
VOCAB = 50257
REPLICATES = 48                 # frozen campaign size (m = 48 paired replicates)
STUDENTS_PER_REPLICATE = 2
PLANNED_STUDENTS = REPLICATES * STUDENTS_PER_REPLICATE
SPLITS = ("train", "dev", "audit")
POOL_SIZES = {"train": 4096, "dev": 256, "audit": 512}

#: the nine frozen RNG roles. No tenth role exists; adding one requires explicit adjudication.
ROLES = (
    "donor_train",
    "donor_dev",
    "donor_audit",
    "permutation_train",
    "permutation_dev",
    "permutation_audit",
    "student_init",
    "minibatch_schedule",
    "context_donor",
)
DONOR_ROLE = {split: f"donor_{split}" for split in SPLITS}
PERMUTATION_ROLE = {split: f"permutation_{split}" for split in SPLITS}
#: roles whose streams must never influence the treatment
EVALUATION_ONLY_ROLES = ("context_donor",)

#: the frozen fixed training contract (protocol v0.4 sections 3 and 4; design v0.2 section 4.1)
TRAINING_CONTRACT = {
    "steps": 10000,
    "horizon": 10000,
    "microbatch": 8,
    "accumulation": 4,
    "effective_batch": 32,
    "optimizer": "AdamW",
    "betas": (0.9, 0.999),
    "eps": 1e-8,
    "weight_decay": 0.0,
    "lr_peak": 1e-3,
    "lr_final": 1e-4,
    "lr_final_ratio": 0.1,
    "warmup": 200,
    "clip_norm": 1.0,
    "eval_every": 500,
    "precision": "fp32",
    "amp": False,
    "tf32": False,
    "deterministic_algorithms": True,
    "cublas_workspace_config": ":4096:8",
    "terminal_checkpoint_step": 10000,
    "checkpoint_selection": "none",
    "early_stopping": "none",
    "extension": "none",
}
#: tokens that must not appear in any ID2 training code path
FORBIDDEN_TRAINING_TOKENS = ("extension", "early_stop", "earlystopping", "20_000", "20000",
                             "200,000", "best_checkpoint", "fixed_lr_extension", "resume_to",
                             "second_phase", "phase_b")

#: fixture-only namespace. The formal campaign uses a different root namespace; no value derived
#: here may be used as a formal seed.
FIXTURE_NAMESPACE = "latentbridge/h9-id2/preflight-fixture/v1"


def training_config():
    """The `config` dict consumed by the inherited paired trainer, built from the frozen contract."""
    return {"microbatch": TRAINING_CONTRACT["microbatch"],
            "accumulation": TRAINING_CONTRACT["accumulation"],
            "warmup": TRAINING_CONTRACT["warmup"],
            "lr_peak": TRAINING_CONTRACT["lr_peak"],
            "lr_final_ratio": TRAINING_CONTRACT["lr_final_ratio"],
            "clip_norm": TRAINING_CONTRACT["clip_norm"],
            "eval_every": TRAINING_CONTRACT["eval_every"],
            "weight_decay": TRAINING_CONTRACT["weight_decay"],
            "betas": list(TRAINING_CONTRACT["betas"]),
            "eps": TRAINING_CONTRACT["eps"],
            "steps": TRAINING_CONTRACT["steps"]}


def assert_training_contract(config=None, steps=None, horizon=None):
    """Raise unless the live training configuration equals the frozen contract exactly."""
    expected = TRAINING_CONTRACT
    if config is not None:
        live = {"microbatch": config["microbatch"], "accumulation": config["accumulation"],
                "warmup": config["warmup"], "lr_peak": config["lr_peak"],
                "lr_final_ratio": config["lr_final_ratio"], "clip_norm": config["clip_norm"],
                "eval_every": config["eval_every"], "weight_decay": config["weight_decay"],
                "betas": tuple(config["betas"]), "eps": config["eps"], "steps": config["steps"]}
        for key, value in live.items():
            assert value == expected[key], f"training contract mismatch: {key} {value!r} != {expected[key]!r}"
        assert float(expected["lr_peak"]) * float(expected["lr_final_ratio"]) == expected["lr_final"]
    if steps is not None:
        assert int(steps) == expected["steps"], f"steps {steps} != {expected['steps']}"
    if horizon is not None:
        assert int(horizon) == expected["horizon"], f"horizon {horizon} != {expected['horizon']}"
    return True


def terminal_lr():
    """The LR the inherited schedule returns exactly at the terminal step (must be the final LR)."""
    return learning_rate(TRAINING_CONTRACT["steps"], TRAINING_CONTRACT["lr_peak"],
                         TRAINING_CONTRACT["lr_final_ratio"], TRAINING_CONTRACT["warmup"],
                         TRAINING_CONTRACT["horizon"])


class Id2Streams:
    """Nine-role RNG interface. Seeds are READ from the supplied mapping, never re-derived.

    The mapping must contain exactly the nine role names. For the formal campaign these values come
    from the future formal RNG manifest; for preflight they are fixture values derived under
    `FIXTURE_NAMESPACE`.
    """

    def __init__(self, seeds, label=""):
        missing = [role for role in ROLES if role not in seeds]
        extra = [name for name in seeds if name not in ROLES]
        assert not missing, f"missing RNG roles: {missing}"
        assert not extra, f"undeclared RNG roles (a tenth role requires explicit adjudication): {extra}"
        self.label = label
        self.seeds = {role: int(seeds[role]) for role in ROLES}
        self._generators = {role: generator(self.seeds[role], role) for role in ROLES}
        self._consumed = set()

    # ------------------------------------------------------------------ streams
    def _stream(self, role):
        assert role in self._generators, role
        self._consumed.add(role)
        return self._generators[role]

    def donor_pool(self, split):
        """The Native donor pool for one split: iid uniform token indices, `[N, 128]` int64."""
        assert split in SPLITS, split
        return seed_cost(self._stream(DONOR_ROLE[split]), self.seeds[DONOR_ROLE[split]],
                         POOL_SIZES[split])

    def permutations(self, split):
        """`[L, N]` int64: one independent plain permutation of `0..N-1` per position."""
        assert split in SPLITS, split
        stream = self._stream(PERMUTATION_ROLE[split])
        return torch.stack([torch.randperm(POOL_SIZES[split], generator=stream)
                            for _ in range(LENGTH)])

    def student_init_seed(self):
        return self.seeds["student_init"]

    def minibatch_schedule(self, pool_size, batch_size, steps):
        """Frozen minibatch schedule shared by the two arms of a pair."""
        permutation = torch.randperm(pool_size, generator=self._stream("minibatch_schedule"))
        return [[int(i) for i in permutation[(s * batch_size) % pool_size:
                                             (s * batch_size) % pool_size + batch_size]]
                for s in range(steps)]

    def context_donor_derangement(self, size=1024):
        """The evaluation-only context-intervention donor object (never part of the treatment)."""
        from .b0_phase1 import derangement
        return derangement(size, self._stream("context_donor"))

    @property
    def consumed_roles(self):
        return sorted(self._consumed)


def fixture_streams(fixture_replicate, root=FIXTURE_NAMESPACE):
    """Deterministic NON-FORMAL seeds for preflight fixtures only.

    Values are derived under `FIXTURE_NAMESPACE` and are explicitly not the formal campaign seeds.
    """
    root_seed = stream_seed(root, "root")
    seeds = {role: stream_seed(root_seed, f"{root}/replicate-{int(fixture_replicate):02d}/{role}")
             for role in ROLES}
    return Id2Streams(seeds, label=f"fixture-replicate-{fixture_replicate:02d}")


# ------------------------------------------------------------------------------- construction
def build_arms(model, block, donor_pool, permutations, device, block_index=5):
    """Construct the Native and Q_pos arms plus independently recomputed labels.

    `H_pos[n, t] = H_nat[permutations[t, n], t]` is realised position by position so that no
    `[N, L, d]` int64 index tensor is ever materialised.

    `block_index` is the zero-based block whose residual update is the target. The C7 guard asserts
    that the boundary was produced by exactly that block: a block-5 boundary paired with block-9
    labels is a silent, shape-compatible and scientifically fatal mismatch, so it must fail loudly.
    """
    n = donor_pool.shape[0]
    assert permutations.shape == (LENGTH, n), (permutations.shape, (LENGTH, n))
    assert block is model.transformer.h[block_index], (
        f"C7: the label block (index {list(model.transformer.h).index(block)}) "
        f"is not model.transformer.h[{block_index}]")
    hidden_nat = prefix_states(model, donor_pool, device, block_index=block_index)
    hidden_pos = torch.stack([hidden_nat[permutations[t], t, :] for t in range(LENGTH)], dim=1)
    labels_nat = teacher_update(block, hidden_nat)
    labels_pos = teacher_update(block, hidden_pos)
    return {"donor_pool": donor_pool, "permutations": permutations,
            "H_nat": hidden_nat, "H_pos": hidden_pos,
            "Y_nat": labels_nat, "Y_pos": labels_pos, "block_index": block_index}


def tensor_sha(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def _row_bytes(matrix):
    """`[N, d]` float32 -> `[N, d*4]` uint8, bit-exact."""
    array = np.ascontiguousarray(matrix.detach().cpu().numpy())
    return array.view(np.uint8).reshape(array.shape[0], -1)


def sorted_row_bytes(matrix):
    """Rows of `matrix` sorted lexicographically by their exact bytes (a canonical multiset form).

    This is an exact, collision-free substitute for sorted row hashes: two matrices have the same
    multiset of rows iff their canonical byte forms are identical.
    """
    raw = _row_bytes(matrix)
    order = np.lexsort(raw[:, ::-1].T)
    return raw[order]


def multiset_equal_exact(matrix_a, matrix_b):
    """Exact per-position multiset equality (bitwise; no floating tolerance)."""
    if matrix_a.shape != matrix_b.shape:
        return False
    return np.array_equal(sorted_row_bytes(matrix_a), sorted_row_bytes(matrix_b))


def multiset_digest(arms):
    """Per-position multiset commitments for the Native and Q_pos arms of one split.

    Records, per position, a digest of each arm's canonical sorted-row byte form, plus a single
    combined digest over all positions.
    """
    per_position = {"native": [], "pos": []}
    combined = hashlib.sha256()
    for t in range(LENGTH):
        native = hashlib.sha256(sorted_row_bytes(arms["H_nat"][:, t, :]).tobytes()).hexdigest()
        pos = hashlib.sha256(sorted_row_bytes(arms["H_pos"][:, t, :]).tobytes()).hexdigest()
        per_position["native"].append(native)
        per_position["pos"].append(pos)
        combined.update(native.encode())
        combined.update(pos.encode())
    per_position["combined"] = combined.hexdigest()
    return per_position


def check_construction_invariants(arms, rtol=None, width=768):
    """Invariants C1-C4 of the accepted design. `rtol` exists only to force a failure in fixtures.

    C1 shape/dtype, C2 true permutations, C3 inverse-permutation round trip (bitwise, an independent
    route from the forward gather), C4 exact per-position multiset identity.
    """
    failures = []
    n = arms["donor_pool"].shape[0]
    for name, key in (("H_nat", "H_nat"), ("H_pos", "H_pos"), ("Y_nat", "Y_nat"),
                      ("Y_pos", "Y_pos")):
        tensor = arms[key]
        if tuple(tensor.shape) != (n, LENGTH, width):
            failures.append(f"C1 {name} shape {tuple(tensor.shape)}")
        if tensor.dtype != torch.float32:
            failures.append(f"C1 {name} dtype {tensor.dtype}")

    permutations = arms["permutations"]
    if tuple(permutations.shape) != (LENGTH, n):
        failures.append(f"C2 permutation shape {tuple(permutations.shape)}")
    else:
        expected = torch.arange(n, dtype=permutations.dtype)
        if not all(bool(torch.equal(permutations[t].sort().values, expected))
                   for t in range(LENGTH)):
            failures.append("C2 permutations are not permutations of 0..N-1")

    # C3: an INDEPENDENT route — map Q_pos back through the inverse permutation and compare bytes.
    # It is only meaningful when the permutations are bijective (C2), so it is skipped otherwise:
    # an inverse of a non-bijection is undefined and would index with uninitialised memory.
    bijective = tuple(permutations.shape) == (LENGTH, n) and all(
        bool(torch.equal(permutations[t].sort().values,
                         torch.arange(n, dtype=permutations.dtype))) for t in range(LENGTH))
    if not bijective:
        failures.append("C3 skipped: permutations are not bijective (see C2)")
    if bijective:
        inverse = torch.full_like(permutations, -1)
        index = torch.arange(n).expand(LENGTH, n).contiguous()
        inverse.scatter_(1, permutations, index)
        for t in range(LENGTH):
            round_trip = arms["H_pos"][inverse[t], t, :]
            if not torch.equal(round_trip, arms["H_nat"][:, t, :]):
                failures.append(f"C3 inverse round trip differs at position {t}")
                break

    # C4: exact per-position multiset identity of the empirical vector marginals.
    digests = multiset_digest(arms)
    if digests["native"] != digests["pos"]:
        bad = [t for t in range(LENGTH) if digests["native"][t] != digests["pos"][t]]
        failures.append(f"C4 marginal multiset differs at positions {bad[:8]}"
                        f"{'...' if len(bad) > 8 else ''} ({len(bad)} positions)")

    if rtol is not None:                                  # fixture-only forced-failure hook
        failures.append("forced failure via rtol fixture hook")
    return {"passed": not failures, "failures": failures, "positions": LENGTH,
            "multiset_digest": digests}


def check_split_independence(pools):
    """C5: report exact donor-row collisions across the three splits. Never rejects anything."""
    hashes = {split: {tensor_sha(pools[split][i:i + 1]) for i in range(pools[split].shape[0])}
              for split in SPLITS}
    collisions = {}
    for a, b in (("train", "dev"), ("train", "audit"), ("dev", "audit")):
        collisions[f"{a}_{b}"] = len(hashes[a] & hashes[b])
    return {"collisions": collisions, "rejected": False,
            "note": "accidental iid collisions are recorded, never rejected (frozen rule)"}


def check_label_provenance(block, arms, tolerance=1e-6):
    """C6: labels must be recomputed from the FINAL tensors; a spliced label must be rejected."""
    recomputed = {"nat": teacher_update(block, arms["H_nat"]),
                  "pos": teacher_update(block, arms["H_pos"])}
    direct = {arm: {"max_abs_difference": float((recomputed[arm] - arms[f"Y_{arm}"]).abs().max()),
                    "accepted": bool(torch.equal(recomputed[arm], arms[f"Y_{arm}"]))}
              for arm in ("nat", "pos")}
    # negative fixture: splice the two arms' first halves — a label that is NOT a direct query
    spliced = torch.cat([arms["Y_nat"][:, :LENGTH // 2, :], arms["Y_pos"][:, LENGTH // 2:, :]], dim=1)
    reference = teacher_update(block, torch.cat(
        [arms["H_nat"][:, :LENGTH // 2, :], arms["H_pos"][:, LENGTH // 2:, :]], dim=1))
    spliced_difference = float((spliced - reference).abs().max())
    return {"direct": direct, "direct_accepted": all(e["accepted"] for e in direct.values()),
            "spliced_fixture_max_abs_difference": spliced_difference,
            "spliced_fixture_rejected": bool(spliced_difference > tolerance),
            "tolerance": tolerance}


# ------------------------------------------------------------------------------ blind progress
PROGRESS_ALLOWED_KEYS = frozenset({
    "completed_pairs", "total_pairs", "completed_students", "total_students",
    "running_replicate", "infrastructure_events", "algorithmic_failures", "failures",
    "elapsed_seconds", "status", "updated_utc", "replicate_id", "arm", "category", "reason",
    "steps", "terminal_step", "seconds", "nonfinite", "clipping_fraction",
    "own_q_raw_error", "own_q_normalized_error", "hashes", "checkpoint_sha256",
    "hard_stops", "unclassified_hard_stops", "invalidity_events",
})


def assert_blind_progress(record):
    """Reject any progress payload that could surface a target outcome during training."""
    forbidden = [key for key in record if key not in PROGRESS_ALLOWED_KEYS]
    assert not forbidden, f"progress payload exposes non-whitelisted keys: {sorted(forbidden)}"
    return True


# ----------------------------------------------------------------------------------- fixtures
def synthetic_arms(count=64, length=LENGTH, width=8, seed=11):
    """Small deterministic synthetic arms for fixtures (no Teacher, no target, no training)."""
    g = torch.Generator().manual_seed(seed)
    hidden = torch.randn(count, length, width, generator=g)
    permutations = torch.stack([torch.randperm(count, generator=g) for _ in range(length)])
    pos = torch.stack([hidden[permutations[t], t, :] for t in range(length)], dim=1)
    return {"donor_pool": torch.zeros(count, VOCAB, dtype=torch.int64),
            "permutations": permutations, "H_nat": hidden, "H_pos": pos,
            "Y_nat": hidden.clone(), "Y_pos": pos.clone()}


# ---------------------------------------------------------------------------- failure semantics
#: the three REVIEWED categories a failure may finally be assigned.
REVIEWED_FAILURE_CATEGORIES = ("INFRASTRUCTURE", "ALGORITHMIC", "INVALIDITY")
#: the non-replayable initial state of an unrecognised failure. It is NOT a diagnosis: it is a hard
#: stop that persists until a named review explicitly reclassifies it (see `reclassify_failure`).
UNCLASSIFIED = "UNCLASSIFIED_HARD_STOP"
FAILURE_CATEGORIES = (UNCLASSIFIED,) + REVIEWED_FAILURE_CATEGORIES

FAILURE_POLICY = {
    # an unknown exception may actually be implementation invalidity or a new algorithmic outcome,
    # so it must never be presumed infrastructure merely because it is not recognised: no replay,
    # no replacement seed, no continuation, and no action at all until it is reclassified.
    UNCLASSIFIED: {"hard_stop": True, "replay_allowed": False, "replacement_seed_allowed": False,
                   "continue_replicate": False, "preserve_exception_log_and_state": True,
                   "requires_explicit_reclassification": True,
                   "counts_as_scientific_outcome": False, "may_trigger_not_estimable": False},
    "INFRASTRUCTURE": {"replay_allowed": True, "replay_requires_identical_dataset": True,
                       "replay_requires_identical_rng": True,
                       "replay_requires_identical_seed": True, "replacement_seed_allowed": False,
                       "preserve_partial_under_superseded": True,
                       "counts_as_scientific_outcome": False,
                       "may_trigger_not_estimable": False},
    "ALGORITHMIC": {"replay_allowed": False, "replacement_seed_allowed": False,
                    "preserve_record": True, "counts_as_scientific_outcome": True,
                    "retune_allowed": False, "replay_to_obtain_better_result": False,
                    "may_trigger_not_estimable": True},
    "INVALIDITY": {"hard_stop": True, "replay_allowed": False,
                   "replacement_seed_allowed": False, "counts_as_scientific_outcome": False,
                   "may_trigger_not_estimable": True},
}

#: Automatic infrastructure classification is deliberately NARROW and fail-closed. Only these
#: exception types are infrastructure by type, and they all denote external/resource failure or an
#: unreachable peer rather than anything about the implementation:
#:   MemoryError     resource exhaustion
#:   ConnectionError network/transport interruption (includes BrokenPipe/ConnectionReset/Aborted)
#:   TimeoutError    a peer or device did not answer in time
#: Note that ConnectionError and TimeoutError are themselves OSError subclasses, so they are tested
#: BEFORE any errno logic.
INFRASTRUCTURE_SAFE_TYPES = (MemoryError, ConnectionError, TimeoutError)

#: Explicit frozen errno allowlist. Each entry must denote external/resource failure. Deliberately
#: ABSENT, because they usually indicate the implementation rather than the environment:
#:   ENOENT  (FileNotFoundError, missing path)
#:   EACCES / EPERM (PermissionError, wrong permissions or configuration)
#:   EEXIST, EISDIR, ENOTDIR, EINVAL (configuration/path errors)
#: Generic OSError is therefore NOT infrastructure by superclass membership: it must match an
#: allowlisted errno, otherwise it is UNCLASSIFIED_HARD_STOP.
INFRASTRUCTURE_ERRNO_ALLOWLIST = frozenset({
    errno.ENOSPC,          # no space left on device — disk exhaustion
    errno.ENOMEM,          # out of memory reported through errno
    errno.EIO,             # low-level I/O error (device/filesystem failure)
    errno.EMFILE,          # process file-descriptor limit reached
    errno.ENFILE,          # system-wide file-descriptor limit reached
    errno.EAGAIN,          # transient resource contention
    errno.ETIMEDOUT,       # device/peer timeout
    errno.ENETDOWN,        # network down
    errno.ENETUNREACH,     # network unreachable
    errno.EHOSTUNREACH,    # host unreachable
})


def _is_automatic_infrastructure(error):
    """True only for a narrow, explicitly allowlisted external/resource failure signature."""
    if isinstance(error, INFRASTRUCTURE_SAFE_TYPES):
        return True
    if isinstance(error, OSError) and error.errno in INFRASTRUCTURE_ERRNO_ALLOWLIST:
        return True
    return False


def classify_failure(error=None, invariant_failure=None, nonfinite_event=None,
                     finite_poor_outcome=None):
    """Map one implementation outcome onto the frozen failure policy.

    Precedence and defaults:

      1. an explicit treatment/pairing/target/checkpoint identity failure -> INVALIDITY (hard stop);
      2. a non-finite loss/gradient/parameter event, or a finite-but-poor learning outcome
         -> ALGORITHMIC (a scientific outcome; the frozen NOT_ESTIMABLE path applies if a required
         endpoint becomes uncomputable);
      3. a NARROW, explicitly allowlisted infrastructure signature — a safe type
         (MemoryError / ConnectionError / TimeoutError) or an allowlisted errno denoting external
         resource or network failure -> INFRASTRUCTURE (exact same-data / same-RNG / same-seed
         replay only). Generic `OSError` is NOT infrastructure by superclass membership, and
         `FileNotFoundError` / `PermissionError` / path and configuration errors are NOT
         infrastructure;
      4. **anything else, including an unrecognised exception -> UNCLASSIFIED_HARD_STOP**: no
         automatic replay, no replacement seed, no continuation of the replicate, preserve the
         exception/log/state, and take no action until a named review explicitly reclassifies it.

    An unknown exception is never presumed infrastructure merely because it is not recognised.
    Nothing here ever authorises a replacement seed.
    """
    if invariant_failure:
        category, detail = "INVALIDITY", str(invariant_failure)
    elif nonfinite_event:
        category, detail = "ALGORITHMIC", str(nonfinite_event)
    elif finite_poor_outcome:
        category, detail = "ALGORITHMIC", str(finite_poor_outcome)
    elif error is not None and _is_automatic_infrastructure(error):
        category, detail = "INFRASTRUCTURE", f"{type(error).__name__}: {error}"
    elif error is not None:
        category = UNCLASSIFIED
        detail = (f"unrecognised exception {type(error).__name__} (errno={getattr(error, 'errno', None)}): "
                  f"{error} — not presumed infrastructure; requires explicit review before any action")
    else:
        category, detail = UNCLASSIFIED, "no outcome supplied — requires explicit review"
    return {"category": category, "detail": detail, "policy": dict(FAILURE_POLICY[category]),
            "requires_operator_review": category == UNCLASSIFIED,
            "replacement_seed_allowed": False,
            "reclassified": False}


#: statuses that represent a RESOLVED scientific state, eligible for target evaluation.
#: ALGORITHMIC is a resolved scientific OUTCOME (the frozen NOT_ESTIMABLE rules apply downstream if
#: a required endpoint cannot be computed) — it is not a normal completion and is never rewritten
#: into one.
EVALUATION_ELIGIBLE_STATUSES = ("COMPLETE", "ALGORITHMIC")

#: statuses that OPERATIONALLY STOP the campaign. They terminate the *attempt*, but they are NOT
#: evaluation-eligible resolved states: formal target evaluation must refuse while any is present.
EVALUATION_BLOCKING_STATUSES = {
    UNCLASSIFIED: "unresolved unknown failure — operationally stopped, not an evaluation-eligible "
                  "resolved state; requires a named review and explicit reclassification first",
    "INVALIDITY": "implementation invalidity — campaign hard stop; the campaign must not proceed as "
                  "though it were valid",
    "INFRASTRUCTURE": "unresolved infrastructure failure — an exact same-data / same-RNG / same-seed "
                      "replay is still pending",
}


def evaluation_eligibility(arm_statuses, expected_students=PLANNED_STUDENTS, replicates=None):
    """Decide whether FORMAL TARGET EVALUATION may proceed.

    `arm_statuses` maps `replicate_id -> {arm_name: status}` (as recorded in the replicate records).
    `replicates` is the replicate id iterable to require; it defaults to 1..REPLICATES, which
    preserves the historical behaviour exactly. A 24-replicate campaign must pass its own set, or
    the 24 absent replicates would make every evaluation refuse.

    Evaluation is eligible only when every planned Student carries a RESOLVED scientific status.
    An operationally stopped attempt is terminal for that attempt but is **not** an eligible resolved
    state, so the presence of any blocking status makes the whole barrier refuse. A status that is
    absent, empty or unrecognised also refuses — silence is never treated as success.
    """
    counts, blocking, missing = {}, [], []
    observed = 0
    for replicate in (replicates if replicates is not None else range(1, REPLICATES + 1)):
        arms = arm_statuses.get(replicate) or {}
        for arm in ("nat", "pos"):
            observed += 1
            status = arms.get(arm)
            if status is None:
                missing.append({"replicate": replicate, "arm": arm, "status": None})
                continue
            counts[status] = counts.get(status, 0) + 1
            if status in EVALUATION_BLOCKING_STATUSES:
                blocking.append({"replicate": replicate, "arm": arm, "status": status,
                                 "why": EVALUATION_BLOCKING_STATUSES[status]})
            elif status not in EVALUATION_ELIGIBLE_STATUSES:
                blocking.append({"replicate": replicate, "arm": arm, "status": status,
                                 "why": "unrecognised status — not a resolved state"})
    resolved = sum(counts.get(status, 0) for status in EVALUATION_ELIGIBLE_STATUSES)
    eligible = (not blocking and not missing and observed == expected_students
                and resolved == expected_students)
    return {"eligible": eligible, "expected_students": expected_students,
            "observed_students": observed, "resolved_students": resolved,
            "status_counts": counts, "blocking": blocking, "missing": missing,
            "operationally_stopped": bool(blocking) or bool(missing),
            "decision": ("ELIGIBLE" if eligible else "REFUSED"),
            "rule": "target evaluation proceeds only when every planned Student carries a resolved "
                    "scientific status (COMPLETE) or a resolved scientific outcome (ALGORITHMIC); "
                    "UNCLASSIFIED_HARD_STOP, INVALIDITY and unresolved INFRASTRUCTURE refuse"}


def reclassify_failure(record, category, reviewer, rationale):
    """Explicitly reclassify an UNCLASSIFIED_HARD_STOP after named review.

    This is the ONLY way an unrecognised failure may leave its hard-stop state, and it may only be
    reclassified into one of the three reviewed categories. The reviewer and rationale are recorded.
    Reclassification to INFRASTRUCTURE still forbids a replacement seed.
    """
    assert record.get("category") == UNCLASSIFIED,         f"only {UNCLASSIFIED} records may be reclassified, got {record.get('category')!r}"
    assert category in REVIEWED_FAILURE_CATEGORIES,         f"reclassification target must be one of {REVIEWED_FAILURE_CATEGORIES}, got {category!r}"
    assert reviewer and str(reviewer).strip(), "reclassification requires a named reviewer"
    assert rationale and str(rationale).strip(), "reclassification requires a recorded rationale"
    return {"category": category, "detail": record["detail"],
            "policy": dict(FAILURE_POLICY[category]), "requires_operator_review": False,
            "replacement_seed_allowed": False, "reclassified": True,
            "reclassified_from": UNCLASSIFIED, "reviewer": reviewer, "rationale": rationale}


def assert_no_formal_seeds(path=Path("manifests/h9_id2_rng_manifest.json")):
    """Guard for the preflight phase: no formal RNG manifest may exist yet.

    `path` defaults to the historical ID2 path; a later campaign passes its own manifest path.
    """
    formal = Path(path)
    return {"formal_rng_manifest": str(formal), "formal_rng_manifest_exists": formal.exists(),
            "expected": False, "passed": not formal.exists()}


# ------------------------------------------------------------------ execution authorization
#: AUTHORIZATION STATE ONLY. These are the frozen formal identities that must exist and match before
#: a campaign may train. Nothing here alters how training is performed.
EXECUTION_TAG = "h9-id2-execution-frozen-v1"
FROZEN_RNG_MANIFEST = Path("manifests/h9_id2_rng_manifest.json")
FROZEN_RNG_MANIFEST_SHA256 = (
    "0e4fce9db29337ab4354c0bc7cb48fa7b01d80ca1275a2207bceadc9033a1496")
FROZEN_TARGET_LEDGER = Path("manifests/h9_id2_target_ledger.json")
FROZEN_TARGET_LEDGER_SHA256 = (
    "f0fcaaba9a0b58857f8d71175696dba5d99bef626a8d63a238950b8692ead9e3")


def _file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def execution_authorization(tag=EXECUTION_TAG):
    """Is formal training authorized? Authorization state only — no execution semantics.

    Authorized iff the frozen formal RNG manifest and the frozen target ledger both exist and match
    their recorded hashes, and the immutable execution tag resolves. A missing, altered or
    untagged state is never authorized.
    """
    checks, reasons = {}, []
    checks["rng_manifest_exists"] = FROZEN_RNG_MANIFEST.exists()
    if not checks["rng_manifest_exists"]:
        reasons.append(f"{FROZEN_RNG_MANIFEST} is absent")
        checks["rng_manifest_hash_matches"] = False
    else:
        checks["rng_manifest_hash_matches"] = (
            _file_sha256(FROZEN_RNG_MANIFEST) == FROZEN_RNG_MANIFEST_SHA256)
        if not checks["rng_manifest_hash_matches"]:
            reasons.append("formal RNG manifest does not match its frozen SHA256")
    checks["target_ledger_exists"] = FROZEN_TARGET_LEDGER.exists()
    if not checks["target_ledger_exists"]:
        reasons.append(f"{FROZEN_TARGET_LEDGER} is absent")
        checks["target_ledger_hash_matches"] = False
    else:
        checks["target_ledger_hash_matches"] = (
            _file_sha256(FROZEN_TARGET_LEDGER) == FROZEN_TARGET_LEDGER_SHA256)
        if not checks["target_ledger_hash_matches"]:
            reasons.append("target ledger does not match its frozen SHA256")
    resolved = ""
    try:
        import subprocess
        # `--verify` is REQUIRED: plain `rev-parse <rev>` echoes its argument on failure, which
        # would make an absent tag look resolved.
        completed = subprocess.run(("git", "-c", f"safe.directory={Path.cwd().as_posix()}",
                                    "rev-parse", "--verify", "--quiet", f"{tag}^{{commit}}"),
                                   capture_output=True, text=True)
        resolved = completed.stdout.strip() if completed.returncode == 0 else ""
    except Exception as error:                                    # noqa: BLE001
        reasons.append(f"cannot resolve execution tag: {type(error).__name__}: {error}")
    checks["execution_tag_resolves"] = bool(resolved)
    checks["execution_tag_commit"] = resolved or None
    if not resolved:
        reasons.append(f"execution tag {tag} does not resolve")
    authorized = all(checks[key] for key in ("rng_manifest_exists", "rng_manifest_hash_matches",
                                             "target_ledger_exists", "target_ledger_hash_matches",
                                             "execution_tag_resolves"))
    return {"authorized": authorized, "tag": tag, "checks": checks, "reasons": reasons}
