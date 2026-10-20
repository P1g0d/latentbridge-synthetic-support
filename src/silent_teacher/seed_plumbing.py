"""Stage B seed plumbing (pre-freeze reproducibility fix).

Every trainable-model runner derives an explicit numeric initialization seed from a frozen
(master seed, namespace) pair and constructs its model inside `torch.random.fork_rng` with
that seed. Identical (master seed, namespace, architecture) therefore yields bit-identical
step-0 parameters regardless of any prior global-RNG consumption (GPU ladder, proxy
construction, pool generation, logging) — and the fork/restore means no side effects on the
surrounding RNG stream.

Batch schedules are materialized to disk as hashed artifacts; the train runner loads the
artifact instead of regenerating schedules inline.
"""
import hashlib
from pathlib import Path
import torch

from .probes import stream_seed


def init_seed(master_seed, namespace):
    """Explicit numeric initialization seed derived from the frozen (master seed, namespace)."""
    return stream_seed(master_seed, namespace)


def seeded_module(module_factory, master_seed, namespace, **kwargs):
    """Construct a module with an explicit namespace-derived initialization seed.

    Precise guarantee (audit item Q): `torch.random.fork_rng(devices=[])` snapshots and
    restores ONLY the CPU global RNG. The Stage B model initialization runs on CPU (modules
    are constructed on CPU and moved to the device afterwards), so this gives: deterministic
    step-0 parameters for identical (master_seed, namespace, architecture) regardless of any
    prior CPU RNG consumption, with no side effect on the CPU RNG stream. It does NOT isolate
    the CUDA RNG — a future initialization that constructs directly on CUDA (consuming CUDA
    randomness) would require device RNG isolation as well."""
    numeric_seed = init_seed(master_seed, namespace)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(numeric_seed)
        module = module_factory(**kwargs)
    return module, numeric_seed


def build_schedule_artifact(master_seed, namespace, pool_sequences, batch_sequences, steps, out_path):
    """Materialize the frozen sequence-index schedule to `out_path` and return its metadata.

    The PASSED namespace genuinely drives generation: the permutation is derived from
    `stream_seed(master_seed, namespace)` (contract fixed after the audit found the namespace
    parameter was silently ignored in favour of the legacy `b0_perm` stream). Same
    (master seed, namespace) => same schedule; different namespace => different schedule."""
    from .teacher_prefix import batch_indices, batch_permutation
    permutation = batch_permutation(master_seed, namespace, pool_sequences)
    schedule = [batch_indices(permutation, batch_sequences, step) for step in range(1, steps + 1)]
    payload = (torch.stack(schedule) if torch.is_tensor(schedule[0])
               else torch.tensor(schedule, dtype=torch.int64))
    torch.save(payload, out_path)
    digest = hashlib.sha256(payload.contiguous().numpy().tobytes()).hexdigest()
    return {"path": str(Path(out_path).as_posix()), "namespace": namespace,
            "numeric_seed": stream_seed(master_seed, namespace), "pool_sequences": pool_sequences,
            "batch_sequences": batch_sequences, "steps": steps,
            "schedule_sha256": digest,
            "first_effective_batches": [list(map(int, schedule[i])) for i in range(min(5, steps))]}


def load_schedule_artifact(path, expected_sha256=None):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    digest = hashlib.sha256(payload.contiguous().numpy().tobytes()).hexdigest()
    if expected_sha256 is not None:
        assert digest == expected_sha256, f"schedule artifact hash mismatch: {digest}"
    return payload.tolist(), digest


def tensor_sha256(tensor):
    """Device-agnostic stable tensor hash: detach -> cpu -> contiguous -> numpy -> SHA-256."""
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def rng_state_snapshots():
    """Global RNG state snapshots (Python / torch CPU / CUDA when available) plus hashes.

    Large states are pickled to binary snapshots; only hashes go into JSON manifests."""
    import pickle
    import random
    snapshots = {"python_random_state": pickle.dumps(random.getstate()),
                 "torch_cpu_rng_state": torch.get_rng_state()}
    if torch.cuda.is_available():
        snapshots["torch_cuda_rng_state"] = torch.cuda.get_rng_state()
    hashes = {}
    for key, value in snapshots.items():
        buffer = value if isinstance(value, torch.Tensor) else torch.frombuffer(
            bytearray(value), dtype=torch.uint8)
        hashes[key] = hashlib.sha256(buffer.contiguous().numpy().tobytes()).hexdigest()
    return snapshots, hashes
