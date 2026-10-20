"""CMR1 candidate governed qualification runtime: identity capture and numerical-mode assertions.

The candidate runtime is the repository's canonical isolated environment ``.venv`` (see ``README.md``),
which satisfies every pin in ``requirements-lock.txt``.  This module **asserts** the frozen numerical
mode rather than inheriting whatever the interpreter happens to default to, and records the resulting
identity for the future B11 qualification identity manifest.

Every setting below is tied to an actual frozen requirement or to measured runtime behaviour:

===========================  ==========================================================
setting                      basis
===========================  ==========================================================
FP32 / ``highest`` precision  B4 CS-CAL-1/CS-CAL-2 "FP32"; protocol "FP32; AMP off; TF32 off"
CUDA matmul TF32 off         same clauses ("TF32 disabled")
cuDNN TF32 off               same clauses; the ambient default is **True** and must be overridden
cuDNN ``deterministic=True`` same clauses ("the frozen deterministic settings enabled")
cuDNN ``benchmark=False``    benchmark autotuning selects nondeterministic algorithms
deterministic algorithms     same clauses; verified fail-closed rather than warn-only
autocast / AMP off           same clauses ("autocast disabled"); asserted, never entered
``CUBLAS_WORKSPACE_CONFIG``  **measured requirement**: under ``use_deterministic_algorithms(True)`` a
                             CUDA matmul raises ``RuntimeError`` naming this variable (CUDA >= 10.2);
                             with ``:4096:8`` the same matmul succeeds and repeats bit-identically
===========================  ==========================================================

``CUBLAS_WORKSPACE_CONFIG`` is read by cuBLAS when the CUDA context is created, so it must be set
**before** the first CUDA operation.  :func:`enforce_cublas_workspace_config` therefore runs at import
time of this module; importing it first is part of the governed-runtime discipline.
"""
from __future__ import annotations

import os
import platform
import sys
from pathlib import Path

# --------------------------------------------------------------------------- frozen runtime pins
REQUIRED_TORCH = "2.8.0+cu128"
REQUIRED_TRANSFORMERS = "4.56.2"
REQUIRED_HUB = "0.36.2"
CUBLAS_WORKSPACE_CONFIG_VALUE = ":4096:8"
#: ``:16:8`` is the documented lower-memory alternative and was also verified to work; ``:4096:8``
#: is used because it is PyTorch's documented recommendation.  Either is deterministic - the choice
#: is a runtime-configuration detail, not a scientific one.

CANDIDATE_RUNTIME_NAME = "cmr1-qualification-candidate"


class GovernedRuntimeError(RuntimeError):
    """The candidate governed runtime does not satisfy a frozen requirement."""


def enforce_cublas_workspace_config() -> str:
    """Set ``CUBLAS_WORKSPACE_CONFIG`` if absent.  Must precede CUDA initialisation."""
    current = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if current is None:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = CUBLAS_WORKSPACE_CONFIG_VALUE
        return CUBLAS_WORKSPACE_CONFIG_VALUE
    return current


# Set at import time: cuBLAS reads this when the context is created.
enforce_cublas_workspace_config()

import torch  # noqa: E402


def enforce_governed_numerical_mode() -> dict:
    """Assert the frozen numerical mode; return a record of the asserted state.

    Fails closed (:class:`GovernedRuntimeError`) if any requirement cannot be met.
    """
    state: dict[str, object] = {}

    # FP32 / exact matmul precision
    torch.set_float32_matmul_precision("highest")
    if torch.get_float32_matmul_precision() != "highest":
        raise GovernedRuntimeError("failed to assert float32 matmul precision 'highest'")
    state["float32_matmul_precision"] = torch.get_float32_matmul_precision()

    # TF32 off wherever the backend exposes it
    if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
        torch.backends.cuda.matmul.allow_tf32 = False
        state["cuda_matmul_allow_tf32"] = bool(torch.backends.cuda.matmul.allow_tf32)
        if torch.backends.cuda.matmul.allow_tf32:
            raise GovernedRuntimeError("failed to disable CUDA matmul TF32")
    if hasattr(torch.backends.cudnn, "allow_tf32"):
        torch.backends.cudnn.allow_tf32 = False
        state["cudnn_allow_tf32"] = bool(torch.backends.cudnn.allow_tf32)
        if torch.backends.cudnn.allow_tf32:
            raise GovernedRuntimeError("failed to disable cuDNN TF32")

    # deterministic settings
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=False)
    state["cudnn_deterministic"] = bool(torch.backends.cudnn.deterministic)
    state["cudnn_benchmark"] = bool(torch.backends.cudnn.benchmark)
    state["deterministic_algorithms"] = bool(torch.are_deterministic_algorithms_enabled())
    state["determinism_warn_only"] = bool(torch.is_deterministic_algorithms_warn_only_enabled())
    if not state["deterministic_algorithms"] or state["determinism_warn_only"]:
        raise GovernedRuntimeError("failed to enable fail-closed deterministic algorithms")

    # autocast / AMP must be OFF and never entered
    state["cuda_autocast_active"] = bool(torch.is_autocast_enabled())
    try:
        state["cpu_autocast_active"] = bool(torch.is_autocast_enabled("cpu"))
    except TypeError:                                                       # pragma: no cover
        state["cpu_autocast_active"] = bool(torch.is_autocast_cpu_enabled())
    state["cublas_workspace_config"] = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if state["cuda_autocast_active"] or state["cpu_autocast_active"]:
        raise GovernedRuntimeError("an autocast context is active; the governed mode requires AMP off")
    if not state["cublas_workspace_config"]:
        raise GovernedRuntimeError(
            "CUBLAS_WORKSPACE_CONFIG is unset; deterministic CUDA matmul fails without it")

    state["numerical_mode"] = "ASSERTED"
    return state


def assert_reference_runtime() -> dict:
    """Assert the scientific reference implementation versions.

    Uses the repository ``.venv`` install when it already carries the pinned versions; otherwise
    falls back to the isolated extracted tree used during Stage-A/I1-I3 verification.  The installed
    *ambient* transformers must never silently become the reference path.
    """
    import transformers
    import huggingface_hub

    if transformers.__version__ != REQUIRED_TRANSFORMERS:
        raise GovernedRuntimeError(
            f"loaded transformers=={transformers.__version__}; the scientific reference requires "
            f"=={REQUIRED_TRANSFORMERS}")
    if huggingface_hub.__version__ != REQUIRED_HUB:
        raise GovernedRuntimeError(
            f"loaded huggingface_hub=={huggingface_hub.__version__}; the lock pins =={REQUIRED_HUB}")
    return {
        "transformers": transformers.__version__,
        "transformers_module": str(Path(transformers.__file__).resolve()),
        "huggingface_hub": huggingface_hub.__version__,
        "huggingface_hub_module": str(Path(huggingface_hub.__file__).resolve()),
    }


def runtime_identity() -> dict:
    """Capture the full candidate governed runtime identity (no governed values involved)."""
    identity: dict[str, object] = {
        "candidate_runtime_name": CANDIDATE_RUNTIME_NAME,
        "python": sys.version.split()[0],
        "python_implementation": platform.python_implementation(),
        "python_executable": sys.executable,
        "os": platform.platform(),
        "machine": platform.machine(),
        "torch": torch.__version__,
        "torch_git_version": getattr(torch.version, "git_version", None),
        "torch_module": str(Path(torch.__file__).resolve()),
        "torch_cuda_version": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_count": torch.cuda.device_count(),
        "cudnn_version": torch.backends.cudnn.version() if hasattr(torch.backends, "cudnn") else None,
    }
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(torch.cuda.current_device())
        identity.update({
            "device_name": props.name,
            "device_compute_capability": f"{props.major}.{props.minor}",
            "device_total_memory_bytes": int(props.total_memory),
            "device_multi_processor_count": int(props.multi_processor_count),
        })
    identity["numerical_mode_asserted"] = enforce_governed_numerical_mode()
    identity["reference_runtime"] = assert_reference_runtime()
    identity["cublas_workspace_config_value"] = CUBLAS_WORKSPACE_CONFIG_VALUE
    return identity


def assert_candidate_runtime_satisfies_pins() -> dict:
    """Verify the locked dependency pins this runtime claims to satisfy."""
    import importlib.metadata as md

    required = {
        "torch": REQUIRED_TORCH,
        "transformers": REQUIRED_TRANSFORMERS,
        "huggingface_hub": REQUIRED_HUB,
    }
    observed = {}
    for package, expected in required.items():
        try:
            found = md.version(package)
        except md.PackageNotFoundError:                                     # pragma: no cover
            raise GovernedRuntimeError(f"{package} is not installed in this runtime")
        observed[package] = found
        if found != expected:
            raise GovernedRuntimeError(
                f"{package}=={found} does not satisfy the locked pin =={expected}")
    return observed
