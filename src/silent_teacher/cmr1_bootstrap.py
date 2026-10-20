"""CMR1 governed-process bootstrap and complete lock enforcement (execution-critical).

Correctness must not depend on import luck.  This module establishes the required **process-level**
environment state *before* torch/CUDA/cuBLAS is initialised, and fails closed rather than silently
overwriting an unsafe state that is already active.

Import this module first in any governed process::

    import silent_teacher.cmr1_bootstrap as bootstrap
    bootstrap.assert_bootstrap_ready()          # before touching torch
    bootstrap.assert_complete_lock()            # all 29 authoritative pins
    bootstrap.assert_governed_numerical_boundary()   # mode + required dtypes

``CUBLAS_WORKSPACE_CONFIG`` is read by cuBLAS when the CUDA context is created, so it must be set before
the first CUDA operation.  Two ways to get this wrong are both made to fail closed:

* the variable is absent (deterministic CUDA matmul then raises, as measured) -- we set it, or refuse;
* the variable is present with an **incorrect** value -- we refuse rather than overwrite, because an
  externally supplied value may indicate a deliberately different (non-frozen) numerical mode.
"""
from __future__ import annotations

import os
import sys
import weakref
from pathlib import Path

CUBLAS_WORKSPACE_CONFIG_VALUE = ":4096:8"
LOCK_FILENAME = "requirements-lock.txt"

REQUIRED_PINS: dict[str, str] = {
    "certifi": "2026.7.22", "charset-normalizer": "3.5.1", "colorama": "0.4.6",
    "filelock": "3.32.7", "fsspec": "2026.7.0", "huggingface_hub": "0.36.2", "idna": "3.19",
    "iniconfig": "2.3.0", "Jinja2": "3.1.6", "MarkupSafe": "3.0.3", "mpmath": "1.3.0",
    "networkx": "3.6.1", "numpy": "2.3.3", "packaging": "26.3", "pluggy": "1.6.0",
    "Pygments": "2.21.0", "pytest": "8.4.2", "PyYAML": "6.0.3", "regex": "2026.9.10",
    "requests": "2.34.2", "safetensors": "0.6.2", "setuptools": "78.1.0", "sympy": "1.14.0",
    "tokenizers": "0.22.2", "torch": "2.8.0+cu128", "tqdm": "4.70.1",
    "transformers": "4.56.2", "typing_extensions": "4.16.0", "urllib3": "2.8.0",
}


class GovernedBootstrapError(RuntimeError):
    """The governed process bootstrap could not establish or verify the required state."""


class LockMismatch(GovernedBootstrapError):
    """The installed environment does not satisfy the authoritative lock."""


def parse_lock(path: str | Path = LOCK_FILENAME) -> dict[str, str]:
    """Parse the authoritative lock file into ``{package: version}`` (comments ignored)."""
    pins: dict[str, str] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "==" not in line:
            continue
        name, version = line.split("==", 1)
        pins[name.strip()] = version.strip()
    return pins


def assert_complete_lock(path: str | Path = LOCK_FILENAME) -> dict:
    """Validate **every** authoritative pin, not a headline subset.  Fails closed."""
    declared = parse_lock(path)
    if not declared:
        raise LockMismatch(f"the lock at {path} declares no pins")
    if set(declared) != set(REQUIRED_PINS):
        missing = sorted(set(REQUIRED_PINS) - set(declared))
        extra = sorted(set(declared) - set(REQUIRED_PINS))
        raise LockMismatch(
            f"HARD STOP - the lock file does not match the authoritative pin set "
            f"(missing {missing}, extra {extra})")

    import importlib.metadata as md
    observed: dict[str, str] = {}
    mismatched: list[str] = []
    for package, expected in sorted(declared.items()):
        try:
            found = md.version(package)
        except md.PackageNotFoundError:
            mismatched.append(f"{package}: NOT INSTALLED (expected {expected})")
            continue
        observed[package] = found
        if found != expected:
            mismatched.append(f"{package}: {found} != {expected}")
    if mismatched:
        raise LockMismatch(
            "HARD STOP - the governed runtime does not satisfy the authoritative lock: "
            + "; ".join(mismatched))
    return {"pin_count": len(declared), "observed": observed, "complete_lock": "PASS"}


def establish_cublas_workspace_config() -> str:
    """Set ``CUBLAS_WORKSPACE_CONFIG``, refusing an incorrect pre-existing value.

    Called at import time of this module.  It never overwrites a different value: an externally supplied
    value may encode a different numerical mode, and silently replacing it would hide that.
    """
    current = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if current is None:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = CUBLAS_WORKSPACE_CONFIG_VALUE
        return CUBLAS_WORKSPACE_CONFIG_VALUE
    if current != CUBLAS_WORKSPACE_CONFIG_VALUE:
        raise GovernedBootstrapError(
            f"HARD STOP - CUBLAS_WORKSPACE_CONFIG is {current!r}, but the frozen governed value is "
            f"{CUBLAS_WORKSPACE_CONFIG_VALUE!r}. Refusing to overwrite an externally supplied value: "
            "it may encode a deliberately different numerical mode.")
    return current


def assert_bootstrap_ready() -> dict:
    """Assert the bootstrap ran before torch/CUDA initialisation.

    Fails closed if torch is already imported or CUDA is already initialised with a state we cannot
    guarantee -- e.g. if the workspace configuration was established too late to affect the context.
    """
    if "torch" in sys.modules:
        import torch
        if torch.cuda.is_initialized():                                  # pragma: no cover
            raise GovernedBootstrapError(
                "HARD STOP - CUDA was already initialised; CUBLAS_WORKSPACE_CONFIG cannot be "
                "guaranteed to have applied to the created context. Start a fresh governed process "
                "that imports this module first.")
    configured = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if configured != CUBLAS_WORKSPACE_CONFIG_VALUE:
        raise GovernedBootstrapError(
            f"HARD STOP - CUBLAS_WORKSPACE_CONFIG is {configured!r}; the governed value "
            f"{CUBLAS_WORKSPACE_CONFIG_VALUE!r} must be established before CUDA initialisation")
    return {"cublas_workspace_config": configured, "torch_imported": "torch" in sys.modules,
            "bootstrap_ready": "PASS"}


# ``CUBLAS_WORKSPACE_CONFIG`` must exist before cuBLAS initialises, which is why this runs at import.
establish_cublas_workspace_config()


def assert_fp32(tensor, *, label: str) -> object:
    """Assert a governed tensor is FP32.  **No silent cast** -- a wrong dtype fails closed."""
    import torch

    if not isinstance(tensor, torch.Tensor):
        raise GovernedBootstrapError(f"{label} must be a tensor")
    if tensor.dtype != torch.float32:
        raise GovernedBootstrapError(
            f"HARD STOP - {label} has dtype {tensor.dtype}; governed FP32 objects are never silently "
            "cast into compliance")
    return tensor


def assert_governed_numerical_boundary(*, teacher=None, student=None,
                                       fixtures: dict | None = None,
                                       inputs: dict | None = None) -> dict:
    """Assert the full governed execution-boundary state in one place.

    Covers the numerical mode plus the required FP32 status of Teacher/Student parameters, governed
    inputs and governed fixture tensors.  Wrong dtype fails closed.
    """
    from silent_teacher import cmr1_runtime

    state = {"numerical_mode": cmr1_runtime.enforce_governed_numerical_mode()}
    for label, module in (("Teacher parameters", teacher), ("Student parameters", student)):
        if module is None:
            continue
        wrong = [name for name, parameter in module.named_parameters()
                 if parameter.dtype.__str__() != "torch.float32"]
        if wrong:
            raise GovernedBootstrapError(
                f"HARD STOP - {label} are not all FP32; offending: {wrong[:5]}")
        state[label] = "FP32"
    for group_name, bundle in (("governed inputs", inputs), ("governed fixtures", fixtures)):
        if not bundle:
            continue
        for name, tensor in bundle.items():
            assert_fp32(tensor, label=f"{group_name}[{name}]")
        state[group_name] = "FP32"
    state["execution_boundary"] = "ASSERTED"
    return state


# A runtime token is issued only after actual bootstrap validation.  Caller-provided booleans or
# lookalike objects never satisfy the entry check.  The registry carries the attested scope outside
# mutable token attributes; every numerical entry validates mode/dtype again before computation.
_RUNTIME_CONTEXTS: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


class GovernedRuntimeContext:
    __slots__ = ("__weakref__",)


def create_governed_runtime_context(*, capability, scope: str,
                                    lock_path: str | Path = LOCK_FILENAME) -> GovernedRuntimeContext:
    from silent_teacher.cmr1_authorization import capability_authorizes
    if not capability_authorizes(capability, scope):
        raise GovernedBootstrapError("a loader-issued capability for this execution scope is required")
    assert_bootstrap_ready()
    assert_complete_lock(lock_path)
    assert_governed_numerical_boundary()
    context = GovernedRuntimeContext()
    _RUNTIME_CONTEXTS[context] = scope
    return context


def assert_governed_entry(context, *, scope: str, teacher=None, student=None,
                          fixtures: dict | None = None, inputs: dict | None = None) -> dict:
    if not isinstance(context, GovernedRuntimeContext) or _RUNTIME_CONTEXTS.get(context) != scope:
        raise GovernedBootstrapError("governed execution requires an attested runtime context for this phase")
    # The context was minted before CUDA initialization.  Later governed entries must remain
    # executable after the first authorized CUDA operation, while still rejecting a changed cuBLAS
    # setting; re-running the pre-initialization check here would incorrectly halt every later step.
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != CUBLAS_WORKSPACE_CONFIG_VALUE:
        raise GovernedBootstrapError("cuBLAS workspace setting changed after runtime attestation")
    assert_complete_lock()
    return assert_governed_numerical_boundary(teacher=teacher, student=student,
                                               fixtures=fixtures, inputs=inputs)


def bootstrap_manifest() -> dict:
    return {
        "lock_file": LOCK_FILENAME,
        "pin_count": len(REQUIRED_PINS),
        "cublas_workspace_config_value": CUBLAS_WORKSPACE_CONFIG_VALUE,
        "ordering_requirement": "this module must be imported before torch/CUDA initialisation",
        "incorrect_preexisting_value": "rejected, never overwritten",
        "late_cuda_initialisation": "rejected -- a fresh governed process is required",
        "lock_validation": "all pins, fail closed",
        "dtype_policy": "wrong dtype fails closed; no silent cast",
    }
