"""Canonical serialization and content hashing for CMR1 governed realized objects.

One unambiguous representation per governed realized object (B11).  Every identity is a SHA-256 over
a canonical byte stream that records shape, dtype, logical semantics and the raw tensor payload, so a
regeneration either reproduces the identical bytes or fails closed.

Design rules:

* exactly **one** representation exists - there is no format switch, no legacy mode and no
  alternative ordering;
* the header is a **canonical JSON** document with sorted keys, UTF-8 and no insignificant whitespace;
* tensors are stored as **contiguous little-endian raw bytes** of the declared dtype, preceded by
  shape and dtype in the header, so the byte order never depends on the host;
* no wall-clock time, no filesystem path, no host identifier and no unordered-iteration artifact
  enters any identity;
* floats are hashed as their exact bit patterns - never re-rounded, re-formatted or narrowed.
"""
from __future__ import annotations

import hashlib
import json
import math
import struct
import sys
from typing import Mapping, Sequence

import torch

SERIALIZATION_FORMAT_VERSION = "latentbridge.cmr1.canonical_object.v1"

#: dtypes numpy cannot express directly; hashed through a byte view instead.
_BYTE_VIEW_FALLBACK = (torch.bfloat16, torch.float8_e4m3fn, torch.float8_e5m2)


class SerializationError(RuntimeError):
    """A canonical-object requirement was violated."""


def _canonical_json(document: Mapping[str, object]) -> bytes:
    def validate(value):
        if isinstance(value, Mapping):
            if any(type(key) is not str for key in value):
                raise SerializationError("canonical JSON keys must be exact strings; key coercion is forbidden")
            for item in value.values():
                validate(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                validate(item)
        elif type(value) is float and not math.isfinite(value):
            raise SerializationError("nonfinite canonical JSON metadata is forbidden")
        elif value is not None and type(value) not in (str, int, float, bool):
            raise SerializationError(f"unsupported canonical JSON metadata type {type(value).__name__}")
    validate(document)
    return json.dumps(document, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("utf-8")


def _dtype_token(tensor: torch.Tensor) -> str:
    return str(tensor.dtype).replace("torch.", "")


def _tensor_payload(tensor: torch.Tensor) -> bytes:
    """Canonical little-endian raw payload bytes for one tensor."""
    if not isinstance(tensor, torch.Tensor):
        raise SerializationError(f"expected a tensor, got {type(tensor).__name__}")
    if tensor.is_complex():
        raise SerializationError("complex tensors have no canonical CMR1 byte order")
    contiguous = tensor.detach().to("cpu").contiguous()

    if contiguous.dtype in _BYTE_VIEW_FALLBACK:
        return contiguous.view(torch.uint8).numpy().tobytes(order="C")

    array = contiguous.numpy()
    if array.dtype.itemsize > 1 and array.dtype.byteorder not in ("<", "|", "="):
        array = array.byteswap()                     # pragma: no cover - big-endian host
    elif array.dtype.itemsize > 1 and array.dtype.byteorder == "=" and sys.byteorder == "big":
        array = array.byteswap()                     # pragma: no cover - big-endian host
    return array.tobytes(order="C")


def serialize_named_tensors(named_tensors: Mapping[str, torch.Tensor], *,
                            semantics: Mapping[str, str] | None = None,
                            extra: Mapping[str, object] | None = None) -> bytes:
    """The single canonical byte stream for a named tensor collection."""
    if not named_tensors:
        raise SerializationError("a canonical object must contain at least one tensor")
    semantics = dict(semantics or {})
    entries = []
    payloads: list[bytes] = []
    for name in sorted(named_tensors):
        tensor = named_tensors[name]
        if tensor.dtype not in (torch.float32, torch.int64, torch.int32, torch.uint8,
                                torch.bool, * _BYTE_VIEW_FALLBACK):
            raise SerializationError(
                f"dtype {tensor.dtype} is not an approved canonical CMR1 payload dtype")
        entries.append({
            "name": name,
            "shape": [int(dim) for dim in tensor.shape],
            "dtype": _dtype_token(tensor),
            "semantics": semantics.get(name, "unspecified"),
        })
        payloads.append(_tensor_payload(tensor))
    header = {
        "format": SERIALIZATION_FORMAT_VERSION,
        "tensors": entries,
        "tensor_count": len(entries),
        "extra": dict(extra or {}),
    }
    blob = _canonical_json(header)
    return struct.pack("<Q", len(blob)) + blob + b"".join(payloads)


def sha256_of(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


def tensor_object_identity(named_tensors: Mapping[str, torch.Tensor], *,
                           semantics: Mapping[str, str] | None = None,
                           extra: Mapping[str, object] | None = None) -> str:
    """Content identity of a tensor collection."""
    return sha256_of(serialize_named_tensors(named_tensors, semantics=semantics, extra=extra))


def _typed_element(value: object) -> dict:
    """Type-preserving canonical encoding of one sequence element.

    **No lossy coercion.** ``int(value)`` would collapse ``1.5`` and ``1`` and silently destroy the
    bool/int distinction, so each element is encoded with an explicit type tag and its exact value.
    """
    if isinstance(value, bool):                       # bool before int: bool is an int subclass
        return {"t": "bool", "v": value}
    if isinstance(value, int):
        return {"t": "int", "v": str(value)}          # decimal, explicit width-agnostic encoding
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return {"t": "float", "v": repr(value)}   # explicit, never silently accepted as finite
        return {"t": "float", "v": value.hex()}       # exact binary representation, not decimal
    if isinstance(value, str):
        return {"t": "str", "v": value}               # exact, no normalization
    if isinstance(value, bytes):
        return {"t": "bytes", "v": value.hex()}
    if value is None:
        return {"t": "none"}
    raise SerializationError(
        f"unsupported sequence element type {type(value).__name__}; a lossy conversion would make "
        "the canonical representation non-injective")


def sequence_object_identity(sequence: "Sequence[object] | torch.Tensor", *, kind: str,
                             extra: Mapping[str, object] | None = None) -> str:
    """Content identity of a typed sequence object (e.g. a realized minibatch order).

    Injective over the supported element types: ``[1.5]`` and ``[1]`` differ, ``True`` and ``1``
    differ, an ``int`` and an equal-valued ``float`` differ, and a ``list`` is distinguished from a
    ``tuple`` by the container tag.
    """
    if isinstance(sequence, torch.Tensor):
        if sequence.dtype != torch.int64:
            raise SerializationError("sequence tensors must be int64")
        container = "tensor_int64"
        elements = [_typed_element(value) for value in sequence.reshape(-1).tolist()]
    elif isinstance(sequence, (list, tuple)):
        container = "list" if isinstance(sequence, list) else "tuple"
        elements = [_typed_element(value) for value in sequence]
    else:
        raise SerializationError(
            f"unsupported sequence container {type(sequence).__name__}; supported: list, tuple, "
            "int64 tensor")
    document = {
        "format": SERIALIZATION_FORMAT_VERSION,
        "kind": kind,
        "container": container,
        "length": len(elements),
        "shape": list(sequence.shape) if isinstance(sequence, torch.Tensor) else [len(sequence)],
        "elements": elements,
        "extra": dict(extra or {}),
    }
    return sha256_of(_canonical_json(document))


def permutation_object_identity(permutations: Sequence[Sequence[int]]) -> str:
    """Content identity of a realized per-position permutation object ``(pi_0 ... pi_{T-1})``.

    Validates that every position really is a permutation of ``0..donors-1`` before hashing.
    """
    if not isinstance(permutations, (list, tuple)):
        raise SerializationError("permutation positions must be a list or tuple")
    rows = []
    for position, perm in enumerate(permutations):
        if not isinstance(perm, (list, tuple)):
            raise SerializationError(f"position {position} must be a list or tuple")
        if any(type(value) is not int for value in perm):
            raise SerializationError(
                f"position {position} contains a non-canonical integer; floats, bools and "
                "coercible objects are forbidden")
        rows.append(list(perm))
    if not rows:
        raise SerializationError("a realized permutation object must have at least one position")
    width = len(rows[0])
    for index, row in enumerate(rows):
        if len(row) != width:
            raise SerializationError(f"position {index} has length {len(row)}, expected {width}")
        if sorted(row) != list(range(width)):
            raise SerializationError(f"position {index} is not a permutation of 0..{width - 1}")
    document = {
        "format": SERIALIZATION_FORMAT_VERSION,
        "kind": "cmr1.qpos.position_permutations",
        "positions": len(rows),
        "donors": width,
        "index_type": "python-int-exact",
        "index_domain": [0, width - 1],
        "shape": [len(rows), width],
        "permutations": rows,
    }
    return sha256_of(_canonical_json(document))


def file_sha256(path: str) -> str:
    """Streaming SHA-256 of a file (used for external artifacts such as Teacher weights)."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def serialization_manifest() -> dict:
    """Identity of the serialization scheme itself (enters the manifest as an encoding identity)."""
    return {
        "format_version": SERIALIZATION_FORMAT_VERSION,
        "representations_per_object": 1,
        "header_encoding": "canonical JSON: sorted keys, UTF-8, separators (',', ':'), ASCII-escaped",
        "tensor_encoding": "contiguous little-endian raw dtype bytes after a length-prefixed header",
        "digest": "SHA-256 over length_prefixed_header || concatenated_payloads",
        "float_handling": "exact bit patterns; no rounding, reformatting or narrowing",
        "excluded_from_identity": ["wall-clock time", "filesystem paths", "host identifiers",
                                   "unordered iteration order"],
    }
