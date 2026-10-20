"""H9*-XR1 formal code-identity freeze gate (v3 architecture).

Two defects in the v1/v2 gate are fixed here:

1. **Implicit historical-tag binding.** `freeze_gate()` had a module-level default tag, so a
   no-argument call silently validated against whatever tag that constant happened to name. That is
   how the v2 ledger freeze came to check itself against v1. The expected execution identity is now a
   REQUIRED argument. A missing value raises `FreezeGateHardStop` (a fail-closed sentinel, not a
   silent default), so no caller can validate against a historical identity by omission.

2. **Platform-dependent byte identity.** The drift check compared the raw worktree SHA256. On a
   Windows checkout with `core.autocrlf=true` the worktree is CRLF while git stores LF-normalised
   blobs, so the same content produced two different hashes and a legitimate frozen identity could
   false-fail on another platform. The authoritative check is now the **Git-normalised blob
   identity**: for every covered file the worktree's cleaned blob, the staged (index) blob, the blob
   recorded in the closure and the blob stored in the tagged commit must all be the same object.
   Semantic changes always change the blob; CRLF/LF checkout normalisation alone never does. Raw
   worktree SHA256 is still recorded and reported, but it is **informational only**.

AN EXECUTION IDENTITY MUST BE SUPPLIED EXPLICITLY. Example:

    freeze_gate(expected_tag="h9-xr1-execution-frozen-v3")
"""
import hashlib
import json
import subprocess
from pathlib import Path

EXECUTION_MANIFEST = Path("manifests/h9_xr1_execution_hashes.json")

#: sentinel: distinguishes "caller supplied nothing" from any real tag name
_UNBOUND = object()
#: the identity convention this gate enforces, recorded verbatim in the v3 closure
IDENTITY_CONVENTION = (
    "Git-normalised blob identity. For every covered file the worktree's cleaned blob "
    "(git hash-object), the staged index blob, the blob recorded in the execution closure and the "
    "blob stored in the tagged commit must be the same object. Raw worktree SHA256 is informational "
    "only, because it depends on the checkout's line-ending configuration.")
#: file modes / statuses that indicate a covered file is not a plain tracked blob
_TRACKED_MODE_PREFIX = "100"


class FreezeGateHardStop(RuntimeError):
    """Raised when the running code is not the frozen scientific code identity."""


def sha_file(path):
    """Raw worktree SHA256. INFORMATIONAL ONLY — see IDENTITY_CONVENTION."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(*args):
    completed = subprocess.run(("git", "-c", f"safe.directory={Path.cwd().as_posix()}", *args),
                               capture_output=True, text=True)
    return (completed.stdout.strip() if completed.returncode == 0 else ""), completed.returncode


def git(*args):
    return _git(*args)[0]


def worktree_blob(path):
    """The path's Git-cleaned blob id: applies the repository's filters, so CRLF worktree content
    and LF worktree content with the same logical text produce the SAME id."""
    return _git("hash-object", "--path", str(path), str(path))[0]


def index_blob(path):
    """The staged blob id from the index, or '' when the path is not staged."""
    out, code = _git("ls-files", "-s", "--", str(path))
    if code != 0 or not out:
        return ""
    fields = out.split()
    return fields[1] if len(fields) >= 2 else ""


def is_tracked(path):
    out, code = _git("ls-files", "-s", "--", str(path))
    return code == 0 and bool(out) and out.split()[0].startswith(_TRACKED_MODE_PREFIX)


def tree_blob(rev, path):
    """The blob id stored for `path` in `rev`.

    The path is converted to POSIX separators: the `rev:path` form is a Git tree path, not a local
    filesystem path, and a Windows-style backslash makes `rev-parse` fail silently (returning nothing),
    which would look like a missing file rather than a path-form error.
    """
    return _git("rev-parse", f"{rev}:{Path(path).as_posix()}")[0]


def freeze_gate(expected_tag=_UNBOUND, manifest_path=EXECUTION_MANIFEST):
    """Verify the running code is the named frozen XR1 scientific code identity. Fail closed.

    `expected_tag` is REQUIRED. There is no default execution identity: omitting it is an error, not
    an implicit check against a historical tag.
    """
    if expected_tag is _UNBOUND or expected_tag is None or not str(expected_tag).strip():
        raise FreezeGateHardStop(
            "no expected execution identity supplied: freeze_gate() now requires an explicit "
            "expected_tag (for example expected_tag='h9-xr1-execution-frozen-v3'). There is no "
            "default tag, because a default would let a formal command silently validate against a "
            "historical identity — the defect that blocked the v2 ledger freeze.")
    expected_tag = str(expected_tag).strip()

    manifest_path = Path(manifest_path)
    report = {"manifest": str(manifest_path), "expected_tag": expected_tag,
              "identity_convention": IDENTITY_CONVENTION, "checks": {}, "failures": [],
              "informational": {}}

    def check(name, ok, detail=""):
        report["checks"][name] = bool(ok)
        if not ok:
            report["failures"].append(f"{name}: {detail}" if detail else name)
        return bool(ok)

    if not manifest_path.exists():
        raise FreezeGateHardStop(f"execution manifest absent: {manifest_path}")
    report["informational"]["manifest_raw_sha256"] = sha_file(manifest_path)

    closure = json.loads(manifest_path.read_text(encoding="utf-8"))
    check("manifest_schema", closure.get("schema") == "latentbridge.h9_xr1.execution_hashes.v1")
    check("manifest_status_frozen", closure.get("status") == "EXECUTION_IDENTITY_FROZEN")

    resolved, code = _git("rev-parse", "--verify", "--quiet", f"{expected_tag}^{{commit}}")
    check("expected_tag_resolves", bool(resolved) and code == 0,
          f"{expected_tag} does not resolve")
    report["expected_tag_commit"] = resolved or None
    if not resolved:
        report["passed"] = False
        raise FreezeGateHardStop(
            f"XR1 freeze gate FAILED — expected execution identity {expected_tag} does not resolve: "
            + "; ".join(report["failures"]))

    # the manifest itself must be the blob stored in the named commit
    check("manifest_matches_tagged_blob",
          worktree_blob(manifest_path) == tree_blob(expected_tag, manifest_path),
          f"worktree {worktree_blob(manifest_path)[:12]} != tag "
          f"{tree_blob(expected_tag, manifest_path)[:12]}")

    covered = closure.get("execution_code") or {}
    check("closure_non_empty", len(covered) > 0)
    missing, untracked, worktree_drift, index_drift, tagged_drift = [], [], [], [], []
    raw_mismatch = []
    for path, entry in sorted(covered.items()):
        full = Path(path)
        if not full.exists():
            missing.append(path)
            continue
        if not is_tracked(path):
            untracked.append(path)
            continue
        want = entry.get("git_blob")
        wt, idx, tagged = worktree_blob(path), index_blob(path), tree_blob(expected_tag, path)
        if wt != want:
            worktree_drift.append(path)
        if idx and idx != want:
            index_drift.append(path)
        if tagged != want:
            tagged_drift.append(path)
        # informational only: a CRLF worktree legitimately differs here
        if entry.get("sha256") and sha_file(path) != entry["sha256"]:
            raw_mismatch.append(path)
    check("no_covered_file_missing", not missing, str(missing[:5]))
    check("all_covered_files_tracked", not untracked, str(untracked[:5]))
    check("no_worktree_blob_drift", not worktree_drift, str(worktree_drift[:5]))
    check("no_staged_blob_drift", not index_drift, str(index_drift[:5]))
    check("no_tagged_blob_drift", not tagged_drift, str(tagged_drift[:5]))
    check("closure_has_no_missing_entries", not (closure.get("missing_execution_code") or []))

    report["covered_files"] = len(covered)
    report["informational"]["raw_sha256_differing_files"] = raw_mismatch
    report["informational"]["raw_sha256_note"] = (
        "files listed here have a raw worktree SHA256 differing from the recorded value purely "
        "because of line-ending encoding; the authoritative blob identity matched. Informational only."
        if raw_mismatch else "no raw worktree SHA256 differences")
    report["passed"] = not report["failures"]
    if not report["passed"]:
        raise FreezeGateHardStop(
            "XR1 freeze gate FAILED — the running code is not the frozen scientific code identity: "
            + "; ".join(report["failures"][:6]))
    return report


#: every formal XR1 call site must bind this identity explicitly
FORMAL_EXECUTION_TAG_V3 = "h9-xr1-execution-frozen-v3"
