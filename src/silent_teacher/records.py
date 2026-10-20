import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path
import torch


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def state_hash(model):
    h = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        h.update(name.encode())
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def command(*args):
    if args[0] == "git":
        args = ("git", "-c", f"safe.directory={Path.cwd().as_posix()}", *args[1:])
    p = subprocess.run(args, capture_output=True, text=True)
    return p.stdout.strip() if p.returncode == 0 else p.stderr.strip()


def environment():
    return {
        "python": sys.version, "platform": platform.platform(),
        "packages": command(sys.executable, "-m", "pip", "freeze"),
        "git_commit": command("git", "rev-parse", "HEAD"),
        "git_status": command("git", "status", "--porcelain"),
        "git_diff": command("git", "diff", "HEAD"),
        "source_hashes": {str(p): file_hash(p) for base in ("src", "scripts", "tests", "configs", "protocols") for p in sorted(Path(base).rglob("*")) if p.is_file() and "__pycache__" not in p.parts},
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
        "driver": command("nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"),
        "tf32": False, "amp": False, "deterministic": True,
    }
