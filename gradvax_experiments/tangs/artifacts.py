"""Auditable run artifacts and immutable configuration handling."""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from .config import CDMAD_COMMIT, CONFIG_VERSION, RunConfig


def _json_default(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot JSON-encode {type(value).__name__}")
def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        if math.isnan(value):
            return "nan"
        return "inf" if value > 0 else "-inf"
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        return _json_safe(value.item())
    return value



def canonical_json(value: Any) -> str:
    return json.dumps(
        _json_safe(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        default=_json_default,
        allow_nan=False,
    )


def scientific_config(config: RunConfig) -> dict[str, Any]:
    value = config.to_dict()
    for field in ("data_root", "output_root", "resume"):
        value.pop(field, None)
    return value


def config_hash(config: RunConfig) -> str:
    return hashlib.sha256(canonical_json(scientific_config(config)).encode()).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(
            _json_safe(value),
            handle,
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            default=_json_default,
            allow_nan=False,
        )
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def append_jsonl(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(canonical_json(value))
        handle.write("\n")
        handle.flush()


def _git_value(repo: Path, arguments: list[str]) -> str | None:
    try:
        result = subprocess.run(
            ["git", *arguments],
            cwd=repo,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def environment_manifest(project_root: Path) -> dict[str, Any]:
    try:
        import torch
        import torchvision

        torch_version = torch.__version__
        torchvision_version = torchvision.__version__
        cuda_version = torch.version.cuda
        cudnn_version = torch.backends.cudnn.version()
        gpu_names = [
            torch.cuda.get_device_name(index)
            for index in range(torch.cuda.device_count())
        ]
    except ImportError:
        torch_version = None
        torchvision_version = None
        cuda_version = None
        cudnn_version = None
        gpu_names = []

    status = _git_value(project_root, ["status", "--short"])
    return {
        "config_version": CONFIG_VERSION,
        "pinned_cdmad_commit": CDMAD_COMMIT,
        "created_unix": time.time(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "executable": sys.executable,
        "command": sys.argv,
        "torch": torch_version,
        "torchvision": torchvision_version,
        "cuda": cuda_version,
        "cudnn": cudnn_version,
        "gpu_names": gpu_names,
        "git_commit": _git_value(project_root, ["rev-parse", "HEAD"]),
        "git_dirty": bool(status),
        "git_status": status,
    }


def prepare_run(
    config: RunConfig,
    split_manifest: dict[str, Any],
    project_root: Path,
) -> tuple[Path, str]:
    run_dir = Path(config.output_root).expanduser().resolve() / config.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    digest = config_hash(config)
    config_path = run_dir / "resolved_config.json"
    if config_path.exists():
        with config_path.open(encoding="utf-8") as handle:
            previous = json.load(handle)
        if previous.get("config_hash") != digest:
            raise RuntimeError(
                f"Run directory {run_dir} belongs to another configuration."
            )
        if not config.resume:
            raise RuntimeError(
                f"Run directory {run_dir} already exists. Use --resume auto or a new --run-id."
            )
    else:
        atomic_json(
            config_path,
            {
                "config_hash": digest,
                "scientific_config": scientific_config(config),
                "runtime_paths": {
                    "data_root": config.data_root,
                    "output_root": config.output_root,
                },
            },
        )
        atomic_json(run_dir / "split_manifest.json", split_manifest)
        manifest = environment_manifest(project_root)
        manifest["config_hash"] = digest
        manifest["split_hash"] = split_manifest["split_hash"]
        manifest["partition_hash"] = split_manifest["partition_hash"]
        atomic_json(run_dir / "run_manifest.json", manifest)
    return run_dir, digest
