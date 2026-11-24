"""第 5 步：封版（不可变版本快照 + 哈希链 + 可复现清单）。

赛题的硬门槛之一是"记录完整、可复现"。所以每一次上线都不覆盖旧模型：
  versions/v0001/… 一旦写入就不再修改，manifest 里带 prev_hash，
  形成一条可逐版校验的哈希链（任何一版被篡改，链就断）。
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from .gate import Version


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _force_writable(p: Path) -> None:
    """上次运行可能已把版本文件只读化；重跑同一目录时先恢复可写。"""
    if p.exists():
        try:
            os.chmod(p, 0o666)
        except Exception:
            pass


def env_fingerprint() -> Dict[str, Any]:
    out = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
    }
    try:
        import sklearn

        out["sklearn"] = sklearn.__version__
    except Exception:
        out["sklearn"] = None
    return out


def data_fingerprint(windows) -> str:
    """对数据流做指纹：窗口数量、形状、每窗首行的哈希。用于复现校验。"""
    parts = []
    for w in windows:
        arr = np.asarray(w.X_train[: min(5, len(w.X_train))], dtype=np.float64)
        parts.append(f"{w.name}:{w.X_train.shape}:{_sha256_bytes(np.nan_to_num(arr).tobytes())}")
    return _sha256_bytes("|".join(parts).encode("utf-8"))


class Sealer:
    def __init__(self, run_dir: Path) -> None:
        self.run_dir = Path(run_dir)
        self.versions_dir = self.run_dir / "versions"
        self.versions_dir.mkdir(parents=True, exist_ok=True)
        self.chain_path = self.run_dir / "version_chain.json"
        self.chain: list = []
        if self.chain_path.exists():
            try:
                self.chain = json.loads(self.chain_path.read_text(encoding="utf-8"))
            except Exception:
                self.chain = []

    def _prev_hash(self) -> str:
        return self.chain[-1]["chain_hash"] if self.chain else "0" * 64

    def seal(
        self,
        version: Version,
        task: Optional[Any] = None,
        windows: Optional[list] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        vdir = self.versions_dir / f"v{version.vid:04d}"
        vdir.mkdir(parents=True, exist_ok=True)

        model_path = vdir / "model.pkl"
        mpath = vdir / "manifest.json"
        _force_writable(model_path)
        _force_writable(mpath)
        _force_writable(vdir / "config.json")
        with model_path.open("wb") as f:
            pickle.dump({"cfg": version.cfg, "model": version.model}, f, protocol=4)
        model_hash = _sha256_file(model_path)

        cfg = json.dumps(version.cfg, ensure_ascii=False, sort_keys=True)
        prev = self._prev_hash()
        chain_hash = _sha256_bytes(f"{prev}|{cfg}|{model_hash}".encode("utf-8"))

        manifest = {
            "version": version.vid,
            "created_at": version.created_at or _now(),
            "note": version.note,
            "prev_hash": prev,
            "chain_hash": chain_hash,
            "model_sha256": model_hash,
            "config": version.cfg,
            "metrics": version.metrics,
            "env": env_fingerprint(),
            "data_fingerprint": data_fingerprint(windows) if windows else None,
            "task": {
                "name": getattr(task, "name", None),
                "task_type": getattr(task, "task_type", None),
                "scale": getattr(task, "scale", None),
            },
            "extra": extra or {},
        }
        mpath = vdir / "manifest.json"
        mpath.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        (vdir / "config.json").write_text(json.dumps(version.cfg, ensure_ascii=False, indent=2), encoding="utf-8")

        # 只读化：防止运行中误改历史版本（manifest 最后设置，方便调试时先可写）
        try:
            os.chmod(mpath, 0o444)
            os.chmod(model_path, 0o444)
        except Exception:
            pass

        self.chain.append(
            {
                "version": version.vid,
                "created_at": manifest["created_at"],
                "chain_hash": chain_hash,
                "prev_hash": prev,
                "model_sha256": model_hash,
            }
        )
        self.chain_path.write_text(json.dumps(self.chain, ensure_ascii=False, indent=2), encoding="utf-8")
        version.sealed = True
        return manifest

    @staticmethod
    def verify_chain(chain: list) -> Dict[str, Any]:
        """逐版校验哈希链完整性，输出给审计/答辩用。"""
        broken = []
        prev = "0" * 64
        for item in chain:
            if item.get("prev_hash") != prev:
                broken.append(item.get("version"))
            prev = item.get("chain_hash", "")
        return {"n_versions": len(chain), "broken_versions": broken, "intact": not broken}

    def write_reproduce(self, run_id: str, cmd: str, seed: int, dataset: str) -> Path:
        p = self.run_dir / "reproduce.md"
        p.write_text(
            "# 复现指引\n\n"
            f"- run_id: `{run_id}`\n"
            f"- dataset: `{dataset}`\n"
            f"- seed: `{seed}`\n"
            "- 环境: 见 `versions/*/manifest.json` 中的 env 字段\n\n"
            "## 一键复现\n\n```bash\n" + cmd + "\n```\n\n"
            "## 产物校验\n\n"
            "- `versions/vNNNN/manifest.json` 的 `chain_hash` 应与 `version_chain.json` 一致\n"
            "- `data_fingerprint` 用于确认数据流未被替换\n"
            "- 每个 trial 的随机种子记录在 `trace.jsonl` 的 `trial` 事件中\n",
            encoding="utf-8",
        )
        return p
