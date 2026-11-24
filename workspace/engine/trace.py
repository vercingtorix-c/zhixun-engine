"""实验记录：全轨迹留痕（对应赛题《实验记录规范》）。

每条记录都是 append-only 的 JSONL，事件粒度到"每一步做什么、为什么、耗时多少"，
答辩时可以直接把某个窗口的完整决策链摊开讲。
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="milliseconds")


class Trace:
    def __init__(self, path: Path, run_id: str, meta: Optional[Dict[str, Any]] = None) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id
        self.seq = 0
        self.t0 = time.perf_counter()
        self._fh = self.path.open("a", encoding="utf-8")
        self.event(step="run", event="start", payload=meta or {})

    def event(self, step: str, event: str, window: Optional[int] = None, payload: Optional[Dict[str, Any]] = None) -> None:
        self.seq += 1
        rec = {
            "seq": self.seq,
            "ts": now_iso(),
            "elapsed_s": round(time.perf_counter() - self.t0, 3),
            "run_id": self.run_id,
            "window": window,
            "step": step,
            "event": event,
            "payload": payload or {},
        }
        self._fh.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
        self._fh.flush()

    def close(self, payload: Optional[Dict[str, Any]] = None) -> None:
        self.event(step="run", event="end", payload=payload or {})
        self._fh.close()

    def __enter__(self) -> "Trace":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if not self._fh.closed:
            if exc_type is not None:
                self.event(step="run", event="error", payload={"type": str(exc_type), "msg": str(exc)})
            self.close()
