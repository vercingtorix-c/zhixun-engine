"""第 4 步：评估上线（挑战者 vs 在位者）。

工业现场最怕"为了指标好看频繁换模型"。上线门槛必须同时管住四件事：
  1. 增益：挑战者要在多个验证子集上稳定超过在位者，而不是只赢一个窗；
  2. 稳定性：多次重抽样的得分方差不能大（避免"这次运气好"）；
  3. 抗干扰：注入噪声后性能不能崩（应对隐藏测试集里的脏数据/空跑包）；
  4. 成本与安全：推理延迟、非有限输出比例都在预算内。

四项全过才 promote；否则 reject 并保留在位者，连续失败则 rollback 到上一个封版。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from . import model as model_lib
from .types import TaskSpec


@dataclass
class Version:
    vid: int
    cfg: Dict[str, Any]
    model: Any = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    created_at: str = ""
    sealed: bool = False
    note: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.vid,
            "created_at": self.created_at,
            "sealed": self.sealed,
            "note": self.note,
            "metrics": self.metrics,
            "config": {
                "family": self.cfg.get("family"),
                "params": self.cfg.get("params", {}),
                "recency_half_life": self.cfg.get("recency_half_life"),
                "drop_features": self.cfg.get("drop_features", []),
            },
        }


@dataclass
class GateConfig:
    min_gain: float = 0.004          # 至少提升 0.4 个百分点才值得换模型
    max_std: float = 0.025           # 重抽样得分标准差上限
    max_noise_drop: float = 0.10     # 输入噪声下的允许性能跌幅
    max_latency_ms: float = 120.0
    max_refuse_ratio: float = 0.01
    bootstrap: int = 3


def _score_on(task: TaskSpec, cfg, model, X, y, idx=None) -> float:
    if idx is not None:
        X, y = X[idx], y[idx]
    pred, rep = model_lib.safe_predict(cfg, model, X, task)
    if rep["refused"] and rep["nan_ratio"] > 0.5:
        return -1.0
    return float(task.score(y, pred))


def evaluate_pair(
    task: TaskSpec,
    cand: Version,
    incumbent: Version,
    val_sets: List[Tuple[str, np.ndarray, np.ndarray]],
    gc: GateConfig,
    rng: np.random.Generator,
) -> Dict[str, Any]:
    """在多个验证子集上比较两个版本，输出可审计的门槛判定明细。"""
    per_set: List[Dict[str, Any]] = []
    for name, X, y in val_sets:
        c = _score_on(task, cand.cfg, cand.model, X, y)
        i = _score_on(task, incumbent.cfg, incumbent.model, X, y)
        per_set.append({"set": name, "challenger": round(c, 4), "incumbent": round(i, 4), "delta": round(c - i, 4)})

    deltas = [d["delta"] for d in per_set]
    gains = [d["challenger"] for d in per_set]
    comp_delta = float(np.mean(deltas))
    comp_score = float(np.mean(gains))

    # 稳定性：在合并验证集上做 bootstrap 重抽样
    Xall = np.vstack([v[1] for v in val_sets])
    yall = np.concatenate([v[2] for v in val_sets])
    boots = []
    for _ in range(gc.bootstrap):
        idx = rng.integers(0, len(yall), size=len(yall))
        boots.append(_score_on(task, cand.cfg, cand.model, Xall, yall, idx=idx))
    std = float(np.std(boots))

    # 抗干扰：注入 2% 噪声后，得分跌幅
    scale = float(np.nanstd(Xall)) if np.nanstd(Xall) > 1e-9 else 1.0
    Xn = Xall + rng.normal(0, 0.02 * scale, size=Xall.shape)
    noise_score = _score_on(task, cand.cfg, cand.model, Xn, yall)
    noise_drop = float(comp_score - noise_score)

    lat = model_lib.Latency()
    for _ in range(3):
        lat.measure(cand.cfg, cand.model, Xall[: min(400, len(Xall))], task)
    p95 = lat.p95()

    # 非有限输出比例
    pred, rep = model_lib.safe_predict(cand.cfg, cand.model, Xall, task)
    refuse_ratio = float(rep["nan_ratio"])

    checks = {
        "gain": comp_delta >= gc.min_gain,
        "stability": std <= gc.max_std,
        "robustness": noise_drop <= gc.max_noise_drop,
        "latency": p95 <= gc.max_latency_ms,
        "safety": refuse_ratio <= gc.max_refuse_ratio,
    }
    reasons = []
    if not checks["gain"]:
        reasons.append(f"增益 {comp_delta:+.4f} < 门槛 {gc.min_gain:+.4f}")
    if not checks["stability"]:
        reasons.append(f"重抽样标准差 {std:.4f} > 上限 {gc.max_std}")
    if not checks["robustness"]:
        reasons.append(f"噪声下跌幅 {noise_drop:.4f} > 上限 {gc.max_noise_drop}")
    if not checks["latency"]:
        reasons.append(f"推理 P95 {p95:.1f}ms > 预算 {gc.max_latency_ms}ms")
    if not checks["safety"]:
        reasons.append(f"非有限输出比例 {refuse_ratio:.4f} > 上限 {gc.max_refuse_ratio}")

    decision = "promote" if all(checks.values()) else "reject"
    return {
        "decision": decision,
        "composite_delta": round(comp_delta, 4),
        "composite_score": round(comp_score, 4),
        "per_set": per_set,
        "std": round(std, 4),
        "noise_drop": round(noise_drop, 4),
        "latency_p95_ms": round(p95, 2),
        "refuse_ratio": round(refuse_ratio, 4),
        "checks": checks,
        "reasons": reasons or ["全部门槛通过"],
    }
