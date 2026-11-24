"""安全拒绝机制（SafeGate）。

赛题安全合规明确要求「须具备拒绝机制」。工业自演化系统不能为了"不断上线"
而硬猜：当输入越出训练参考量程、分类置信度不足、或输出越出物理合理区间时，
系统应**明确拒绝输出并记录证据**，而不是给出一个看似正常实则危险的预测。

三重拒答（纯 numpy，sklearn 与官方 NumPy MLP 双路径复用）：
  1. 输入越界    ：特征值超出参考期 [P1 - 3σ, P99 + 3σ] 之外的比例过高 → 拒答
  2. 置信度不足  ：分类最大 softmax 置信度低于阈值 → 拒答（"我不确定"）
  3. 输出越界    ：回归预测值越出参考输出量程的合理膨胀区间 → 拒答/钳制

设计原则：拒绝本身必须"留痕可审计"——返回 SafeReport，写进 trace，答辩时能
逐条解释"这一窗为什么拒绝了哪些样本、依据是什么"。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class SafeReport:
    refused: bool = False
    input_oob_ratio: float = 0.0
    low_conf_ratio: float = 0.0
    output_oob_ratio: float = 0.0
    reasons: List[str] = field(default_factory=list)
    mask: Optional[np.ndarray] = None  # 每样本是否被拒答（True=拒答）

    def to_dict(self) -> Dict[str, Any]:
        return {
            "refused": self.refused,
            "input_oob_ratio": round(self.input_oob_ratio, 4),
            "low_conf_ratio": round(self.low_conf_ratio, 4),
            "output_oob_ratio": round(self.output_oob_ratio, 4),
            "reasons": self.reasons,
        }


def input_range_bounds(X_ref: np.ndarray, margin: float = 3.0):
    """参考期每个特征的合理量程 [P1 - margin*σ, P99 + margin*σ]。"""
    X_ref = np.asarray(X_ref, dtype=np.float64)
    sigma = np.nanstd(X_ref, axis=0)
    sigma = np.where(sigma < 1e-9, 1.0, sigma)
    lo = np.nanpercentile(X_ref, 1, axis=0) - margin * sigma
    hi = np.nanpercentile(X_ref, 99, axis=0) + margin * sigma
    return lo, hi


def check_input_oob(X_cur: np.ndarray, X_ref: np.ndarray, margin: float = 3.0):
    """输入越界检测：返回越界样本比例与逐样本掩码。"""
    X_cur = np.asarray(X_cur, dtype=np.float64)
    lo, hi = input_range_bounds(X_ref, margin)
    outside = (X_cur < lo) | (X_cur > hi)
    sample_bad = outside.any(axis=1)
    return float(sample_bad.mean()), sample_bad


def check_confidence(proba: np.ndarray, threshold: float = 0.6):
    """分类置信度检测：最大 softmax 概率低于阈值判为"不确定"。"""
    proba = np.asarray(proba, dtype=np.float64)
    if proba.ndim == 1:
        proba = proba.reshape(-1, 1)
    conf = proba.max(axis=1) if proba.shape[1] > 1 else np.abs(proba.ravel())
    low = conf < threshold
    return float(low.mean()), low


def check_output_range(pred: np.ndarray, y_ref: np.ndarray, inflate: float = 2.0):
    """回归输出越界检测：预测值越出参考输出的合理膨胀区间。"""
    pred = np.asarray(pred, dtype=np.float64).ravel()
    y_ref = np.asarray(y_ref, dtype=np.float64).ravel()
    lo = float(np.nanmin(y_ref))
    hi = float(np.nanmax(y_ref))
    span = max(hi - lo, 1e-9)
    lo2 = lo - inflate * span
    hi2 = hi + inflate * span
    bad = (pred < lo2) | (pred > hi2)
    return float(bad.mean()), bad


class SafeGate:
    """把三重拒答统一成一个可审计门禁。"""

    def __init__(self, cfg: Optional[Dict[str, Any]] = None, task_type: str = "regression") -> None:
        c = cfg or {}
        self.task_type = task_type
        self.input_margin = float(c.get("input_margin", 3.0))
        self.conf_threshold = float(c.get("conf_threshold", 0.6))
        self.output_inflate = float(c.get("output_inflate", 2.0))
        self.input_oob_cap = float(c.get("input_oob_cap", 0.5))
        self.conf_low_cap = float(c.get("conf_low_cap", 0.3))
        self.output_oob_cap = float(c.get("output_oob_cap", 0.2))
        self.X_ref: Optional[np.ndarray] = None
        self.y_ref: Optional[np.ndarray] = None

    def set_reference(self, X_ref: np.ndarray, y_ref: Optional[np.ndarray]) -> None:
        self.X_ref = np.asarray(X_ref, dtype=np.float64)
        self.y_ref = None if y_ref is None else np.asarray(y_ref, dtype=np.float64).ravel()

    def check(
        self,
        X: np.ndarray,
        pred: np.ndarray,
        proba: Optional[np.ndarray] = None,
    ) -> SafeReport:
        rep = SafeReport()
        X = np.asarray(X, dtype=np.float64)
        pred = np.asarray(pred, dtype=np.float64)
        mask = np.zeros(len(X), dtype=bool)

        # 1) 输入越界
        if self.X_ref is not None and self.X_ref.shape[1] == X.shape[1]:
            ioob, imask = check_input_oob(X, self.X_ref, self.input_margin)
            rep.input_oob_ratio = ioob
            if ioob > self.input_oob_cap:
                rep.reasons.append(f"输入越界比例 {ioob:.1%} > 上限 {self.input_oob_cap:.0%}")
                mask |= imask

        # 2) 分类置信度不足
        if proba is not None:
            lconf, cmask = check_confidence(proba, self.conf_threshold)
            rep.low_conf_ratio = lconf
            if lconf > self.conf_low_cap:
                rep.reasons.append(f"低置信度比例 {lconf:.1%} > 上限 {self.conf_low_cap:.0%}")
                mask |= cmask

        # 3) 输出越界（仅回归：分类输出是概率分布，无"物理量程"概念）
        if self.task_type == "regression" and self.y_ref is not None and self.y_ref.size >= 5:
            ooob, omask = check_output_range(pred, self.y_ref, self.output_inflate)
            rep.output_oob_ratio = ooob
            if ooob > self.output_oob_cap:
                rep.reasons.append(f"输出越界比例 {ooob:.1%} > 上限 {self.output_oob_cap:.0%}")
                mask |= omask

        rep.refused = bool(mask.any())
        rep.mask = mask
        return rep
