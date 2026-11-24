"""统一数据契约。

赛题要求：在"漂移数据流"上无人干预地自我演化。为了让 UCI Gas（传感器漂移）、
CMAPSS（发动机退化）、XJTU-SY（轴承全寿命）三类异构工业数据能走同一套闭环，
这里把一切都抽象成「按时间到达的数据窗（Window）」。

一个 Window 里有两份数据：
  - X_train / y_train : 已到货、带标签，允许用来训练（在线可得的监督信号）
  - X_eval  / y_eval  : 评测窗。y_eval 在真实评测中是隐藏的，
                        引擎内部任何决策都不得读取它（见 loop.py 的 use_eval_labels 开关）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np


@dataclass
class TaskSpec:
    """任务定义：决定打分口径、指标方向、以及"性能"如何归一化。"""

    name: str
    task_type: str  # "regression" | "classification"
    n_classes: int = 0
    # 归一化基线：回归用 y 的标准差，分类用 1/类别数（随机猜的水平）
    scale: float = 1.0

    def score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """分数越高越好。回归用归一化后的 1-NRMSE（近似 R²），分类用 macro-F1。"""
        y_true = np.asarray(y_true).ravel()
        y_pred = np.asarray(y_pred).ravel()
        if len(y_true) == 0:
            return float("nan")
        if self.task_type == "regression":
            rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
            # 有界得分 ∈ (0,1]：RMSE=0 得 1，RMSE=scale 得 0.5，越差越趋近 0。
            # 相比"1-RMSE/scale"，漂移窗的极端误差不会把得分打到很负，便于横向比较。
            return float(self.scale / (self.scale + rmse)) if self.scale > 1e-9 else float(rmse == 0)
        # 分类：macro-F1（不依赖 sklearn，避免评测环境依赖差异）
        labels = np.unique(np.concatenate([y_true, y_pred]))
        f1s = []
        for c in labels:
            tp = float(np.sum((y_pred == c) & (y_true == c)))
            fp = float(np.sum((y_pred == c) & (y_true != c)))
            fn = float(np.sum((y_pred != c) & (y_true == c)))
            if tp + fp + fn == 0:
                continue
            prec = tp / (tp + fp) if tp + fp > 0 else 0.0
            rec = tp / (tp + fn) if tp + fn > 0 else 0.0
            f1s.append(0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec))
        return float(np.mean(f1s)) if f1s else 0.0

    def loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """越小越好，供优化器内部使用。"""
        if self.task_type == "regression":
            y_true = np.asarray(y_true).ravel()
            y_pred = np.asarray(y_pred).ravel()
            return float(np.sqrt(np.mean((y_true - y_pred) ** 2))) / max(self.scale, 1e-9)
        return float(1.0 - self.score(y_true, y_pred))


@dataclass
class Window:
    index: int
    name: str
    X_train: np.ndarray
    y_train: np.ndarray
    X_eval: np.ndarray
    y_eval: Optional[np.ndarray] = None
    # 该窗的真实漂移类型，仅用于事后归因打分（引擎不得读取）
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def has_eval_labels(self) -> bool:
        return self.y_eval is not None

    def summary(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "name": self.name,
            "n_train": int(len(self.X_train)),
            "n_eval": int(len(self.X_eval)),
            "n_features": int(self.X_train.shape[1]),
            "has_eval_labels": self.has_eval_labels,
        }


def make_task_from_window(w: Window, name: str) -> TaskSpec:
    """按首个窗口的标签分布推断回归/分类并定标。"""
    y = np.asarray(w.y_train).ravel()
    uniq = np.unique(y)
    is_int_like = np.all(np.abs(y - np.round(y)) < 1e-9)
    if len(uniq) <= 20 and is_int_like and len(uniq) > 1:
        n_cls = int(len(uniq))
        return TaskSpec(name=name, task_type="classification", n_classes=n_cls, scale=1.0 / n_cls)
    return TaskSpec(
        name=name,
        task_type="regression",
        scale=float(np.std(y)) if float(np.std(y)) > 1e-9 else 1.0,
    )
