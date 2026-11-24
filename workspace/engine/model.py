"""模型层：搜索空间定义、配置白名单校验、训练/推理与安全兜底。"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .types import TaskSpec

try:
    from sklearn.ensemble import (
        HistGradientBoostingClassifier,
        HistGradientBoostingRegressor,
        RandomForestClassifier,
        RandomForestRegressor,
    )

    HAS_SKLEARN = True
except Exception:  # pragma: no cover
    HAS_SKLEARN = False


# --- 搜索空间（可解释、有界、按族展开）-------------------------------------
CONFIG_SPACE: Dict[str, Dict[str, Tuple[float, float, str]]] = {
    "hgb": {
        "learning_rate": (0.03, 0.20, "float"),
        "max_leaf_nodes": (7, 63, "int"),
        "max_iter": (40, 150, "int"),
        "min_samples_leaf": (5, 40, "int"),
        "l2_regularization": (0.0, 3.0, "float"),
    },
    "rf": {
        "n_estimators": (60, 150, "int"),
        "max_depth": (4, 14, "int"),
        "min_samples_leaf": (2, 30, "int"),
        "max_features": (0.2, 1.0, "float"),
    },
}

SAFETY_LIMITS = {
    "max_trials_per_window": 60,
    "max_gpu_minutes_per_window": 90.0,
    "max_total_minutes": 600.0,
}


def default_config(task: TaskSpec) -> Dict[str, Any]:
    """在位者初始配置（也是"静态基线"用的配置）。"""
    return {
        "family": "hgb",
        "params": {
            "learning_rate": 0.08,
            "max_leaf_nodes": 15,
            "max_iter": 80,
            "min_samples_leaf": 20,
            "l2_regularization": 1.0,
        },
        "recency_half_life": 4.0,
        "drop_features": [],
        "version": 0,
    }


def validate_config(cfg: Dict[str, Any]) -> Tuple[bool, str]:
    """安全白名单：只接受定义过的族与有界超参，防止优化器把系统带出可控范围。"""
    fam = cfg.get("family")
    if fam not in CONFIG_SPACE:
        return False, f"未授权的模型族: {fam}"
    params = cfg.get("params", {})
    for k, v in params.items():
        if k not in CONFIG_SPACE[fam]:
            return False, f"未授权的超参: {k}"
        lo, hi, kind = CONFIG_SPACE[fam][k]
        try:
            fv = float(v)
        except Exception:
            return False, f"超参 {k} 非数值: {v}"
        if not np.isfinite(fv):
            return False, f"超参 {k} 非有限值"
        if fv < lo or fv > hi:
            return False, f"超参 {k}={fv} 超出安全区间 [{lo}, {hi}]"
        if kind == "int" and abs(fv - round(fv)) > 1e-9:
            return False, f"超参 {k} 必须为整数"
    hlf = cfg.get("recency_half_life")
    if hlf is not None and (not np.isfinite(float(hlf)) or float(hlf) <= 0):
        return False, f"recency_half_life 非法: {hlf}"
    return True, "ok"


def _clean_params(fam: str, params: Dict[str, Any]) -> Dict[str, Any]:
    out = {}
    for k, v in params.items():
        lo, hi, kind = CONFIG_SPACE[fam][k]
        fv = min(max(float(v), lo), hi)
        out[k] = int(round(fv)) if kind == "int" else float(fv)
    return out


def build_estimator(cfg: Dict[str, Any], task: TaskSpec, seed: int = 0):
    fam = cfg["family"]
    params = _clean_params(fam, cfg.get("params", {}))
    if not HAS_SKLEARN:  # pragma: no cover - 纯兜底
        from sklearn.linear_model import Ridge

        return Ridge(alpha=1.0)

    is_reg = task.task_type == "regression"
    if fam == "hgb":
        cls = HistGradientBoostingRegressor if is_reg else HistGradientBoostingClassifier
        kw = dict(params)
        if not is_reg:
            kw["max_iter"] = max(30, kw.get("max_iter", 120))
        return cls(random_state=seed, early_stopping=False, **kw)
    if fam == "rf":
        cls = RandomForestRegressor if is_reg else RandomForestClassifier
        return cls(random_state=seed, n_jobs=1, **params)
    raise ValueError(f"未知模型族 {fam}")


def _apply_feature_mask(X: np.ndarray, drop: List[int]) -> np.ndarray:
    if not drop:
        return X
    keep = [i for i in range(X.shape[1]) if i not in set(drop)]
    return X[:, keep]


def fit(
    cfg: Dict[str, Any],
    task: TaskSpec,
    X: np.ndarray,
    y: np.ndarray,
    sample_weight: Optional[np.ndarray] = None,
    seed: int = 0,
) -> Any:
    X = _apply_feature_mask(X, cfg.get("drop_features", []))
    est = build_estimator(cfg, task, seed=seed)
    if sample_weight is not None and HAS_SKLEARN:
        try:
            est.fit(X, y, sample_weight=sample_weight)
            return est
        except TypeError:
            pass
    est.fit(X, y)
    return est


def predict(cfg: Dict[str, Any], model: Any, X: np.ndarray, task: TaskSpec) -> np.ndarray:
    X = _apply_feature_mask(X, cfg.get("drop_features", []))
    pred = np.asarray(model.predict(X)).ravel()
    return pred


def safe_predict(
    cfg: Dict[str, Any], model: Any, X: np.ndarray, task: TaskSpec
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """拒绝机制：一旦出现 NaN/Inf 或越界输出，返回兜底预测并记录拒答事件。"""
    report = {"refused": False, "reason": "", "nan_ratio": 0.0}
    try:
        pred = predict(cfg, model, X, task)
    except Exception as e:  # 崩溃也要兜住，不能让闭环挂掉
        report["refused"] = True
        report["reason"] = f"推理异常: {e}"
        return np.zeros(len(X)), report

    bad = ~np.isfinite(pred)
    report["nan_ratio"] = float(bad.mean())
    if bad.any():
        report["refused"] = True
        report["reason"] = f"输出非有限值比例 {report['nan_ratio']:.3f}，已用中位数兜底"
        good = pred[~bad]
        fill = float(np.median(good)) if good.size else 0.0
        pred = np.where(bad, fill, pred)
    return pred, report


def time_budget_guard(minutes_used: float, minutes_cap: float) -> bool:
    return minutes_used < minutes_cap


class Latency:
    """推理延迟采样，供上线门槛判断。"""

    def __init__(self) -> None:
        self.samples: List[float] = []

    def measure(self, cfg, model, X, task) -> float:
        t0 = time.perf_counter()
        predict(cfg, model, X, task)
        dt = (time.perf_counter() - t0) * 1000.0
        self.samples.append(dt)
        return dt

    def p95(self) -> float:
        if not self.samples:
            return 0.0
        return float(np.percentile(self.samples, 95))
