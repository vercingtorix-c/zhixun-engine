"""官方数据包 JSONL → 漂移数据窗。

三个官方任务的字段、目标与指标已在 A/B/C 的 TASK.md 与 configs/*.json 中明确，
这里把 public/train.jsonl 按"整单元"切分成按时间到达的窗口：
  - A 气体识别：按 source_batch（1..10）切
  - B 轴承寿命：按 bearing 整根切，内部按 snapshot 时序
  - C 发动机寿命：按 (source_subset, unit) 整台切，内部按 cycle 时序

严格遵循官方约束：同一单元（批次/轴承/发动机）不跨窗口，窗口内部不随机打散时序。
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from . import data as od


PROFILES: Dict[str, Dict[str, Any]] = {
    "A": {
        "task": "classification",
        "classes": ["Ethanol", "Ethylene", "Ammonia", "Acetaldehyde", "Acetone", "Toluene"],
        "unit": "source_batch",
        "hidden": 32,
    },
    "B": {
        "task": "regression",
        "classes": [],
        "unit": "bearing",
        "hidden": 24,
    },
    "C": {
        "task": "regression",
        "classes": [],
        "unit": ("source_subset", "unit"),
        "hidden": 32,
    },
}


def _natural_key(s: str) -> List[Any]:
    """把 Bearing1_1 这类字符串拆成可排序的 (数字, 数字) 序列。"""
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s)]


def _unit_key(profile: str, meta: Dict[str, Any]) -> Tuple[Any, ...]:
    spec = PROFILES[profile]
    key = spec["unit"]
    if isinstance(key, str):
        return (key, meta.get(key))
    return tuple(meta.get(k) for k in key)


def _unit_sort_key(profile: str, unit: Tuple[Any, ...], meta: Dict[str, Any]) -> Tuple[Any, ...]:
    """把单元标识转成稳定排序键（B 按 Bearing1_1 数字序，C 按 FD 子集序+引擎号）。"""
    out = []
    for v in unit:
        if isinstance(v, str) and re.match(r"^(Bearing|FD)\d", v):
            out.append(_natural_key(v))
        elif isinstance(v, (int, float)):
            out.append(v)
        else:
            out.append(str(v))
    return tuple(out)


def _within_order(profile: str, meta: Dict[str, Any]) -> Tuple[Any, ...]:
    """单元内部的时序键：A 用 source_index，B 用 snapshot，C 用 cycle。"""
    if profile == "A":
        return (meta.get("source_index", 0),)
    if profile == "B":
        return (meta.get("snapshot", 0),)
    if profile == "C":
        return (meta.get("cycle", 0),)
    return (0,)


def read_rows(path: Path) -> List[Dict[str, Any]]:
    rows = []
    seen = set()
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict) or not {"id", "inputs", "answer"} <= set(row):
            raise ValueError(f"{path.name}:{number}: id/inputs/answer required")
        if row["id"] in seen:
            raise ValueError(f"{path.name}:{number}: duplicate id {row['id']}")
        seen.add(row["id"])
        rows.append(row)
    if not rows:
        raise ValueError(f"{path.name}: empty dataset")
    return rows


def build_windows(profile: str, train_path: Path, n_windows: int = 10) -> List[Dict[str, Any]]:
    """把 public/train.jsonl 切成 n_windows 个按时间顺序到达的窗口。

    返回每个窗口：
      {index, name, X, y, unit_ids, meta}
    其中 X/y 是已按单元内时序排列、且跨单元顺序排列的完整矩阵；循环层再自行切 fit/val。
    """
    if profile not in PROFILES:
        raise ValueError(f"unknown profile {profile}")
    spec = PROFILES[profile]
    classes = spec["classes"]
    rows = read_rows(train_path)

    # 按单元分组
    units: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = {}
    order_hint: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
    for r in rows:
        meta = r.get("meta", {})
        uk = _unit_key(profile, meta)
        units.setdefault(uk, []).append(r)
        order_hint.setdefault(uk, meta)

    # 单元排序（时序）
    sorted_units = sorted(units.keys(), key=lambda u: _unit_sort_key(profile, u, order_hint[u]))

    # 每个单元内部按时序排序，并 flatten 输入 + 映射目标
    unit_matrices = []
    for uk in sorted_units:
        ur = sorted(units[uk], key=lambda r: _within_order(profile, r.get("meta", {})))
        paths = None
        X_rows, y_rows = [], []
        for r in ur:
            p, v = od.flatten(r["inputs"])
            if paths is None:
                paths = p
            elif p != paths:
                raise ValueError(f"feature schema changed in unit {uk}")
            X_rows.append(v)
            ans = r["answer"]
            if profile == "A":
                if ans not in classes:
                    raise ValueError(f"unknown class {ans}")
                y_rows.append(classes.index(ans))
            else:
                y_rows.append(float(ans))
        unit_matrices.append((uk, np.asarray(X_rows, dtype=np.float64),
                              np.asarray(y_rows, dtype=np.float64 if profile != "A" else np.int64)))

    # 把单元按时序 chunk 成 n_windows 个窗口
    n_units = len(unit_matrices)
    if n_windows > n_units:
        n_windows = n_units
    if n_windows < 1:
        n_windows = 1
    bounds = np.linspace(0, n_units, n_windows + 1).astype(int)
    bounds[-1] = n_units

    windows = []
    for w in range(n_windows):
        chunk = unit_matrices[bounds[w]:bounds[w + 1]]
        if not chunk:
            continue
        X = np.vstack([c[1] for c in chunk])
        y = np.concatenate([c[2] for c in chunk])
        unit_ids = [str(c[0]) for c in chunk]
        windows.append({
            "index": w,
            "name": f"W{w:02d}-" + "-".join(unit_ids)[:60],
            "X": X,
            "y": y,
            "unit_ids": unit_ids,
            "n_units": len(chunk),
        })
    return windows


def load_checkpoint(ckpt_path: Path):
    from . import model as om
    return om.Model.load(ckpt_path)


def official_metrics(profile: str, model, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    """官方口径指标。A 返回 macro_f1/accuracy；B/C 返回 rmse/mae/asymmetric。"""
    pred = model.predict(X)
    if profile == "A":
        labels = pred.argmax(axis=1)
        classes = PROFILES["A"]["classes"]
        f1s = []
        for c in range(len(classes)):
            tp = float(np.sum((labels == c) & (y == c)))
            fp = float(np.sum((labels == c) & (y != c)))
            fn = float(np.sum((labels != c) & (y == c)))
            prec = tp / (tp + fp) if (tp + fp) else 0.0
            rec = tp / (tp + fn) if (tp + fn) else 0.0
            f1s.append(0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec))
        return {
            "macro_f1": float(np.mean(f1s)) if f1s else 0.0,
            "accuracy": float(np.mean(labels == y)),
        }
    # regression
    v = pred.ravel()
    err = v - y.ravel()
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))
    out = {"rmse": rmse, "mae": mae}
    # NASA 非对称分只在 C（循环单位、误差有界）有意义；B 为分钟单位，不计算
    if profile == "C":
        asym = float(np.sum(np.where(err < 0, np.exp(-err / 13.0) - 1.0, np.exp(err / 10.0) - 1.0)))
        out["asymmetric"] = asym
    return out


def better(profile: str, cand: Dict[str, float], inc: Dict[str, float], eps: float = 1e-6) -> bool:
    """cand 是否严格优于 inc（按官方主指标方向）。"""
    if profile == "A":
        return cand["macro_f1"] > inc["macro_f1"] + eps
    return cand["rmse"] < inc["rmse"] - eps
