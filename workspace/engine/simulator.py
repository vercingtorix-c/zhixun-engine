"""数字孪生仿真场（Simulation Mode）：在无漂移时主动预演未来工况。

核心思想（用户方案）：
  工业生产在"看起来正常"时，其实最危险的是下一批原料、下一次设备老化、
  下一次传感器量程漂移。与其等它发生再被动响应，不如让一个从当前工况学到的
  代理模型，按一套"漂移算子库"主动合成未来可能出现的工况，喂给预适应缓冲——
  让模型在真实漂移还没来时就被"预免疫"。

  一旦真实漂移被监测到，上层会冻结仿真场，把省下的算力全部转给优化器。

实现约束（诚实、可解释、可复现）：
  - 代理模型 = 当前窗口学到的工业过程映射 f(x)（"物理"的轻量替代）。
  - 漂移算子 = 可解释的参数化扰动：协变量（增益/偏置）、概念（关系项线性扰动）、
    突变（全量重标定）、渐进（在窗内插值过渡）。
  - 回归任务支持全部算子；分类任务只做协变量增强（不臆造标签翻转），
    并在场景说明中如实标注。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from . import model as model_lib
from .types import TaskSpec


@dataclass
class Scenario:
    kind: str
    features: List[int]
    gain: np.ndarray
    offset: np.ndarray
    concept_coef: np.ndarray
    likelihood: float
    reason: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "features": [int(f) for f in self.features],
            "gain": [round(float(g), 3) for g in self.gain[self.features]],
            "offset": [round(float(o), 3) for o in self.offset[self.features]],
            "concept_coef": [round(float(c), 4) for c in self.concept_coef[self.features]],
            "likelihood": round(self.likelihood, 3),
            "reason": self.reason,
        }


PRIOR = {"covariate": 0.35, "concept": 0.25, "sudden": 0.15, "gradual": 0.25}
KIND_REASON = {
    "covariate": "传感器量程漂移/原料批次波动（输入分布变，关系不变）",
    "concept": "工艺参数关系变化（输入不变，映射关系变）",
    "sudden": "设备更换/换型（量程与关系同时跳变）",
    "gradual": "长期老化导致的渐进漂移",
}


class Simulator:
    def __init__(self, task: TaskSpec, cfg: Optional[Dict[str, Any]] = None) -> None:
        self.task = task
        self.cfg = cfg or {}
        self.surrogate = None
        self.surrogate_cfg = None
        self.fitted_on: Optional[str] = None

    # ------------------------------------------------------------------
    def fit(self, X: np.ndarray, y: np.ndarray, tag: str = "") -> None:
        """从当前稳定工况学一个过程代理模型。"""
        if len(y) < 20:
            return
        c = model_lib.default_config(self.task)
        c["params"]["learning_rate"] = 0.1
        c["params"]["max_iter"] = 120
        self.surrogate_cfg = c
        self.surrogate = model_lib.fit(c, self.task, X, y, seed=0)
        self.fitted_on = tag

    # ------------------------------------------------------------------
    def sample_scenarios(
        self,
        n_features: int,
        rng: np.random.Generator,
        history: Optional[List[Dict[str, Any]]] = None,
        feature_std: Optional[np.ndarray] = None,
        n: int = 2,
    ) -> List[Scenario]:
        """按先验 + 历史漂移频率，采样 n 个"未来可能发生"的漂移场景。"""
        probs = dict(PRIOR)
        if history:
            kinds = [h.get("drift_type") for h in history if h.get("drift_type") in PRIOR]
            if kinds:
                for k in PRIOR:
                    probs[k] = 0.5 * PRIOR[k] + 0.5 * (kinds.count(k) / len(kinds))
        kinds = list(probs)
        weights = np.array([probs[k] for k in kinds])
        weights = weights / weights.sum()

        std = feature_std if feature_std is not None else np.ones(n_features)
        std = np.where(std < 1e-9, 1.0, std)

        out: List[Scenario] = []
        for _ in range(n):
            kind = kinds[int(rng.choice(len(kinds), p=weights))]
            n_sel = int(rng.integers(1, min(4, n_features) + 1))
            feats = list(rng.choice(n_features, size=max(1, n_sel), replace=False))
            gain = np.ones(n_features)
            offset = np.zeros(n_features)
            concept = np.zeros(n_features)

            if kind == "covariate":
                for f in feats:
                    gain[f] = float(rng.normal(1.0, 0.16))
                    offset[f] = float(rng.normal(0.0, 0.5 * std[f]))
            elif kind == "concept":
                for f in feats:
                    concept[f] = float(rng.normal(0.0, 0.35))
            elif kind == "sudden":
                for f in feats:
                    gain[f] = float(rng.uniform(0.75, 1.3))
                    offset[f] = float(rng.normal(0.0, 0.8 * std[f]))
                for f in feats:
                    concept[f] = float(rng.normal(0.0, 0.5))
            elif kind == "gradual":
                for f in feats:
                    gain[f] = float(rng.normal(1.0, 0.08))
                    offset[f] = float(rng.normal(0.0, 0.3 * std[f]))
                    concept[f] = float(rng.normal(0.0, 0.12))

            gain = np.clip(gain, 0.6, 1.5)
            concept = np.clip(concept, -1.2, 1.2)
            out.append(
                Scenario(
                    kind=kind,
                    features=feats,
                    gain=gain,
                    offset=offset,
                    concept_coef=concept,
                    likelihood=float(probs[kind]),
                    reason=KIND_REASON[kind],
                )
            )
        return out

    # ------------------------------------------------------------------
    def generate(
        self,
        X_ref: np.ndarray,
        scenario: Scenario,
        n: int,
        rng: np.random.Generator,
    ) -> Optional[tuple]:
        """按场景合成 (X_sim, y_sim)。代理模型未就绪时返回 None。"""
        if self.surrogate is None or len(X_ref) == 0:
            return None
        idx = rng.integers(0, len(X_ref), size=n)
        base = np.asarray(X_ref[idx], dtype=np.float64)

        X_sim = base * scenario.gain + scenario.offset

        if self.task.task_type == "regression":
            y_base, rep = model_lib.safe_predict(self.surrogate_cfg, self.surrogate, base, self.task)
            # 概念扰动：关系项线性变化（作用于扰动后的特征）
            concept_term = np.zeros(n)
            for f in scenario.features:
                if abs(scenario.concept_coef[f]) > 1e-9:
                    concept_term += scenario.concept_coef[f] * X_sim[:, f]
            noise_scale = 0.35 * self.task.scale if self.task.scale > 1e-9 else 0.35
            y_sim = y_base + concept_term + rng.normal(0.0, noise_scale, size=n)
            return X_sim, y_sim

        # 分类：只做协变量增强，标签沿用被抽样样本的真实标签（不臆造标签翻转）
        y_sim = np.asarray(self.task.y_ref if hasattr(self.task, "y_ref") else None)
        return X_sim, None  # 占位，由调用方填充标签

    def generate_labeled(
        self,
        X_ref: np.ndarray,
        y_ref: np.ndarray,
        scenario: Scenario,
        n: int,
        rng: np.random.Generator,
    ) -> Optional[tuple]:
        """生成带标签的合成窗；分类任务直接沿用抽样行标签。"""
        if self.surrogate is None or len(X_ref) == 0:
            return None
        idx = rng.integers(0, len(X_ref), size=n)
        base = np.asarray(X_ref[idx], dtype=np.float64)
        X_sim = base * scenario.gain + scenario.offset
        if self.task.task_type == "regression":
            return self.generate(X_ref, scenario, n, rng)
        y_sim = np.asarray(y_ref[idx])
        return X_sim, y_sim
