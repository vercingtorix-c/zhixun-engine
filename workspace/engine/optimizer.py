"""第 3 步：实验优化（在预算与安全约束下自动调参）。

刻意不用"暴力网格 + 全量训练"：赛题明确指出要看成本与论证质量。
这里用 **逐次减半（successive halving）**：
  - 每轮先给所有候选很小的验证子集，快速淘汰大部分；
  - 晋级者在更大的子集上复核；
  - 每个 trial 都落盘：配置、预算、得分、耗时、随机种子 → 可复现、可审计。

搜索空间有界且白名单校验（见 model.validate_config），越界配置直接拒答，
保证"自动化的自主性"不会把系统带出安全边界。
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import numpy as np

from . import model as model_lib
from .types import TaskSpec


def estimate_minutes(n_samples: int, cfg: Dict[str, Any]) -> float:
    """估算单次试验的算力消耗（GPU 分钟），用于预算控制与成本论证。"""
    p = cfg.get("params", {})
    iters = float(p.get("max_iter", p.get("n_estimators", 100)))
    leaves = float(p.get("max_leaf_nodes", p.get("max_depth", 15)))
    return max(0.02, n_samples / 3000.0 * iters / 100.0 * (1.0 + leaves / 63.0) * 0.35)


@dataclass
class Trial:
    trial_id: int
    config: Dict[str, Any]
    stage: int
    val_score: float
    loss: float
    minutes: float
    n_samples: int
    seed: int
    status: str = "ok"
    note: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trial_id": self.trial_id,
            "stage": self.stage,
            "val_score": round(self.val_score, 5),
            "loss": round(self.loss, 5),
            "minutes": round(self.minutes, 3),
            "n_samples": self.n_samples,
            "seed": self.seed,
            "status": self.status,
            "note": self.note,
            "config": {
                "family": self.config.get("family"),
                "params": self.config.get("params", {}),
                "recency_half_life": self.config.get("recency_half_life"),
                "drop_features": self.config.get("drop_features", []),
            },
        }


@dataclass
class OptimizeResult:
    champion: Dict[str, Any]
    champion_score: float
    trials: List[Trial] = field(default_factory=list)
    minutes_used: float = 0.0
    wall_seconds: float = 0.0
    rejected: List[Dict[str, Any]] = field(default_factory=list)
    stopped_reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "champion": {
                "family": self.champion.get("family"),
                "params": self.champion.get("params", {}),
                "recency_half_life": self.champion.get("recency_half_life"),
                "drop_features": self.champion.get("drop_features", []),
            },
            "champion_score": round(self.champion_score, 5),
            "minutes_used": round(self.minutes_used, 3),
            "wall_seconds": round(self.wall_seconds, 2),
            "n_trials": len(self.trials),
            "rejected": self.rejected,
            "stopped_reason": self.stopped_reason,
        }


def _sample_config(rng: np.random.Generator, incumbent: Dict[str, Any]) -> Dict[str, Any]:
    family = "hgb" if rng.random() < 0.93 else "rf"
    space = model_lib.CONFIG_SPACE[family]
    params: Dict[str, Any] = {}
    for k, (lo, hi, kind) in space.items():
        if kind == "int":
            u = rng.random() ** 1.6  # 偏向小复杂度，先压成本
            params[k] = int(round(lo + (hi - lo) * u))
        else:
            params[k] = float(lo + (hi - lo) * rng.random())
    hlf = float(rng.choice([1.5, 2.0, 2.5, 4.0, 6.0]))
    drop = list(incumbent.get("drop_features", [])) if rng.random() < 0.5 else []
    return {
        "family": family,
        "params": params,
        "recency_half_life": hlf,
        "drop_features": drop,
        "proposed_by": "random-search",
    }


def _incumbent_variant(incumbent: Dict[str, Any], strategy: Dict[str, Any]) -> Dict[str, Any]:
    """把在位配置也作为一个候选：优化不允许比现状更差。"""
    return {
        "family": incumbent.get("family", "hgb"),
        "params": dict(incumbent.get("params", {})),
        "recency_half_life": float(strategy.get("recency_half_life", 4.0) or 4.0),
        "drop_features": list(strategy.get("drop_features", [])),
        "proposed_by": "incumbent-adapted",
    }


def combine_buffer(
    buffer: List[Dict[str, Any]], half_life: float, cur_window: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """回放缓冲 + 近期加权：老数据既不能全扔（方差大），也不能等权（会拖累）。"""
    Xs, ys, ws = [], [], []
    for item in buffer:
        # 显式给出权重（例如数字孪生合成样本）时直接采用；否则按窗口年龄衰减
        if "w" in item:
            w = float(item["w"])
        else:
            age = max(0, cur_window - int(item["window"]))
            w = 0.5 ** (age / max(float(half_life), 1e-3))
        Xs.append(item["X"])
        ys.append(item["y"])
        ws.append(np.full(len(item["y"]), w))
    return np.vstack(Xs), np.concatenate(ys), np.concatenate(ws)


def _split_val(n: int, rng: np.random.Generator, frac: float = 0.28) -> Tuple[np.ndarray, np.ndarray]:
    idx = rng.permutation(n)
    n_val = max(30, int(n * frac))
    return idx[n_val:], idx[:n_val]


def optimize(
    task: TaskSpec,
    buffer: List[Dict[str, Any]],
    cur_window: int,
    incumbent: Dict[str, Any],
    strategy: Dict[str, Any],
    budget_trials: int,
    budget_minutes: float,
    seed: int = 0,
    total_minutes_left: float = 1e9,
) -> OptimizeResult:
    t_start = time.perf_counter()
    hlf = float(strategy.get("recency_half_life", 4.0) or 4.0)
    X, y, w = combine_buffer(buffer, hlf, cur_window)
    rng_val = np.random.default_rng(seed + 999)
    tr_idx, va_idx = _split_val(len(y), rng_val)
    n_use_stage3 = len(va_idx)

    # 预算为 0：只按在位配置重训（低成本路线）
    if budget_trials <= 0:
        cfg = _incumbent_variant(incumbent, strategy)
        m = model_lib.fit(cfg, task, X[tr_idx], y[tr_idx], w[tr_idx], seed=seed)
        pred, rep = model_lib.safe_predict(cfg, m, X[va_idx], task)
        sc = float(task.score(y[va_idx], pred))
        return OptimizeResult(
            champion=cfg,
            champion_score=sc,
            trials=[Trial(0, cfg, 3, sc, task.loss(y[va_idx], pred), 0.0, len(tr_idx), seed,
                          note="沿用在位配置（仅重训）")],
            minutes_used=0.0,
            wall_seconds=time.perf_counter() - t_start,
            stopped_reason="预算为 0，仅执行在位配置重训",
        )

    # 候选池：在位配置 + 随机搜索
    cands: List[Dict[str, Any]] = [_incumbent_variant(incumbent, strategy)]
    for _ in range(max(0, budget_trials - 1)):
        cands.append(_sample_config(rng_val, incumbent))

    rejected: List[Dict[str, Any]] = []
    accepted: List[Dict[str, Any]] = []
    for c in cands:
        ok, why = model_lib.validate_config(c)
        if ok:
            accepted.append(c)
        else:
            rejected.append({"family": c.get("family"), "reason": why})

    if not accepted:
        cfg = _incumbent_variant(incumbent, strategy)
        return OptimizeResult(
            champion=cfg,
            champion_score=-1e9,
            rejected=rejected,
            stopped_reason="全部候选被安全白名单拒绝，回退在位配置",
            wall_seconds=time.perf_counter() - t_start,
        )

    trials: List[Trial] = []
    minutes_used = 0.0
    survivors = accepted
    best: Tuple[float, Dict[str, Any]] = (float("inf"), survivors[0])
    stopped_reason = "完成逐次减半"
    budget_hit = False

    for stage, (frac, note) in enumerate(
        [(0.30, "stage1-快速筛选"), (0.65, "stage2-复核"), (1.00, "stage3-全量验证")]
    ):
        if not survivors:
            break
        n_use = max(30, min(n_use_stage3, int(n_use_stage3 * frac)))
        sub = va_idx[:n_use]
        scored: List[Tuple[float, Dict[str, Any]]] = []

        for c in survivors:
            est = estimate_minutes(len(tr_idx), c)
            if minutes_used + est > budget_minutes or minutes_used + est > total_minutes_left:
                stopped_reason = f"触及预算上限（已用 {minutes_used:.2f} 分钟）提前收敛"
                budget_hit = True
                break
            cfg = dict(c)
            cfg["drop_features"] = list(c.get("drop_features", []))
            m = model_lib.fit(cfg, task, X[tr_idx], y[tr_idx], w[tr_idx], seed=seed + stage)
            pred, rep = model_lib.safe_predict(cfg, m, X[sub], task)
            if rep["refused"]:
                trials.append(Trial(len(trials), cfg, stage, -1e9, 1e9, est, len(tr_idx),
                                    seed + stage, status="refused", note=rep["reason"]))
                minutes_used += est
                continue
            loss = float(task.loss(y[sub], pred))
            score = float(task.score(y[sub], pred))
            trials.append(Trial(len(trials), cfg, stage, score, loss, est, len(tr_idx),
                                seed + stage, note=note))
            minutes_used += est
            scored.append((loss, cfg))

        if not scored:
            stopped_reason = "本轮全部候选被拒答，停止搜索"
            break

        scored.sort(key=lambda t: t[0])
        keep = max(1, int(math.ceil(len(scored) / 3.0)))
        survivors = [c for _, c in scored[:keep]]

        # 阶段 3（全量验证）的胜者才是真冠军；低阶段只用于淘汰
        if stage == len([0.30, 0.65, 1.00]) - 1:
            best = (scored[0][0], scored[0][1])

        if budget_hit:
            # 预算耗尽时，用当前阶段最优作为冠军（并标注为"预算受限收敛"）
            if best[0] == float("inf"):
                best = (scored[0][0], scored[0][1])
            break

    champion = best[1] if best[1] is not None else survivors[0]
    # 冠军用完整验证集复核一次，保证与其它策略同口径可比
    m = model_lib.fit(champion, task, X[tr_idx], y[tr_idx], w[tr_idx], seed=seed + 77)
    pred, rep = model_lib.safe_predict(champion, m, X[va_idx], task)
    champ_score = float(task.score(y[va_idx], pred)) if not rep["refused"] else -1e9
    champ_loss = float(task.loss(y[va_idx], pred))
    trials.append(Trial(len(trials), champion, 3, champ_score, champ_loss, 0.0, len(tr_idx),
                        seed + 77, note="冠军全量复核"))

    return OptimizeResult(
        champion=champion,
        champion_score=champ_score,
        trials=trials,
        minutes_used=minutes_used,
        wall_seconds=time.perf_counter() - t_start,
        rejected=rejected,
        stopped_reason=stopped_reason,
    )
