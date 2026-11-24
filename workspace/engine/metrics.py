"""评测指标：把"自演化是否有效"量化成赛题口径。

对齐赛题评委看的三件事：
  - 最优阶段表现：全流上的最佳窗口得分
  - 恢复速度：漂移发生后，多少步回到漂移前 90% 的水平
  - 稳定性与抗干扰：稳定期波动、空跑包前后是否塌方
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

DRIFT_KINDS = {"covariate", "concept", "sudden", "gradual"}
STEADY_KINDS = {"stable", "steady_c", "steady_d"}


def episodes(kinds: List[str]) -> List[Dict[str, int]]:
    """把连续的漂移窗合并成一个"漂移事件"，返回起止索引。"""
    out, start = [], None
    for i, k in enumerate(kinds):
        if k in DRIFT_KINDS and start is None:
            start = i
        elif k not in DRIFT_KINDS and start is not None:
            out.append({"start": start, "end": i - 1})
            start = None
    if start is not None:
        out.append({"start": start, "end": len(kinds) - 1})
    return out


def recovery_steps(scores: List[float], kinds: List[str], ratio: float = 0.9) -> List[Dict[str, Any]]:
    res = []
    for ep in episodes(kinds):
        s = ep["start"]
        pre = scores[s - 1] if s > 0 else None
        if pre is None or not np.isfinite(pre) or pre <= 0:
            target = None
        else:
            target = ratio * pre
        hit = None
        if target is not None:
            for j in range(s, len(scores)):
                if np.isfinite(scores[j]) and scores[j] >= target:
                    hit = j - s + 1
                    break
        res.append(
            {
                "start": s,
                "end": ep["end"],
                "kind": kinds[s],
                "pre_score": None if pre is None else round(float(pre), 4),
                "target": None if target is None else round(float(target), 4),
                "steps": hit,
                "recovered": hit is not None,
            }
        )
    return res


def summarize(scores: List[float], kinds: List[str], cost: Dict[str, Any], extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    arr = np.asarray([np.nan if s is None else s for s in scores], dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    stable_mask = np.array([k in STEADY_KINDS for k in kinds])
    stable_scores = arr[stable_mask]
    stable_scores = stable_scores[np.isfinite(stable_scores)]

    rec = recovery_steps([None if not np.isfinite(a) else float(a) for a in arr], kinds)
    recovered = [r for r in rec if r["recovered"]]
    steps_ok = [r["steps"] for r in recovered]

    # 空跑包抗干扰：空跑窗得分不能比上一窗塌太多
    empty_idx = [i for i, k in enumerate(kinds) if k == "empty"]
    empty_impact = []
    for i in empty_idx:
        if i > 0 and np.isfinite(arr[i - 1]) and np.isfinite(arr[i]):
            empty_impact.append(float(arr[i] - arr[i - 1]))

    return {
        "n_windows": len(scores),
        "mean_score": round(float(np.mean(finite)), 4) if finite.size else None,
        "best_score": round(float(np.max(finite)), 4) if finite.size else None,
        "worst_score": round(float(np.min(finite)), 4) if finite.size else None,
        "stability_std": round(float(np.std(stable_scores)), 4) if stable_scores.size else None,
        "max_drawdown": round(float(np.max(np.maximum.accumulate(finite) - finite)), 4) if finite.size else None,
        "recovery": rec,
        "recovery_rate": f"{len(recovered)}/{len(rec)}" if rec else "0/0",
        "avg_recovery_steps": round(float(np.mean(steps_ok)), 2) if steps_ok else None,
        "empty_packet_delta": round(float(np.mean(empty_impact)), 4) if empty_impact else None,
        "cost": cost,
        **(extra or {}),
    }


def compare(policy_metrics: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    """跨策略横向对比表，直接喂给报告。"""
    rows = []
    for name, m in policy_metrics.items():
        rows.append(
            {
                "policy": name,
                "mean_score": m.get("mean_score"),
                "best_score": m.get("best_score"),
                "worst_score": m.get("worst_score"),
                "stability_std": m.get("stability_std"),
                "max_drawdown": m.get("max_drawdown"),
                "recovery_rate": m.get("recovery_rate"),
                "avg_recovery_steps": m.get("avg_recovery_steps"),
                "minutes": m.get("cost", {}).get("minutes"),
                "n_trials": m.get("cost", {}).get("trials"),
            }
        )
    return rows
