"""第 2 步：诊断归因。

监测只告诉"漂了"，不告诉"为什么漂、该动哪里"。这一步回答三个问题：
  1. 是哪一类漂移：协变量漂移（输入分布变）还是概念漂移（输入输出关系变）？
  2. 漂在哪些特征上：按 PSI × 误差关联变化 排序，输出可解释的归因清单。
  3. 该走哪条处置路线：不动 / 隔离 / 只重训 / 重训+重调参 / 重调参+样本加权。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from .monitor import DriftReport


ACTION_BUDGET = {
    "hold": (0, 0.0),
    "quarantine": (0, 0.0),
    "retrain": (0, 1.5),          # 只重训，固定超参，几乎不耗算力
    "recalibrate": (4, 3.0),      # 轻量：少量试验覆盖量程/加权
    "retune": (10, 10.0),         # 完整超参搜索
    "retune_reweight": (12, 12.0),  # 复合漂移：搜索 + 样本加权
    "escalate": (16, 15.0),       # 上次处置失败：扩大搜索
}


@dataclass
class Strategy:
    action: str
    drift_type: str
    confidence: float
    reason: str
    trials: int = 0
    minutes: float = 0.0
    drop_features: List[int] = field(default_factory=list)
    recency_half_life: float = 4.0
    attributions: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action,
            "drift_type": self.drift_type,
            "confidence": round(self.confidence, 3),
            "reason": self.reason,
            "trials": self.trials,
            "minutes": self.minutes,
            "drop_features": self.drop_features,
            "recency_half_life": self.recency_half_life,
            "attributions": self.attributions,
        }


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 5:
        return 0.0
    a, b = a[m], b[m]
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def attribute_features(
    X_cur: np.ndarray,
    y_cur: Optional[np.ndarray],
    residual_cur: Optional[np.ndarray],
    X_ref: np.ndarray,
    residual_ref: Optional[np.ndarray],
    psi_vals: List[float],
) -> List[Dict[str, Any]]:
    """归因打分 = 分布漂移强度(PSI) × 该特征与模型误差关联的变化量。

    直觉：一个特征既"移动了"，又"改变了它对误差的贡献"，才最可能是根因；
    只移动但不影响误差的特征（比如换了个采集通道）优先级低。
    """
    rows: List[Dict[str, Any]] = []
    d = min(X_cur.shape[1], len(psi_vals))
    for j in range(d):
        p = float(psi_vals[j]) if j < len(psi_vals) else 0.0
        d_corr = 0.0
        if y_cur is not None and residual_cur is not None and residual_ref is not None:
            d_corr = abs(_corr(X_cur[:, j], residual_cur) - _corr(X_ref[:, j], residual_ref))
        contrib = p * (1.0 + 3.0 * d_corr)
        if contrib <= 0:
            continue
        rows.append(
            {
                "feature": int(j),
                "psi": round(p, 4),
                "delta_err_corr": round(d_corr, 4),
                "contribution": round(contrib, 4),
            }
        )
    rows.sort(key=lambda r: r["contribution"], reverse=True)
    return rows[:8]


class Diagnoser:
    def __init__(self, cfg: Dict[str, Any] | None = None) -> None:
        c = cfg or {}
        self.cov_hi = float(c.get("cov_hi", 0.30))
        self.con_hi = float(c.get("con_hi", 0.12))
        self.max_drop = int(c.get("max_drop_features", 4))
        self.ref_corr: Optional[np.ndarray] = None

    def set_reference(self, X_ref: np.ndarray, y_ref: Optional[np.ndarray], residual_ref: Optional[np.ndarray]):
        self.ref_X = X_ref
        self.ref_resid = residual_ref

    def diagnose(
        self,
        rep: DriftReport,
        X_cur: np.ndarray,
        y_cur: Optional[np.ndarray],
        residual_cur: Optional[np.ndarray],
        previous_attempt_failed: bool = False,
    ) -> Strategy:
        if rep.level == "invalid" or rep.empty_packet:
            t, m = ACTION_BUDGET["quarantine"]
            return Strategy(
                action="quarantine",
                drift_type="invalid-packet",
                confidence=0.95,
                reason="数据包与标签无关联/有效性不足，隔离该包，保留在位模型，不消耗算力",
                trials=t,
                minutes=m,
            )

        if rep.level == "none":
            # 前瞻预警：当前未超阈值，但强度趋势上升 → 提前低成本重训防御
            if rep.trend == "rising":
                t, m = ACTION_BUDGET["retrain"]
                return Strategy(
                    action="retrain",
                    drift_type="pre-drift",
                    confidence=0.7,
                    reason=f"前瞻预警：漂移强度趋势上升，提前低成本重训防御（预计 {rep.eta_windows} 窗后超阈值）",
                    trials=t,
                    minutes=m,
                )
            t, m = ACTION_BUDGET["hold"]
            return Strategy(
                action="hold",
                drift_type="none",
                confidence=0.9,
                reason="无显著漂移信号，维持在位模型",
                trials=t,
                minutes=m,
            )

        attributions = attribute_features(
            X_cur,
            y_cur,
            residual_cur,
            getattr(self, "ref_X", X_cur),
            getattr(self, "ref_resid", residual_cur),
            rep.feature_psi,
        )
        # 把监测层的方向语义合并进归因，形成"哪个特征、往哪漂、影响多大"的可解释证据链
        direction_by_feat = {d["feature"]: d["direction"] for d in rep.feature_direction}
        for a in attributions:
            a["direction"] = direction_by_feat.get(a["feature"], "")

        cov, con = rep.covariate_score, rep.concept_score
        if cov >= self.cov_hi and con >= self.con_hi:
            dtype, conf = "mixed", min(0.95, 0.6 + 0.2 * cov + 0.2 * con)
        elif con >= self.con_hi and con >= cov * 0.6:
            dtype, conf = "concept", min(0.95, 0.55 + 0.3 * con)
        elif cov >= self.cov_hi:
            dtype, conf = "covariate", min(0.95, 0.55 + 0.25 * cov)
        else:
            dtype, conf = "mild", 0.6

        # 只剔除"高漂移且与误差无关"的特征：这类特征通常是采集通道失效，
        # 留着只会给模型喂噪声；而高漂移+高误差关联的特征要保留（它是新规律所在）。
        drop = [
            a["feature"]
            for a in attributions
            if a["psi"] > 0.5 and a["delta_err_corr"] < 0.05 and len(attributions) > 3
        ][: self.max_drop]

        if dtype in ("concept", "mixed"):
            action = "retune_reweight"
            hlf = 2.5
            reason = (
                f"判定为{'复合' if dtype == 'mixed' else '概念'}漂移：误差抬升 {con:.1%}，"
                "旧超参已不适配新关系，进入超参搜索并提高近期样本权重"
            )
        elif dtype == "covariate":
            action = "recalibrate"
            hlf = 3.0
            reason = (
                f"判定为协变量漂移：协变量强度 {cov:.3f}，关系基本保持，"
                "无需全量重调参，做量程适配+近期重训（成本优先）"
            )
        else:
            action = "retrain"
            hlf = 4.0
            reason = f"轻度漂移（cov={cov:.3f}, con={con:.1%}），先做低成本重训观察"

        if previous_attempt_failed:
            action = "escalate"
            hlf = 2.0
            reason = "上一次处置未通过上线门槛，升级搜索预算并收紧近期加权"

        t, m = ACTION_BUDGET[action]
        return Strategy(
            action=action,
            drift_type=dtype,
            confidence=conf,
            reason=reason,
            trials=t,
            minutes=m,
            drop_features=drop,
            recency_half_life=hlf,
            attributions=attributions,
        )


def budget_scaler(
    rep: Optional[DriftReport],
    reject_streak: int,
    budget_frac: float,
    include_severity: bool = True,
) -> float:
    """成本收益自适应预算：算力投入与漂移严重度、恢复难度、剩余预算挂钩。

    返回预算乘数（0.4 ~ 2.0 封顶），作用在 ACTION_BUDGET 的 trials/minutes 上：
      - 漂移越严重 → 越值得投入（critical 1.6 / warning 1.0 / none 0.4）
      - 前瞻预警（trend=rising）→ 在真超阈值前低投入防御（0.8）
      - 连续处置失败 → 阶梯加码（每失败一次 +0.2，体现"恢复越难越值得"）
      - 剩余预算紧张 → 收缩（<25% 打 0.6 折）

    include_severity=False 时，base 固定 1.0：用于"动作本身已含漂移严重度"的
    调用方（官方续训闭环的 action→epochs 映射已含严重度），此处只保留
    失败加码与预算收缩两个维度，避免双重放大导致过拟合。
    """
    if rep is None:
        base = 0.4 if include_severity else 1.0
    elif not include_severity:
        base = 1.0
    elif rep.level == "critical":
        base = 1.6
    elif rep.level == "warning":
        base = 1.0
    elif rep.level == "none":
        base = 0.8 if rep.trend == "rising" else 0.4
    else:  # invalid / empty
        base = 0.4
    scale = base + 0.2 * max(0, int(reject_streak))
    if budget_frac < 0.25:
        scale *= 0.6
    return float(np.clip(scale, 0.4, 2.0))
