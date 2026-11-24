"""第 1 步：漂移监测。

不依赖单一信号——工业现场的漂移常常"无标签、慢、且伴随无效数据包"，
所以这里并行看三路：
  A. 输入分布漂移：逐特征 PSI + KS（协变量漂移的直接证据）
  B. 关系漂移：在线窗口误差相对参考期的抬升比例（概念漂移的证据）
  C. 数据包有效性：缺失/越界/方差坍缩/与标签无关联 → 空跑包识别
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import numpy as np


def _finite(x: np.ndarray) -> np.ndarray:
    return x[np.isfinite(x)]


def psi(ref: np.ndarray, cur: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index：分布迁移的行业标准度量（>0.25 视为显著漂移）。"""
    ref = _finite(np.asarray(ref, dtype=np.float64))
    cur = _finite(np.asarray(cur, dtype=np.float64))
    if ref.size < 10 or cur.size < 10:
        return 0.0
    qs = np.quantile(ref, np.linspace(0, 1, bins + 1))
    qs[0], qs[-1] = -np.inf, np.inf
    qs = np.unique(qs)
    if qs.size < 3:
        return 0.0
    r_hist, _ = np.histogram(ref, bins=qs)
    c_hist, _ = np.histogram(cur, bins=qs)
    r_pct = np.clip(r_hist / max(r_hist.sum(), 1), 1e-6, None)
    c_pct = np.clip(c_hist / max(c_hist.sum(), 1), 1e-6, None)
    return float(np.sum((c_pct - r_pct) * np.log(c_pct / r_pct)))


def ks_stat(ref: np.ndarray, cur: np.ndarray) -> float:
    """两样本 KS 统计量（不引 scipy，评测环境零额外依赖）。"""
    ref = np.sort(_finite(np.asarray(ref, dtype=np.float64)))
    cur = np.sort(_finite(np.asarray(cur, dtype=np.float64)))
    if ref.size < 5 or cur.size < 5:
        return 0.0
    grid = np.concatenate([ref, cur])
    cdf_r = np.searchsorted(ref, grid, side="right") / ref.size
    cdf_c = np.searchsorted(cur, grid, side="right") / cur.size
    return float(np.max(np.abs(cdf_r - cdf_c)))


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


def _trend_of(history: List[float], warn_thr: float) -> Tuple[str, float, int]:
    """用最近漂移强度序列做线性外推，提前预警"即将超阈值"。"""
    if len(history) < 3:
        return "stable", 0.0, -1
    x = np.arange(len(history), dtype=np.float64)
    slope = float(np.polyfit(x, history, 1)[0])
    if abs(slope) < 1e-4:
        return "stable", slope, -1
    last = float(history[-1])
    eta = -1
    if slope > 0:
        for k in range(1, 5):
            if last + slope * k >= warn_thr:
                eta = k
                break
    if slope > 0.005:
        return "rising", slope, eta
    if slope < -0.005:
        return "falling", slope, -1
    return "stable", slope, -1


def _direction_of(ref_col: np.ndarray, cur_col: np.ndarray) -> str:
    """逐特征漂移方向语义：均值/方差位移 → 可解释根因。"""
    ref = _finite(np.asarray(ref_col, dtype=np.float64))
    cur = _finite(np.asarray(cur_col, dtype=np.float64))
    if ref.size < 5 or cur.size < 5:
        return "数据不足"
    sr = float(np.std(ref)) if np.std(ref) > 1e-9 else 1.0
    mean_shift = (float(np.mean(cur)) - float(np.mean(ref))) / sr
    std_ratio = float(np.std(cur)) / max(float(np.std(ref)), 1e-9)
    if mean_shift > 0.5:
        return f"均值上移 {mean_shift:+.1f}σ（疑似增益/量程漂移）"
    if mean_shift < -0.5:
        return f"均值下移 {mean_shift:+.1f}σ（疑似量程/基线漂移）"
    if std_ratio > 1.5:
        return f"方差增大 {std_ratio:.1f}×（疑似噪声/退化）"
    if std_ratio < 0.6:
        return f"方差坍缩 {std_ratio:.2f}×（疑似通道失效）"
    return f"分布迁移（形状变化，均值{mean_shift:+.1f}σ 方差{std_ratio:.1f}×）"


@dataclass
class DriftReport:
    window: int
    level: str = "none"  # none | warning | critical | invalid
    covariate_score: float = 0.0
    concept_score: float = 0.0
    pred_shift: float = 0.0
    empty_packet: bool = False
    top_features: List[Tuple[int, float]] = field(default_factory=list)
    feature_psi: List[float] = field(default_factory=list)
    feature_ks: List[float] = field(default_factory=list)
    trend: str = "stable"          # stable | rising | falling
    trend_slope: float = 0.0
    eta_windows: int = -1          # 按趋势外推预计几步后超阈值（-1=不超/不可估）
    feature_direction: List[Dict[str, Any]] = field(default_factory=list)
    reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "window": self.window,
            "level": self.level,
            "covariate_score": round(self.covariate_score, 4),
            "concept_score": round(self.concept_score, 4),
            "pred_shift": round(self.pred_shift, 4),
            "empty_packet": self.empty_packet,
            "top_features": [[int(i), round(float(s), 4)] for i, s in self.top_features],
            "trend": self.trend,
            "trend_slope": round(self.trend_slope, 4),
            "eta_windows": int(self.eta_windows),
            "feature_direction": self.feature_direction,
            "reasons": self.reasons,
        }


class DriftMonitor:
    """参考期（最近一次"稳定且已封版"的窗口）vs 当前窗口。"""

    def __init__(self, cfg: Dict[str, Any] | None = None) -> None:
        c = cfg or {}
        self.psi_warn = float(c.get("psi_warn", 0.15))
        self.psi_crit = float(c.get("psi_crit", 0.25))
        self.concept_warn = float(c.get("concept_warn", 0.08))
        self.concept_crit = float(c.get("concept_crit", 0.20))
        self.ref_X: np.ndarray | None = None
        self.ref_pred: np.ndarray | None = None
        self.ref_loss: float | None = None
        self.ref_corr: np.ndarray | None = None
        self.score_history: List[float] = []  # 漂移强度时间序列，用于趋势外推

    def set_reference(
        self, X: np.ndarray, pred: np.ndarray | None, loss: float, y: np.ndarray | None = None
    ) -> None:
        self.ref_X = np.array(X, dtype=np.float64, copy=True)
        self.ref_pred = None if pred is None else np.array(pred, dtype=np.float64, copy=True)
        self.ref_loss = float(loss)
        if y is not None:
            self.ref_corr = np.array([_corr(X[:, j], y) for j in range(X.shape[1])])

    # ------------------------------------------------------------------
    def check(
        self,
        window_index: int,
        X_cur: np.ndarray,
        y_cur: np.ndarray | None,
        pred_cur: np.ndarray | None,
        loss_cur: float | None,
    ) -> DriftReport:
        rep = DriftReport(window=window_index)
        if self.ref_X is None:
            rep.level = "none"
            rep.reasons.append("尚无参考期，跳过监测")
            return rep

        X_cur = np.asarray(X_cur, dtype=np.float64)

        # ---- C. 数据包有效性（先判空跑，避免污染后续统计）
        finite_ratio = float(np.mean(np.isfinite(X_cur))) if X_cur.size else 0.0
        var_collapse = float(np.mean([np.nanstd(X_cur[:, j]) < 1e-9 for j in range(X_cur.shape[1])]))
        # 双重信号判空跑：单特征最强线性 R² + 在位模型的 R²
        max_feature_r2 = 0.0
        pred_r2 = 0.0
        if y_cur is not None:
            r2s = [_corr(X_cur[:, j], y_cur) ** 2 for j in range(X_cur.shape[1])]
            max_feature_r2 = float(np.nanmax(r2s)) if r2s else 0.0
        if y_cur is not None and pred_cur is not None and len(y_cur) > 5:
            ss_res = float(np.sum((np.asarray(y_cur).ravel() - np.asarray(pred_cur).ravel()) ** 2))
            ss_tot = float(np.sum((np.asarray(y_cur).ravel() - np.mean(np.asarray(y_cur).ravel())) ** 2))
            pred_r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
        rep.reasons.append(
            f"有效值占比={finite_ratio:.3f} 最强特征R²={max_feature_r2:.3f} 在位模型R²={pred_r2:.3f}"
        )

        # 判空跑：特征与标签几乎无线性关系（max R² 近 0）且模型也解释不了（pred R² 为负）
        # 双重确认避免把"轻微漂移"误判成空跑
        is_noise_target = (y_cur is not None) and (max_feature_r2 < 0.04) and (pred_r2 < 0.0)
        if finite_ratio < 0.5 or var_collapse > 0.5 or is_noise_target:
            rep.level = "invalid"
            rep.empty_packet = True
            rep.reasons.append("判定为空跑/无效数据包，隔离且不参与训练")
            return rep

        # ---- A. 输入分布漂移
        psi_vals, ks_vals = [], []
        for j in range(min(X_cur.shape[1], self.ref_X.shape[1])):
            psi_vals.append(psi(self.ref_X[:, j], X_cur[:, j]))
            ks_vals.append(ks_stat(self.ref_X[:, j], X_cur[:, j]))
        psi_arr = np.asarray(psi_vals)
        ks_arr = np.asarray(ks_vals)
        rep.feature_psi = [float(v) for v in psi_arr]
        rep.feature_ks = [float(v) for v in ks_arr]
        # 协变量漂移强度 = 高位特征的平均 PSI（忽略长尾，避免单点噪声误报）
        topk = max(1, int(0.25 * max(psi_arr.size, 1)))
        rep.covariate_score = float(np.mean(np.sort(psi_arr)[-topk:]))
        idx = np.argsort(psi_arr)[::-1][:5]
        rep.top_features = [(int(i), float(psi_arr[i])) for i in idx if psi_arr[i] > self.psi_warn]
        rep.feature_direction = [
            {
                "feature": int(i),
                "psi": round(float(psi_arr[i]), 4),
                "direction": _direction_of(self.ref_X[:, i], X_cur[:, i]),
            }
            for i in idx if psi_arr[i] > self.psi_warn
        ]

        # ---- B. 关系漂移（有标签时直接用误差抬升；无标签时用预测分布位移）
        if loss_cur is not None and self.ref_loss is not None and self.ref_loss > 1e-9:
            rep.concept_score = float(max(0.0, (loss_cur - self.ref_loss) / self.ref_loss))
            rep.reasons.append(
                f"在线误差 {loss_cur:.4f} vs 参考 {self.ref_loss:.4f}，抬升 {rep.concept_score:.1%}"
            )
        if pred_cur is not None and self.ref_pred is not None and np.std(self.ref_pred) > 1e-9:
            rep.pred_shift = float(
                abs(np.mean(pred_cur) - np.mean(self.ref_pred)) / max(np.std(self.ref_pred), 1e-9)
            )

        # ---- 综合定级
        if rep.covariate_score >= self.psi_crit or rep.concept_score >= self.concept_crit or rep.pred_shift > 1.5:
            rep.level = "critical"
        elif rep.covariate_score >= self.psi_warn or rep.concept_score >= self.concept_warn or rep.pred_shift > 0.8:
            rep.level = "warning"

        # ---- 趋势外推：当前未超阈值、但强度斜率向上时提前预警
        intensity = float(rep.covariate_score + rep.concept_score)
        self.score_history.append(intensity)
        self.score_history = self.score_history[-6:]
        warn_total = self.psi_warn + self.concept_warn
        rep.trend, rep.trend_slope, rep.eta_windows = _trend_of(self.score_history, warn_total)
        if rep.trend == "rising" and rep.level == "none":
            rep.reasons.append(
                f"前瞻预警：漂移强度趋势上升（斜率 {rep.trend_slope:.3f}/窗），"
                f"预计 {rep.eta_windows} 窗后超阈值"
            )

        n_drift_feat = int(np.sum(ks_arr > 0.2))
        rep.reasons.append(
            f"显著漂移特征数 {n_drift_feat}/{psi_arr.size}，协变量强度 {rep.covariate_score:.3f}"
        )
        return rep
