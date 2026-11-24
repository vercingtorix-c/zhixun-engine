"""漂移/故障的优先级评分与分类归纳。

场景：工业现场往往不是"一次只来一种漂移"，而是同时出现多个突发——
  传感器增益漂移 + 空跑包 + 推理延迟飙升 + 输出出现 NaN + 算力预算告急。

本模块做两件事：
  1. 分类归纳：把五花八门的信号归入四类故障
        safety       安全故障（必须立刻熔断/拒答，最高优先级）
        system       系统故障（资源/延迟/预算，降级与重分配）
        data_quality 数据质量故障（空跑包/缺失/越界，隔离但不紧急）
        performance  性能退化（漂移，常规五步闭环）
  2. 优先级评分：score = 类别基础分 × 严重度权重 × 强度因子；
     安全故障无条件置顶，同一类内按严重度/强度排序。

这样在"同时面对好几个突发"时，系统会先处理会伤到人/数据/系统的安全故障，
再处理资源类故障，最后才去优化性能。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# 类别基础优先级（越大越先处理）
FAULT_META: Dict[str, Dict[str, Any]] = {
    "safety": {"label": "安全故障", "base": 100,
               "action": "熔断/拒答：隔离当前输出与数据，冻结上线，记录告警"},
    "system": {"label": "系统故障", "base": 70,
               "action": "降级运行：压缩算力预算，必要时回退到最近封版"},
    "data_quality": {"label": "数据质量故障", "base": 55,
                     "action": "隔离该数据包：不参与训练，保留在位模型"},
    "performance": {"label": "性能退化（漂移）", "base": 40,
                    "action": "进入五步闭环：诊断归因 → 实验优化 → 评估上线"},
}

SEVERITY_WEIGHT: Dict[str, float] = {"critical": 1.0, "high": 0.8, "medium": 0.55, "low": 0.3}
SEVERITY_ORDER = {"critical": 3, "high": 2, "medium": 1, "low": 0}


@dataclass
class Signal:
    name: str
    fault_class: str
    magnitude: float          # 0~1，越接近 1 越严重
    severity: str             # critical/high/medium/low
    detail: str = ""


@dataclass
class Incident:
    fault_class: str
    label: str
    severity: str
    score: float
    signals: List[Signal] = field(default_factory=list)
    action: str = ""
    rationale: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fault_class": self.fault_class,
            "label": self.label,
            "severity": self.severity,
            "score": round(self.score, 1),
            "signals": [{"name": s.name, "detail": s.detail, "severity": s.severity,
                         "magnitude": round(s.magnitude, 3)} for s in self.signals],
            "action": self.action,
            "rationale": self.rationale,
        }


def _severity_of(magnitude: float) -> str:
    if magnitude >= 0.8:
        return "critical"
    if magnitude >= 0.55:
        return "high"
    if magnitude >= 0.3:
        return "medium"
    return "low"


def _score(fault_class: str, severity: str, magnitude: float) -> float:
    base = FAULT_META[fault_class]["base"]
    return base * SEVERITY_WEIGHT[severity] * (0.5 + 0.5 * float(magnitude))


def signals_from_state(
    drift: Optional[Any] = None,
    nan_ratio: float = 0.0,
    latency_ms: float = 0.0,
    budget_frac: float = 1.0,
    config_violations: int = 0,
) -> List[Signal]:
    """把一窗内可观测的原始信号，转成统一的 Signal 列表。"""
    sigs: List[Signal] = []

    # ---- 安全故障
    if nan_ratio > 0.01:
        mag = min(1.0, nan_ratio * 10.0)
        sigs.append(Signal("非有限输出", "safety", mag, _severity_of(mag),
                           f"NaN/Inf 输出比例 {nan_ratio:.3f}，触发拒答兜底"))
    if config_violations > 0:
        mag = min(1.0, config_violations / 5.0)
        sigs.append(Signal("配置白名单违反", "safety", mag, _severity_of(mag),
                           f"{config_violations} 个候选越界被拒答"))

    # ---- 系统故障
    if latency_ms > 120.0:
        mag = min(1.0, latency_ms / 600.0)
        sigs.append(Signal("推理延迟超标", "system", mag, _severity_of(mag),
                           f"P95 延迟 {latency_ms:.0f}ms，超过上线预算"))
    if budget_frac < 0.25:
        mag = 1.0 - budget_frac
        sigs.append(Signal("算力预算告急", "system", mag, _severity_of(mag),
                           f"剩余预算 {budget_frac:.0%}，需降级"))

    # ---- 数据质量故障
    if drift is not None and getattr(drift, "empty_packet", False):
        sigs.append(Signal("空跑/无效数据包", "data_quality", 1.0, "critical",
                           "特征与标签无关联，判定为无业务意义噪声包"))

    # ---- 性能退化（漂移）
    if drift is not None and getattr(drift, "level", "none") in ("critical", "warning"):
        cov = float(getattr(drift, "covariate_score", 0.0))
        con = float(getattr(drift, "concept_score", 0.0))
        mag = min(1.0, max(cov / 0.4, con / 0.3))
        sev = "critical" if getattr(drift, "level") == "critical" else _severity_of(mag)
        kind = "强漂移" if getattr(drift, "level") == "critical" else "弱漂移"
        sigs.append(Signal(kind, "performance", mag, sev,
                           f"协变量 {cov:.3f} / 概念 {con:.1%}"))

    return sigs


def assess(signals: List[Signal]) -> List[Incident]:
    """把信号按故障类别归纳，评分并排序（安全故障无条件置顶）。"""
    grouped: Dict[str, List[Signal]] = {}
    for s in signals:
        grouped.setdefault(s.fault_class, []).append(s)

    incidents: List[Incident] = []
    for fc, ss in grouped.items():
        worst_sev = max(ss, key=lambda s: SEVERITY_ORDER[s.severity]).severity
        combined_mag = max(s.magnitude for s in ss)
        incidents.append(
            Incident(
                fault_class=fc,
                label=FAULT_META[fc]["label"],
                severity=worst_sev,
                score=_score(fc, worst_sev, combined_mag),
                signals=ss,
                action=FAULT_META[fc]["action"],
                rationale="；".join(s.detail for s in ss),
            )
        )

    incidents.sort(key=lambda i: (0 if i.fault_class == "safety" else 1,
                                  -SEVERITY_ORDER[i.severity],
                                  -i.score))
    return incidents


def top_priority(incidents: List[Incident]) -> Optional[Incident]:
    return incidents[0] if incidents else None
