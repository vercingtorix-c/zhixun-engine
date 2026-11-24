"""五步闭环编排器 + 对照策略。

闭环（无人工干预，对齐赛题官方口径）：
  漂移监测 → 诊断归因 → 实验优化 → 评估上线 → 回滚/封版

  其中第 4/5 步：候选模型过不了门槛 → 显式"回滚"到上一稳定封版；
  通过门槛 → 晋升上线并封版。两层自进化 = 模型层自迭代（重训/重调参）
  + 策略层自改进（诊断策略、优先级评分、仿真开关、预算分配随反馈调整）。

对照策略（同一数据流、同一评测口径）：
  static     只在首窗训练一次，永不更新（对应赛题的"静态基线"）
  periodic   每窗无脑重训，超参固定（只重训不调参）
  retune_all 每窗都全量重调参（不管有没有漂移，成本最高）
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from . import model as model_lib
from .diagnose import Diagnoser, budget_scaler
from .gate import GateConfig, Version, evaluate_pair
from . import incident as incident_lib
from .metrics import summarize
from .monitor import DriftMonitor
from .optimizer import combine_buffer, optimize
from .safety import SafeGate
from .seal import Sealer
from .simulator import Simulator
from .trace import Trace
from .types import TaskSpec, Window

MAX_BUFFER_WINDOWS = 8
MAX_TOTAL_MINUTES = 600.0


def _split_window(w: Window, seed: int, val_frac: float = 0.3) -> Dict[str, np.ndarray]:
    """把一个窗切成训练部分与留出部分（留出部分只用于上线门槛，不进训练缓冲）。"""
    rng = np.random.default_rng(seed + w.index)
    n = len(w.y_train)
    idx = rng.permutation(n)
    n_val = max(20, int(n * val_frac))
    va, tr = idx[:n_val], idx[n_val:]
    return {
        "X_fit": w.X_train[tr],
        "y_fit": w.y_train[tr],
        "X_val": w.X_train[va],
        "y_val": w.y_train[va],
    }


def _proba_of(cfg: Dict[str, Any], model: Any, X: np.ndarray, task: TaskSpec):
    """分类时取概率（用于安全拒答的置信度判断）；回归返回 None。"""
    if task.task_type != "classification":
        return None
    Xm = model_lib._apply_feature_mask(X, cfg.get("drop_features", []))
    try:
        return np.asarray(model.predict_proba(Xm))
    except Exception:
        return None


@dataclass
class WindowRecord:
    window: int
    name: str
    true_score: float
    model_version: int
    drift_level: str
    action: str
    drift_type: str
    promoted: bool
    rollback: bool
    minutes: float
    n_trials: int
    note: str = ""
    incident: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "window": self.window,
            "name": self.name,
            "true_score": None if self.true_score is None else round(float(self.true_score), 4),
            "model_version": self.model_version,
            "drift_level": self.drift_level,
            "action": self.action,
            "drift_type": self.drift_type,
            "promoted": self.promoted,
            "rollback": self.rollback,
            "minutes": round(self.minutes, 3),
            "n_trials": self.n_trials,
            "note": self.note,
            "incident": self.incident,
        }


@dataclass
class RunResult:
    policy: str
    records: List[WindowRecord] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    events: List[Dict[str, Any]] = field(default_factory=list)

    def scores(self) -> List[Optional[float]]:
        return [r.true_score for r in self.records]


class ClosedLoop:
    def __init__(
        self,
        task: TaskSpec,
        windows: List[Window],
        run_dir: Path,
        run_id: str,
        seed: int = 42,
        gate_cfg: Optional[GateConfig] = None,
        monitor_cfg: Optional[Dict[str, Any]] = None,
        diagnose_cfg: Optional[Dict[str, Any]] = None,
        use_simulator: bool = True,
        sim_n_per_window: int = 250,
        sim_weight: float = 0.35,
    ) -> None:
        self.task = task
        self.windows = windows
        self.run_dir = Path(run_dir)
        self.run_id = run_id
        self.seed = seed
        self.gate_cfg = gate_cfg or GateConfig()
        self.monitor = DriftMonitor(monitor_cfg)
        self.diagnoser = Diagnoser(diagnose_cfg)
        self.safety = SafeGate(task_type=self.task.task_type)
        self.sealer = Sealer(self.run_dir)
        self.kinds = [str(w.meta.get("kind", "unknown")) for w in windows]
        self.buffer: List[Dict[str, Any]] = []
        self.val_history: List[Dict[str, np.ndarray]] = []
        self.total_minutes = 0.0
        self.trace: Optional[Trace] = None
        self.rollback_count = 0

        self.use_simulator = use_simulator
        self.sim_n = int(sim_n_per_window)
        self.sim_weight = float(sim_weight)
        self.simulator = Simulator(task)
        self.sim_pool: List[Dict[str, Any]] = []
        self.drift_history: List[Dict[str, Any]] = []
        self.n_synthetic = 0
        self.sim_minutes = 0.0

    # ------------------------------------------------------------------
    def run(self) -> RunResult:
        res = RunResult(policy="closed_loop")
        self.trace = Trace(
            self.run_dir / "trace.jsonl",
            self.run_id,
            meta={
                "policy": "closed_loop",
                "task": self.task.name,
                "n_windows": len(self.windows),
                "seed": self.seed,
                "started_at": datetime.now().isoformat(timespec="seconds"),
            },
        )
        rng = np.random.default_rng(self.seed)

        # ---- 初始部署：首窗训练 + 封版 v0001
        w0 = self.windows[0]
        split0 = _split_window(w0, self.seed)
        cfg = model_lib.default_config(self.task)
        model = model_lib.fit(cfg, self.task, split0["X_fit"], split0["y_fit"], seed=self.seed)
        incumbent = Version(vid=1, cfg=cfg, model=model, created_at=datetime.now().isoformat(timespec="seconds"),
                            note="初始部署（首窗训练）")
        pred_tr, _ = model_lib.safe_predict(cfg, model, split0["X_fit"], self.task)
        loss0 = self.task.loss(split0["y_fit"], pred_tr)
        incumbent.metrics = {"train_loss": round(float(loss0), 4)}
        self.sealer.seal(incumbent, self.task, self.windows, extra={"stage": "initial-deploy"})
        self.trace.event(step="seal", event="initial-version", window=0,
                         payload=incumbent.to_dict())
        incumbent = Version(vid=1, cfg=cfg, model=model, metrics=incumbent.metrics,
                            created_at=incumbent.created_at, sealed=True, note="初始部署")

        self.monitor.set_reference(split0["X_fit"], pred_tr, loss0, split0["y_fit"])
        self.diagnoser.set_reference(split0["X_fit"], split0["y_fit"],
                                     split0["y_fit"] - pred_tr)
        self.safety.set_reference(split0["X_fit"], split0["y_fit"])
        self.buffer.append({"window": 0, "X": split0["X_fit"], "y": split0["y_fit"]})
        self.val_history.append({"X": split0["X_val"], "y": split0["y_val"]})

        prev_failed = False
        reject_streak = 0
        stable_streak = 0
        next_vid = 2

        for w in self.windows:
            t_win = time.perf_counter()
            split = _split_window(w, self.seed)

            # ===== 第 1 步：监测（只用在线可得的训练数据 + 在位模型）
            pred_inc, rep_safe = model_lib.safe_predict(incumbent.cfg, incumbent.model, split["X_fit"], self.task)
            loss_cur = float(self.task.loss(split["y_fit"], pred_inc))
            # ===== 安全拒绝机制：输入越界/置信度不足/输出越界三重拒答（留痕可审计）
            proba_inc = _proba_of(incumbent.cfg, incumbent.model, split["X_fit"], self.task)
            safety_rep = self.safety.check(split["X_fit"], pred_inc, proba_inc)
            if safety_rep.refused:
                self.trace.event(step="safety", event="refusal", window=w.index,
                                 payload=safety_rep.to_dict())
            if w.index == 0:
                drift = None
                action_cfg = {
                    "action": "hold", "drift_type": "none", "confidence": 0.9,
                    "reason": "首窗建立参考期", "trials": 0, "minutes": 0.0,
                    "drop_features": [], "recency_half_life": 4.0, "attributions": [],
                }
                drift_level = "none"
            else:
                drift = self.monitor.check(w.index, split["X_fit"], split["y_fit"], pred_inc, loss_cur)
                self.trace.event(step="monitor", event="drift-check", window=w.index,
                                 payload=drift.to_dict())
                # ===== 第 2 步：诊断归因
                residual_cur = split["y_fit"] - pred_inc
                strat = self.diagnoser.diagnose(drift, split["X_fit"], split["y_fit"], residual_cur,
                                                previous_attempt_failed=prev_failed)
                action_cfg = strat.to_dict()
                drift_level = drift.level
                if drift.level != "none":
                    self.drift_history.append(
                        {"window": w.index, "drift_type": action_cfg["drift_type"],
                         "covariate": drift.covariate_score, "concept": drift.concept_score,
                         "level": drift.level, "top_features": [f for f, _ in drift.top_features]}
                    )
                self.trace.event(step="diagnose", event="strategy", window=w.index,
                                 payload=action_cfg)

            action = action_cfg["action"]
            minutes = 0.0
            n_trials = 0
            promoted = False
            rolled_back = False
            note = ""

            # ===== 故障分类归纳 + 优先级评分（同时面对多个突发时，先处理最高优先级）=====
            lat_now = model_lib.Latency()
            lat_now.measure(incumbent.cfg, incumbent.model, split["X_fit"][: min(200, len(split["X_fit"]))], self.task)
            budget_frac = max(0.0, (MAX_TOTAL_MINUTES - self.total_minutes) / MAX_TOTAL_MINUTES)
            incident_signals = incident_lib.signals_from_state(
                drift=drift,
                nan_ratio=float(rep_safe.get("nan_ratio", 0.0)),
                latency_ms=lat_now.p95(),
                budget_frac=budget_frac,
                config_violations=0,
            )
            incidents = incident_lib.assess(incident_signals)
            top = incident_lib.top_priority(incidents)
            self.trace.event(step="incident", event="assess", window=w.index,
                             payload={"incidents": [i.to_dict() for i in incidents]})

            incident_summary = "无"
            if top is not None:
                incident_summary = f"{top.label}·{top.severity}·{top.score:.0f}分"
                if top.fault_class == "safety" and top.severity in ("critical", "high"):
                    action = "halt"
                    note = f"安全故障优先：{top.label}——熔断/拒答，冻结上线"
                elif top.fault_class == "system":
                    if action in ("retune", "retune_reweight", "escalate"):
                        action = "retrain"
                    note = f"系统故障降级：{top.label}——压缩预算，先保证可用"
                elif top.fault_class == "data_quality":
                    action = "quarantine"
                    note = f"数据质量故障：{top.label}——隔离该包"

            # ===== 第 3 步：优化
            if action in ("hold", "quarantine", "halt"):
                note = note or ("维持/隔离，不消耗算力" if action != "halt" else "安全熔断，不消耗算力")
                # ---- 数字孪生预演：仅当"无漂移"且未被隔离/熔断时，用空闲算力主动合成未来工况
                if self.use_simulator and action == "hold":
                    self._run_simulator(w, split)
                self.trace.event(step="optimize", event="skipped", window=w.index,
                                 payload={"action": action, "incident": incident_summary,
                                          "sim_pool_size": len(self.sim_pool)})
                cand = None
            else:
                # 监测到真实漂移：冻结仿真，把全部算力转给优化器
                if self.use_simulator:
                    self.trace.event(step="simulator", event="freeze-on-drift", window=w.index,
                                     payload={"reason": "真实漂移优先，仿真场暂停，算力转投优化"})
                budget_trials = int(action_cfg["trials"])
                budget_minutes = float(action_cfg["minutes"])
                # ===== 成本收益自适应：算力投入与漂移严重度/失败次数/剩余预算挂钩
                bfrac = max(0.0, (MAX_TOTAL_MINUTES - self.total_minutes) / MAX_TOTAL_MINUTES)
                bscale = budget_scaler(drift, reject_streak, bfrac)
                budget_trials = max(1, int(round(budget_trials * bscale)))
                budget_minutes = budget_minutes * bscale
                self.trace.event(step="budget", event="adaptive", window=w.index,
                                 payload={"scale": round(bscale, 3), "trials": budget_trials,
                                          "minutes": round(budget_minutes, 2),
                                          "budget_left": round(bfrac, 3)})
                buf = [b for b in self.buffer]
                # 预适应缓冲：把仿真场合成样本也纳入训练（权重低于真实样本）
                buf.extend(self.sim_pool)
                # 当前窗只有在"有效数据包"时才进入训练
                if drift is not None and drift.empty_packet:
                    note = "空跑包不入训练缓冲"
                else:
                    buf.append({"window": w.index, "X": split["X_fit"], "y": split["y_fit"]})

                opt = optimize(
                    self.task, buf, w.index, incumbent.cfg, action_cfg,
                    budget_trials=budget_trials, budget_minutes=budget_minutes,
                    seed=self.seed, total_minutes_left=max(0.0, MAX_TOTAL_MINUTES - self.total_minutes),
                )
                minutes = float(opt.minutes_used)
                n_trials = len(opt.trials)
                self.total_minutes += minutes
                for trow in opt.trials:
                    self.trace.event(step="optimize", event="trial", window=w.index, payload=trow.to_dict())
                self.trace.event(step="optimize", event="result", window=w.index, payload=opt.to_dict())

                # ===== 第 4 步：上线评估
                cand_cfg = opt.champion
                cand_cfg["version"] = next_vid
                X_all = np.vstack([b["X"] for b in buf])
                y_all = np.concatenate([b["y"] for b in buf])
                w_all = combine_buffer(buf, float(cand_cfg.get("recency_half_life", 4.0) or 4.0), w.index)[2]
                cand_model = model_lib.fit(cand_cfg, self.task, X_all, y_all, w_all,
                                           seed=self.seed + w.index)
                cand = Version(vid=next_vid, cfg=cand_cfg, model=cand_model,
                               created_at=datetime.now().isoformat(timespec="seconds"),
                               note=f"{action_cfg['drift_type']}漂移处置")

                val_sets = [("当前窗留出集", split["X_val"], split["y_val"])]
                if len(self.val_history) >= 1:
                    vh = self.val_history[-1]
                    val_sets.append(("上一窗留出集", vh["X"], vh["y"]))
                if len(self.val_history) >= 2:
                    vh = self.val_history[-2]
                    val_sets.append(("更早留出集", vh["X"], vh["y"]))

                verdict = evaluate_pair(self.task, cand, incumbent, val_sets, self.gate_cfg, rng)
                self.trace.event(step="gate", event="verdict", window=w.index, payload=verdict)

                if verdict["decision"] == "promote":
                    promoted = True
                    incumbent = cand
                    incumbent.sealed = False
                    manifest = self.sealer.seal(
                        incumbent, self.task, self.windows,
                        extra={"window": w.index, "action": action, "verdict": verdict},
                    )
                    incumbent = Version(vid=next_vid, cfg=cand.cfg, model=cand.model,
                                        metrics={"gate_score": verdict["composite_score"]},
                                        created_at=manifest["created_at"], sealed=True,
                                        note=cand.note)
                    next_vid += 1
                    prev_failed = False
                    reject_streak = 0
                    stable_streak = 0
                    self.monitor.set_reference(split["X_fit"], pred_inc, loss_cur, split["y_fit"])
                    self.diagnoser.set_reference(split["X_fit"], split["y_fit"], split["y_fit"] - pred_inc)
                    note = f"晋升 v{incumbent.vid}：" + str(verdict["reasons"][0])
                else:
                    prev_failed = True
                    reject_streak += 1
                    rolled_back = True
                    self.rollback_count += 1
                    note = f"回滚至 v{incumbent.vid}：" + "；".join(verdict["reasons"])
                    self.trace.event(step="gate", event="rollback", window=w.index,
                                     payload={"rollback_to": incumbent.vid, "reasons": verdict["reasons"],
                                              "reject_streak": reject_streak})
                    # 连续两窗处置失败 → 保持最近封版并升级处置强度（diagnoser 已读 prev_failed 升级）
                    if reject_streak >= 2:
                        self.trace.event(step="gate", event="rollback-to-sealed", window=w.index,
                                         payload={"version": incumbent.vid, "reason": "连续处置未通过门槛"})
                        reject_streak = 0

            # 缓冲更新（空跑包不进入）
            if drift is None or not drift.empty_packet:
                if w.index > 0:
                    self.buffer.append({"window": w.index, "X": split["X_fit"], "y": split["y_fit"]})
            self.buffer = self.buffer[-MAX_BUFFER_WINDOWS:]
            self.val_history.append({"X": split["X_val"], "y": split["y_val"]})
            self.val_history = self.val_history[-3:]

            # 长期稳定时刷新监测参考期（避免参考期过老导致"永远报警"）
            if drift is not None and drift.level == "none":
                stable_streak += 1
                if stable_streak >= 2:
                    pred_now, _ = model_lib.safe_predict(incumbent.cfg, incumbent.model, split["X_fit"], self.task)
                    loss_now = float(self.task.loss(split["y_fit"], pred_now))
                    self.monitor.set_reference(split["X_fit"], pred_now, loss_now, split["y_fit"])
                    self.trace.event(step="monitor", event="reference-refresh", window=w.index, payload={})
                    stable_streak = 0
            else:
                stable_streak = 0

            # ===== 外部评测（y_eval 只在裁判侧使用，引擎不读）
            scored_model = incumbent
            pred_eval, _ = model_lib.safe_predict(scored_model.cfg, scored_model.model, w.X_eval, self.task)
            true_score = float(self.task.score(w.y_eval, pred_eval))

            rec = WindowRecord(
                window=w.index, name=w.name, true_score=true_score,
                model_version=incumbent.vid, drift_level=drift_level, action=action,
                drift_type=action_cfg["drift_type"], promoted=promoted,
                rollback=rolled_back,
                minutes=minutes, n_trials=n_trials, note=note,
                incident=incident_summary,
            )
            res.records.append(rec)
            self.trace.event(step="eval", event="window-score", window=w.index,
                             payload={"true_score": round(true_score, 4), "version": incumbent.vid,
                                      "wall_s": round(time.perf_counter() - t_win, 3)})

        cost = {
            "minutes": round(self.total_minutes, 2),
            "trials": int(sum(r.n_trials for r in res.records)),
            "promotions": int(sum(1 for r in res.records if r.promoted)),
            "rollbacks": int(self.rollback_count),
            "n_synthetic": self.n_synthetic,
            "sim_minutes": round(self.sim_minutes, 3),
            "use_simulator": self.use_simulator,
        }
        res.metrics = summarize(res.scores(), self.kinds, cost,
                                {"policy": "closed_loop", "use_simulator": self.use_simulator,
                                 "n_synthetic": self.n_synthetic})
        self.trace.event(step="run", event="metrics", payload=res.metrics)
        self.trace.close({"metrics": res.metrics})
        return res

    # ------------------------------------------------------------------
    def _run_simulator(self, w: Window, split: Dict[str, np.ndarray]) -> None:
        """无漂移时的空闲算力 → 主动预演未来工况，丰富预适应样本库。"""
        real_buf = [b for b in self.buffer if "w" not in b]
        if len(real_buf) < 1:
            return
        X_ref = np.vstack([b["X"] for b in real_buf])
        y_ref = np.concatenate([b["y"] for b in real_buf])

        # 代理模型学到的是"当前工况"，每次稳定期刷新（或首次）重建
        if self.simulator.surrogate is None or self.simulator.fitted_on != f"w{w.index}":
            self.simulator.fit(X_ref, y_ref, tag=f"w{w.index}")

        std = np.nanstd(X_ref, axis=0)
        rng = np.random.default_rng(self.seed + w.index * 31 + 7)
        scenarios = self.simulator.sample_scenarios(
            X_ref.shape[1], rng, history=self.drift_history, feature_std=std, n=2
        )
        generated = 0
        for sc in scenarios:
            out = self.simulator.generate_labeled(X_ref, y_ref, sc, max(50, self.sim_n // 2), rng)
            if out is None:
                continue
            Xs, ys = out
            # 权重低于真实样本：合成样本是"可能发生"，不是"已经发生"
            self.sim_pool.append({"X": Xs, "y": ys, "w": self.sim_weight})
            generated += len(ys)
            self.trace.event(step="simulator", event="scenario", window=w.index, payload=sc.to_dict())
        # 池子有上限，超出丢最旧的，避免无限膨胀
        cap = 6
        while len(self.sim_pool) > cap:
            self.sim_pool.pop(0)
        self.n_synthetic += generated
        self.sim_minutes += 0.02
        self.trace.event(step="simulator", event="batch", window=w.index,
                         payload={"n_generated": generated, "pool_size": len(self.sim_pool)})


# ---------------------------------------------------------------------------
# 对照策略
# ---------------------------------------------------------------------------


def run_static(task: TaskSpec, windows: List[Window], seed: int = 42) -> RunResult:
    """静态基线：首窗训练一次，永不更新。这是赛题要求"跑通基线"的对照物。"""
    res = RunResult(policy="static")
    split0 = _split_window(windows[0], seed)
    cfg = model_lib.default_config(task)
    model = model_lib.fit(cfg, task, split0["X_fit"], split0["y_fit"], seed=seed)
    for w in windows:
        pred, _ = model_lib.safe_predict(cfg, model, w.X_eval, task)
        sc = float(task.score(w.y_eval, pred))
        res.records.append(WindowRecord(w.index, w.name, sc, 1, "n/a", "hold", "none", False, False, 0.0, 0,
                                        "静态模型"))
    res.metrics = summarize(res.scores(), [str(w.meta.get("kind", "")) for w in windows],
                            {"minutes": 0.0, "trials": 1, "promotions": 0}, {"policy": "static"})
    return res


def run_periodic(task: TaskSpec, windows: List[Window], seed: int = 42) -> RunResult:
    """定期重训：每窗都用固定超参重训，不做漂移诊断也不调参。"""
    res = RunResult(policy="periodic")
    cfg = model_lib.default_config(task)
    buf = []
    model = None
    spent = 0.0
    for w in windows:
        split = _split_window(w, seed)
        is_empty = str(w.meta.get("kind", "")) == "empty"
        if not is_empty:
            buf.append({"window": w.index, "X": split["X_fit"], "y": split["y_fit"]})
            buf = buf[-MAX_BUFFER_WINDOWS:]
        if model is None or not is_empty:
            X = np.vstack([b["X"] for b in buf])
            y = np.concatenate([b["y"] for b in buf])
            model = model_lib.fit(cfg, task, X, y, seed=seed)
            spent += 0.6
        pred, _ = model_lib.safe_predict(cfg, model, w.X_eval, task)
        sc = float(task.score(w.y_eval, pred))
        res.records.append(WindowRecord(w.index, w.name, sc, 1, "n/a", "retrain", "none", False, False,
                                        0.0 if is_empty else 0.6, 0,
                                        "定期重训（固定超参）"))
    res.metrics = summarize(res.scores(), [str(w.meta.get("kind", "")) for w in windows],
                            {"minutes": round(spent, 2), "trials": len(windows) - 1,
                             "promotions": len(windows) - 1}, {"policy": "periodic"})
    return res


def run_retune_all(task: TaskSpec, windows: List[Window], seed: int = 42) -> RunResult:
    """无差别全量重调参：每窗都跑满预算，不看有没有漂移（成本对照）。"""
    res = RunResult(policy="retune_all")
    cfg = model_lib.default_config(task)
    split0 = _split_window(windows[0], seed)
    model = model_lib.fit(cfg, task, split0["X_fit"], split0["y_fit"], seed=seed)
    buf = [{"window": 0, "X": split0["X_fit"], "y": split0["y_fit"]}]
    spent = 0.0
    trials_total = 0
    for w in windows:
        split = _split_window(w, seed)
        is_empty = str(w.meta.get("kind", "")) == "empty"
        window_minutes = 0.0
        window_trials = 0
        if w.index > 0 and not is_empty:
            buf.append({"window": w.index, "X": split["X_fit"], "y": split["y_fit"]})
            buf = buf[-MAX_BUFFER_WINDOWS:]
            opt = optimize(task, buf, w.index, cfg, {"recency_half_life": 4.0, "drop_features": []},
                           budget_trials=10, budget_minutes=10.0, seed=seed)
            spent += opt.minutes_used
            window_minutes = opt.minutes_used
            window_trials = len(opt.trials)
            trials_total += window_trials
            cfg = opt.champion
            X = np.vstack([b["X"] for b in buf])
            y = np.concatenate([b["y"] for b in buf])
            model = model_lib.fit(cfg, task, X, y, seed=seed + w.index)
        pred, _ = model_lib.safe_predict(cfg, model, w.X_eval, task)
        res.records.append(WindowRecord(w.index, w.name, float(task.score(w.y_eval, pred)), 1,
                                        "n/a", "retune", "none", w.index > 0, False,
                                        window_minutes, window_trials, "无差别重调参"))
    res.metrics = summarize(res.scores(), [str(w.meta.get("kind", "")) for w in windows],
                            {"minutes": round(spent, 2), "trials": trials_total,
                             "promotions": len(windows) - 1}, {"policy": "retune_all"})
    return res
