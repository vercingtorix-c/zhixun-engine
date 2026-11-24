"""官方包续训五步闭环。

对齐官方《赛题规则》与 TASK.md / TRAINING_GUIDE.md：
  漂移监测 → 诊断归因 → 实验优化（续训 + 小范围调 lr/epochs）→ 评估上线（回滚/晋升）→ 封版

与自建 sklearn 闭环（engine/loop.py）的关键差异：
  - 模型是官方 NumPy MLP，从 checkpoint.npz 续训，不重新初始化；
  - 训练只读取 public/train.jsonl，validation.jsonl 仅作最终干净评测，不进入训练；
  - 主指标按官方口径：A=Macro-F1，B/C=RMSE（附 MAE，C 另算 NASA 非对称分）。
"""

from __future__ import annotations

import copy
import hashlib
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from . import model as om
from .loader import PROFILES, better, build_windows, official_metrics
from .. import incident as incident_lib
from ..diagnose import Diagnoser, budget_scaler
from ..monitor import DriftMonitor
from ..safety import SafeGate
from ..simulator import Simulator
from ..trace import Trace
from ..types import TaskSpec


MAX_TOTAL_EPOCHS = 60.0


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _split(X: np.ndarray, y: np.ndarray, index: int, seed: int, val_frac: float = 0.25) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed + index)
    n = len(y)
    idx = rng.permutation(n)
    n_val = max(1, int(n * val_frac))
    va, tr = idx[:n_val], idx[n_val:]
    return X[tr], y[tr], X[va], y[va]


def _loss_of(profile: str, model, X: np.ndarray, y: np.ndarray) -> float:
    m = official_metrics(profile, model, X, y)
    if profile == "A":
        return float(1.0 - m["macro_f1"])
    return float(m["rmse"])


def _residual_of(profile: str, model, X: np.ndarray, y: np.ndarray) -> np.ndarray:
    pred = model.predict(X)
    if profile == "A":
        labels = pred.argmax(axis=1)
        return (labels != y).astype(np.float64)
    return (y.astype(np.float64) - pred.ravel()).astype(np.float64)


def _candidate_specs(action: str, base_lr: float, max_candidates: int) -> List[Tuple[float, int]]:
    """把诊断动作映射成 (学习率, epochs) 续训候选，控制算力成本。"""
    if action in ("retrain", "recalibrate"):
        specs = [(base_lr, 1), (base_lr * 0.5, 1)]
    elif action == "escalate":
        specs = [(base_lr * 0.1, 2), (base_lr * 0.5, 3), (base_lr, 3)]
    else:  # retune / retune_reweight / concept / mixed
        specs = [(base_lr * 0.1, 1), (base_lr * 0.5, 1), (base_lr, 2), (base_lr * 0.5, 2)]
    return specs[:max_candidates]


def _simulate_future(profile: str, model, X_ref: np.ndarray, y_ref: np.ndarray,
                     history: List[Dict[str, Any]], seed: int, sim_n: int,
                     n_scenarios: int = 2) -> List[Dict[str, Any]]:
    """数字孪生仿真场：无漂移时合成"未来可能工况"。

    关键合规点：官方要求「梯度更新只用 public/train.jsonl」，所以合成样本
    **只用于候选模型的抗漂移稳健性评估（gate），绝不进入梯度更新**。
    """
    if len(X_ref) < 20:
        return []
    std = np.nanstd(X_ref, axis=0)
    std = np.where(std < 1e-9, 1.0, std)
    scale = float(np.std(y_ref)) if np.std(y_ref) > 1e-9 else 1.0
    task = TaskSpec(
        name=f"official-{profile}",
        task_type="classification" if profile == "A" else "regression",
        n_classes=len(PROFILES[profile]["classes"]) if profile == "A" else 0,
        scale=scale,
    )
    sim = Simulator(task)
    rng = np.random.default_rng(seed + 1000 + len(history))
    scenarios = sim.sample_scenarios(X_ref.shape[1], rng, history=history,
                                     feature_std=std, n=n_scenarios)
    pool: List[Dict[str, Any]] = []
    for sc in scenarios:
        idx = rng.integers(0, len(X_ref), size=sim_n)
        base = np.asarray(X_ref[idx], dtype=np.float64)
        X_sim = base * sc.gain + sc.offset
        if profile == "A":
            # 分类：只做协变量增强，标签沿用抽样行真实标签，不臆造标签翻转
            y_sim = np.asarray(y_ref[idx])
        else:
            # 回归：当前官方模型预测 + 概念关系扰动 + 噪声（self-training 式）
            pred = np.asarray(model.predict(base)).ravel()
            concept = np.zeros(sim_n)
            for f in sc.features:
                if abs(float(sc.concept_coef[f])) > 1e-9:
                    concept += float(sc.concept_coef[f]) * X_sim[:, f]
            y_sim = pred + concept + rng.normal(0.0, 0.35 * scale, size=sim_n)
            y_sim = np.maximum(0.0, y_sim)
        pool.append({"X": X_sim, "y": y_sim, "w": 0.5, "scenario": sc.to_dict()})
    return pool


def run_official(
    profile: str,
    data_dir: Path,
    out_dir: Path,
    seed: int = 42,
    n_windows: int = 10,
    max_candidates: int = 6,
    max_epochs_per_window: int = 4,
) -> Dict[str, Any]:
    if profile not in PROFILES:
        raise ValueError(f"unknown profile {profile}")
    data_dir = Path(data_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_path = data_dir / "public" / "train.jsonl"
    ckpt_path = data_dir / "checkpoint.npz"
    windows = build_windows(profile, train_path, n_windows=n_windows)

    incumbent = om.Model.load(ckpt_path)
    base_lr = float(incumbent.meta["learning_rate"])

    run_id = f"official-{profile}-{datetime.now().strftime('%Y%m%d%H%M%S')}"
    trace = Trace(
        out_dir / "trace.jsonl",
        run_id,
        meta={
            "policy": "official-continue-train",
            "profile": profile,
            "task": incumbent.meta["task"],
            "n_windows": len(windows),
            "seed": seed,
            "parent_checkpoint": _sha256(ckpt_path),
            "started_at": datetime.now().isoformat(timespec="seconds"),
        },
    )

    monitor = DriftMonitor()
    diagnoser = Diagnoser()
    safety = SafeGate(task_type="classification" if profile == "A" else "regression")
    versions_dir = out_dir / "versions"
    versions_dir.mkdir(parents=True, exist_ok=True)
    chain: List[Dict[str, Any]] = []

    def seal(model, vid: int, note: str, parent: Optional[str]) -> Dict[str, Any]:
        d = versions_dir / f"v{vid:04d}"
        d.mkdir(parents=True, exist_ok=True)
        model.save(d / "checkpoint.npz")
        ck_sha = _sha256(d / "checkpoint.npz")
        manifest = {
            "format": "zhixun-seal.v1",
            "version": vid,
            "note": note,
            "parent_checkpoint_sha256": parent,
            "checkpoint_sha256": ck_sha,
            "seed": model.meta["seed"],
            "steps": model.meta["steps"],
            "epochs": model.meta["epochs"],
            "learning_rate": model.meta["learning_rate"],
            "parameter_count": model.meta["parameter_count"],
            "sealed_at": datetime.now().isoformat(timespec="seconds"),
        }
        (d / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
        chain.append(manifest)
        return manifest

    # 初始封版：官方基线检查点
    parent = _sha256(ckpt_path)
    seal(incumbent, 1, "官方基线检查点（续训起点）", None)
    trace.event(step="seal", event="initial-version", window=0, payload={"version": 1, "checkpoint": parent})
    vid_next = 2

    # 参考期：首窗
    w0 = windows[0]
    Xf0, yf0, Xv0, yv0 = _split(w0["X"], w0["y"], 0, seed)
    pred0 = incumbent.predict(Xf0)
    loss0 = _loss_of(profile, incumbent, Xf0, yf0)
    monitor.set_reference(Xf0, pred0, loss0, yf0)
    diagnoser.set_reference(Xf0, yf0, _residual_of(profile, incumbent, Xf0, yf0))
    safety.set_reference(Xf0, yf0)

    prev_failed = False
    reject_streak = 0
    records: List[Dict[str, Any]] = []
    sim_pool: List[Dict[str, Any]] = []
    drift_history: List[Dict[str, Any]] = []
    total_epochs = 0
    total_wall = 0.0

    for w in windows:
        t0 = time.perf_counter()
        Xf, yf, Xv, yv = _split(w["X"], w["y"], w["index"], seed)
        pred_inc = incumbent.predict(Xf)
        loss_cur = _loss_of(profile, incumbent, Xf, yf)
        # ===== 安全拒绝机制：输入越界/置信度不足/输出越界三重拒答（留痕可审计）
        proba_inc = pred_inc if profile == "A" else None
        safety_rep = safety.check(Xf, pred_inc, proba_inc)
        if safety_rep.refused:
            trace.event(step="safety", event="refusal", window=w["index"],
                        payload=safety_rep.to_dict())

        if w["index"] == 0:
            drift = None
            action, drift_type, note = "hold", "none", "首窗建立参考期"
        else:
            drift = monitor.check(w["index"], Xf, yf, pred_inc, loss_cur)
            trace.event(step="monitor", event="drift-check", window=w["index"], payload=drift.to_dict())
            residual = _residual_of(profile, incumbent, Xf, yf)
            strat = diagnoser.diagnose(drift, Xf, yf, residual, previous_attempt_failed=prev_failed)
            action, drift_type, note = strat.action, strat.drift_type, strat.reason
            trace.event(step="diagnose", event="strategy", window=w["index"], payload=strat.to_dict())
            if drift is not None and drift.level != "none":
                drift_history.append({"window": w["index"], "drift_type": drift_type,
                                      "level": drift.level})

        # ===== 故障分类归纳 + 优先级评分（多突发并发时按优先级处理）=====
        t_p = time.perf_counter()
        _ = incumbent.predict(Xf[: min(200, len(Xf))])
        lat_ms = (time.perf_counter() - t_p) * 1000.0
        nan_ratio = float(np.mean(~np.isfinite(pred_inc))) if pred_inc.size else 0.0
        budget_frac = max(0.0, (MAX_TOTAL_EPOCHS - total_epochs) / MAX_TOTAL_EPOCHS)
        inc_signals = incident_lib.signals_from_state(
            drift=drift, nan_ratio=nan_ratio, latency_ms=lat_ms,
            budget_frac=budget_frac, config_violations=0)
        incidents = incident_lib.assess(inc_signals)
        top = incident_lib.top_priority(incidents)
        trace.event(step="incident", event="assess", window=w["index"],
                    payload={"incidents": [i.to_dict() for i in incidents]})
        incident_summary = "无"
        if top is not None:
            incident_summary = f"{top.label}·{top.severity}·{top.score:.0f}分"
            if top.fault_class == "safety" and top.severity in ("critical", "high"):
                action = "halt"
                note = f"安全故障优先：{top.label}——熔断/拒答"
            elif top.fault_class == "system":
                if action in ("retune", "retune_reweight", "escalate"):
                    action = "retrain"
                note = f"系统故障降级：{top.label}——压缩预算"
            elif top.fault_class == "data_quality":
                action = "quarantine"
                note = f"数据质量故障：{top.label}——隔离该包"

        promoted = False
        rolled_back = False
        epochs_spent = 0
        n_candidates = 0
        best_metric: Optional[Dict[str, float]] = None

        if action in ("hold", "quarantine", "halt"):
            note = note or ("维持/隔离，不消耗算力" if action != "halt" else "安全熔断，不消耗算力")
            # ---- 数字孪生预演：仅当"无漂移"且未被隔离/熔断时，用空闲算力合成未来工况
            if action == "hold":
                gen = _simulate_future(profile, incumbent, Xf, yf, drift_history,
                                       seed + w["index"] * 7, max(60, len(Xf)), n_scenarios=2)
                for g in gen:
                    sim_pool.append(g)
                while len(sim_pool) > 8:
                    sim_pool.pop(0)
                if gen:
                    trace.event(step="simulator", event="preplay", window=w["index"],
                                payload={"n_generated": int(sum(len(g["y"]) for g in gen)),
                                         "pool_size": len(sim_pool),
                                         "scenarios": [g["scenario"] for g in gen]})
            trace.event(step="optimize", event="skipped", window=w["index"],
                        payload={"action": action, "sim_pool_size": len(sim_pool),
                                 "incident": incident_summary})
        else:
            # 续训候选：小范围 (lr, epochs)，全部从同一父检查点分支
            specs = _candidate_specs(action, base_lr, max_candidates)
            # ===== 成本收益自适应：连续失败加码 + 剩余预算收缩（严重度已由 action 体现）
            bscale = float(np.clip(budget_scaler(drift, reject_streak, budget_frac,
                                                 include_severity=False), 0.6, 1.5))
            specs = [(lr, max(1, int(round(ep * bscale)))) for lr, ep in specs]
            trace.event(step="budget", event="adaptive", window=w["index"],
                        payload={"scale": round(bscale, 3), "reject_streak": reject_streak,
                                 "budget_left": round(budget_frac, 3)})
            inc_val = official_metrics(profile, incumbent, Xv, yv)
            best = None
            for lr, ep in specs:
                cand = copy.deepcopy(incumbent)
                cand.meta["learning_rate"] = lr
                cand.fit(Xf, yf, ep)
                m = official_metrics(profile, cand, Xv, yv)
                n_candidates += 1
                epochs_spent += ep
                trace.event(step="optimize", event="candidate", window=w["index"],
                            payload={"lr": lr, "epochs": ep, "metric": m})
                if best is None or better(profile, m, best["metric"]):
                    best = {"model": cand, "metric": m, "lr": lr, "epochs": ep}

            # ---- 数字孪生稳健性门槛：候选在合成"未来工况"上不得显著退化（只评估，不进梯度）
            sim_ok = True
            if best is not None and sim_pool:
                Xs = np.vstack([g["X"] for g in sim_pool])
                ys = np.concatenate([g["y"] for g in sim_pool])
                inc_sim = official_metrics(profile, incumbent, Xs, ys)
                cand_sim = official_metrics(profile, best["model"], Xs, ys)
                if profile == "A":
                    sim_ok = cand_sim["macro_f1"] >= inc_sim["macro_f1"] * 0.97
                else:
                    sim_ok = cand_sim["rmse"] <= inc_sim["rmse"] * 1.03
                trace.event(step="gate", event="sim-robustness", window=w["index"],
                            payload={"incumbent": inc_sim, "candidate": cand_sim, "pass": bool(sim_ok)})

            if best is not None and better(profile, best["metric"], inc_val) and sim_ok:
                incumbent = best["model"]
                base_lr = best["lr"]
                promoted = True
                prev_failed = False
                reject_streak = 0
                best_metric = best["metric"]
                manifest = seal(incumbent, vid_next, f"{drift_type}漂移续训", chain[-1]["checkpoint_sha256"])
                trace.event(step="gate", event="promote", window=w["index"], payload=manifest)
                vid_next += 1
                monitor.set_reference(Xf, pred_inc, loss_cur, yf)
                diagnoser.set_reference(Xf, yf, _residual_of(profile, incumbent, Xf, yf))
                note = f"晋升 v{manifest['version']}（续训 {best['epochs']} epoch, lr={best['lr']}）"
            else:
                rolled_back = True
                prev_failed = True
                reject_streak += 1
                if best is None or not better(profile, best["metric"], inc_val):
                    note = f"回滚至 v{vid_next - 1}：续训候选未优于在位模型"
                else:
                    note = f"回滚至 v{vid_next - 1}：候选未通过数字孪生抗漂移门槛"
                trace.event(step="gate", event="rollback", window=w["index"],
                            payload={"rollback_to": vid_next - 1, "best": best["metric"] if best else None,
                                     "incumbent": inc_val, "sim_ok": bool(sim_ok)})

        total_epochs += epochs_spent
        wall = time.perf_counter() - t0
        total_wall += wall

        # 逐窗开发得分（用本窗留出的 gate-val，仅作曲线展示；正式干净评测见 validation.jsonl）
        score_m = official_metrics(profile, incumbent, Xv, yv)
        primary = score_m.get("macro_f1", score_m.get("rmse"))
        rec = {
            "window": w["index"],
            "name": w["name"],
            "n_train": int(len(Xf)),
            "n_units": w["n_units"],
            "action": action,
            "drift_type": drift_type,
            "promoted": promoted,
            "rollback": rolled_back,
            "primary_metric": None if primary is None else round(float(primary), 4),
            "metrics": {k: round(float(v), 4) for k, v in score_m.items()},
            "epochs_spent": epochs_spent,
            "n_candidates": n_candidates,
            "wall_seconds": round(wall, 3),
            "incident": incident_summary,
            "sim_pool_size": len(sim_pool),
            "note": note,
        }
        records.append(rec)
        trace.event(step="eval", event="window-score", window=w["index"], payload=rec)

    # 最终干净评测：validation.jsonl（从未进入训练/门槛）
    val_report: Optional[Dict[str, Any]] = None
    val_path = data_dir / "public" / "validation.jsonl"
    if val_path.exists():
        from .loader import read_rows
        from . import data as od
        rows = read_rows(val_path)
        paths = None
        Xv_all, yv_all = [], []
        classes = PROFILES[profile]["classes"]
        for r in rows:
            p, v = od.flatten(r["inputs"])
            if paths is None:
                paths = p
            elif p != paths:
                raise ValueError("validation feature schema changed")
            Xv_all.append(v)
            ans = r["answer"]
            yv_all.append(classes.index(ans) if profile == "A" else float(ans))
        Xv_all = np.asarray(Xv_all, dtype=np.float64)
        yv_all = np.asarray(yv_all, dtype=np.float64 if profile != "A" else np.int64)
        val_report = official_metrics(profile, incumbent, Xv_all, yv_all)
        val_report["n_rows"] = int(len(yv_all))
        trace.event(step="eval", event="validation-final", payload=val_report)

    result = {
        "run_id": run_id,
        "profile": profile,
        "task": incumbent.meta["task"],
        "n_windows": len(windows),
        "seed": seed,
        "parent_checkpoint": parent,
        "final_checkpoint": chain[-1]["checkpoint_sha256"],
        "n_versions": len(chain),
        "n_promotions": int(sum(1 for r in records if r["promoted"])),
        "n_rollbacks": int(sum(1 for r in records if r["rollback"])),
        "total_epochs": total_epochs,
        "total_wall_seconds": round(total_wall, 2),
        "validation": val_report,
        "windows": records,
        "version_chain": chain,
    }
    (out_dir / "results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    (out_dir / "version_chain.json").write_text(json.dumps(chain, ensure_ascii=False, indent=2) + "\n")
    trace.close({"result": {k: result[k] for k in ("n_versions", "n_promotions", "n_rollbacks", "total_epochs")}})
    return result
