"""一键运行入口。

示例：
  python run.py --dataset synthetic --out runs/demo
  python run.py --dataset gas --out runs/gas
  python run.py --dataset synthetic --fast          # 跳过最贵的对照策略
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from engine import loop as loop_lib
from engine import metrics as metrics_lib
from engine import report as report_lib
from engine import streams
from engine.seal import Sealer
from engine.types import Window


def _metric_text(task) -> str:
    if task.task_type == "regression":
        return "scale/(scale+RMSE) 有界得分（1=完美，越差越趋近 0）"
    return "macro-F1（越高越好）"


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    ap = argparse.ArgumentParser(description="智训引擎：工业模型自演化闭环")
    ap.add_argument("--dataset", default="synthetic",
                    choices=["synthetic", "gas", "cmapss", "xjtu",
                             "official-a", "official-b", "official-c"])
    ap.add_argument("--raw-dir", default=str(Path(__file__).resolve().parent / "data" / "raw"))
    ap.add_argument("--official-dir", default=str(Path(__file__).resolve().parent.parent / "official-data"),
                    help="官方包解压根目录（内含 A-gas/B-bearing/C-engine）")
    ap.add_argument("--windows", type=int, default=12)
    ap.add_argument("--max-candidates", type=int, default=6)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None, help="输出目录，默认 runs/<时间戳>")
    ap.add_argument("--fast", action="store_true", help="跳过最贵的无差别重调参对照")
    ap.add_argument("--no-baselines", action="store_true")
    ap.add_argument("--full", action="store_true", help="额外跑无差别全量重调参对照")
    ap.add_argument("--no-sim", action="store_true", help="关闭数字孪生预演（本方案）")
    args = ap.parse_args()

    # 官方包续训路径（与自建 sklearn 闭环分开）
    if args.dataset.startswith("official-"):
        from engine.official.loop import run_official

        profile = {"official-a": "A", "official-b": "B", "official-c": "C"}[args.dataset]
        subdir = {"A": "A-gas", "B": "B-bearing", "C": "C-engine"}[profile]
        data_dir = Path(args.official_dir) / subdir
        run_id = datetime.now().strftime("%Y%m%d-%H%M%S") + f"-{args.dataset}"
        out_dir = Path(args.out) if args.out else Path(__file__).resolve().parent / "runs" / run_id
        print(f"[官方续训] 任务 {profile}，数据目录 {data_dir}")
        res = run_official(profile, data_dir, out_dir, seed=args.seed,
                           n_windows=args.windows, max_candidates=args.max_candidates)
        _print_official_summary(res)
        print(f"\n产物目录：{out_dir}")
        print("  - results.json          结构化结果（逐窗 + 最终 validation 干净评测）")
        print("  - trace.jsonl           全轨迹实验记录")
        print("  - version_chain.json    封版版本哈希链")
        print("  - versions/vNNNN/       续训检查点 + manifest")
        return 0

    run_id = datetime.now().strftime("%Y%m%d-%H%M%S") + f"-{args.dataset}"
    out_dir = Path(args.out) if args.out else Path(__file__).resolve().parent / "runs" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    windows, task = streams.build_stream(args.dataset, Path(args.raw_dir), seed=args.seed, n_windows=args.windows)
    kinds = [str(w.meta.get("kind", "unknown")) for w in windows]
    print(f"[流] {args.dataset}: {len(windows)} 个窗口，任务 {task.name}（{task.task_type}）")
    print(f"[流] 特征维度 {windows[0].X_train.shape[1]}，首窗样本 {len(windows[0].y_train)}")

    results = {}

    print("[1/4] 运行闭环（含数字孪生预演）…")
    sim_dir = out_dir / "closed_loop"
    sim_dir.mkdir(parents=True, exist_ok=True)
    cl = loop_lib.ClosedLoop(task, windows, sim_dir, run_id, seed=args.seed, use_simulator=not args.no_sim)
    res = cl.run()
    results["closed_loop"] = res

    if not args.no_baselines:
        print("[2/4] 运行闭环（无预演对照）…")
        nosim_dir = out_dir / "closed_loop_nosim"
        nosim_dir.mkdir(parents=True, exist_ok=True)
        cl_nosim = loop_lib.ClosedLoop(task, windows, nosim_dir, run_id + "-nosim",
                                       seed=args.seed, use_simulator=False)
        results["closed_loop_nosim"] = cl_nosim.run()

        print("[3/4] 静态基线 + 定期重训 …")
        results["static"] = loop_lib.run_static(task, windows, seed=args.seed)
        results["periodic"] = loop_lib.run_periodic(task, windows, seed=args.seed)

        if args.full:
            print("[4/4] 无差别全量重调参（最贵，用于成本对照）…")
            results["retune_all"] = loop_lib.run_retune_all(task, windows, seed=args.seed)

    # 汇总
    policy_metrics = {name: r.metrics for name, r in results.items()}
    comparison = metrics_lib.compare(policy_metrics)
    chain_verify = Sealer.verify_chain(cl.sealer.chain)

    payload = {
        "meta": {
            "dataset": args.dataset,
            "task": task.name,
            "task_type": task.task_type,
            "n_windows": len(windows),
            "seed": args.seed,
            "run_id": run_id,
            "metric": _metric_text(task),
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "wall_seconds": round(time.perf_counter() - t0, 1),
        },
        "kinds": kinds,
        "windows": [w.summary() | {"name": w.name} for w in windows],
        "results": {
            name: {
                "policy": name,
                "scores": [r.true_score for r in r.records],
                "windows": [r.to_dict() for r in r.records],
                "metrics": r.metrics,
            }
            for name, r in results.items()
        },
        "comparison": comparison,
        "chain_verify": chain_verify,
    }

    (out_dir / "results.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "report.md").write_text(report_lib.render_markdown(payload), encoding="utf-8")
    (out_dir / "dashboard.html").write_text(report_lib.render_html(payload), encoding="utf-8")
    cl.sealer.write_reproduce(
        run_id,
        f"python run.py --dataset {args.dataset} --windows {args.windows} --seed {args.seed} --out runs/{run_id}",
        args.seed,
        args.dataset,
    )

    # 控制台摘要
    print("\n=== 策略对比 ===")
    header = f"{'策略':<26}{'均分':>8}{'最低':>8}{'恢复率':>8}{'恢复步数':>10}{'算力':>9}{'试验':>7}"
    print(header)
    for row in comparison:
        label = report_lib.POLICY_LABEL.get(row["policy"], row["policy"])
        print(
            f"{label:<26}{_f(row['mean_score']):>8}{_f(row['worst_score']):>8}"
            f"{str(row['recovery_rate']):>8}{_f(row['avg_recovery_steps']):>10}"
            f"{_f(row['minutes'], 1):>9}{str(row['n_trials']):>7}"
        )
    print(f"\n版本链：{chain_verify['n_versions']} 个封版版本，"
          f"{'完整' if chain_verify['intact'] else '断裂 ' + str(chain_verify['broken_versions'])}")
    print(f"\n产物目录：{out_dir}")
    print("  - dashboard.html        答辩看板（双击即开）")
    print("  - report.md             复盘报告")
    print("  - results.json          结构化结果")
    print("  - closed_loop/trace.jsonl     闭环（含预演）全轨迹实验记录")
    print("  - closed_loop_nosim/trace.jsonl  闭环（无预演）全轨迹实验记录")
    print("  - closed_loop/versions/       封版模型与 manifest 哈希链")
    return 0


def _f(v, nd: int = 3) -> str:
    if v is None:
        return "—"
    if isinstance(v, (int, float)):
        return f"{v:.{nd}f}"
    return str(v)


def _print_official_summary(res) -> None:
    mname = "Macro-F1" if res["profile"] == "A" else "RMSE"
    print("\n=== 官方续训结果 ===")
    print(f"任务 {res['profile']}（{res['task']}）· {res['n_windows']} 个窗口 · {res['n_versions']} 个封版版本")
    print(f"晋升 {res['n_promotions']} 次，回滚 {res['n_rollbacks']} 次，累计 {res['total_epochs']} epoch，耗时 {res['total_wall_seconds']}s")
    print(f"\n{'窗':<4}{'动作':<18}{'漂移类型':<16}{'主指标':>10}{'epoch':>6}{'候选':>5}")
    for r in res["windows"]:
        print(f"{r['window']:<4}{r['action']:<18}{r['drift_type']:<16}"
              f"{_f(r['primary_metric']):>10}{r['epochs_spent']:>6}{r['n_candidates']:>5}")
    if res.get("validation"):
        v = res["validation"]
        print("\n最终 validation.jsonl 干净评测（未参与训练/门槛）：")
        for k, val in v.items():
            if k == "n_rows":
                print(f"  样本数 {val}")
            elif isinstance(val, (int, float)):
                print(f"  {k} = {val:.4f}" if isinstance(val, float) else f"  {k} = {val}")


if __name__ == "__main__":
    raise SystemExit(main())
