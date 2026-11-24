"""把 zhixun-engine 运行产物导出为 SwanLab 兼容实验记录。

赛题提交物②要求「云端完整记录」。本工具把本地 trace.jsonl 的全轨迹
整理成 SwanLab 可直接消费的 config + log 序列，保证「本地可复现证据」
与「云端实验记录」同源、同一份数据。

用法:
    python tools/export_swanlab.py <run_dir> [--out <输出目录>]

产出（默认写入 <run_dir>/swanlab_export/）:
    config.json     实验配置（任务/窗口数/随机种子/环境指纹）
    metrics.jsonl   逐窗指标序列（每行 = 一个 SwanLab log 步）
    replay.py       装有 swanlab 后直接运行，一键把 metrics 回放到云端

依赖: 仅标准库 + json；replay.py 才需要 swanlab（按需 pip install swanlab）。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def extract_config(trace_rows: List[Dict[str, Any]], results: Dict[str, Any]) -> Dict[str, Any]:
    cfg: Dict[str, Any] = {}
    for r in trace_rows:
        if r.get("event") == "start":
            cfg.update(r.get("payload") or r.get("meta") or {})
            break
    # 任务信息与最终结果
    if "profile" in results:
        cfg["profile"] = results.get("profile")
    if "validation" in results:
        cfg["validation"] = results.get("validation")
    if "records" in results:
        cfg["n_windows"] = len(results.get("records", []))
    if "metrics" in results:
        cfg["summary_metrics"] = results.get("metrics")
    return cfg


def extract_metrics(trace_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """从 window-score 事件抽 SwanLab 风格 step→metrics 序列。"""
    metrics: List[Dict[str, Any]] = []
    for r in trace_rows:
        if r.get("event") != "window-score":
            continue
        step = int(r.get("window", 0))
        payload = r.get("payload", {})
        entry: Dict[str, Any] = {"step": step}
        m = payload.get("metrics", {})
        if isinstance(m, dict):
            for k, v in m.items():
                if isinstance(v, (int, float)):
                    entry[k] = v
        if payload.get("primary_metric") is not None:
            entry["primary_metric"] = payload["primary_metric"]
        if payload.get("true_score") is not None:
            entry["true_score"] = payload["true_score"]
        if payload.get("action"):
            entry["action"] = payload["action"]
        if payload.get("incident"):
            entry["incident"] = payload["incident"]
        if payload.get("minutes") is not None:
            entry["cost_minutes"] = payload["minutes"]
        metrics.append(entry)
    return metrics


def write_replay(run_dir: Path, out_dir: Path, experiment: str) -> Path:
    replay = f'''"""SwanLab 回放脚本（由 export_swanlab.py 生成）。

用法:
    pip install swanlab
    python {out_dir.name}/replay.py

会把 metrics.jsonl 里的逐窗指标回放到 SwanLab 云端，与本地 trace.jsonl 同源。
"""
import json
from pathlib import Path

try:
    import swanlab
except ImportError:
    raise SystemExit("请先安装 swanlab:  pip install swanlab")

base = Path(__file__).resolve().parent
cfg = json.loads((base / "config.json").read_text(encoding="utf-8"))
metrics = [json.loads(l) for l in (base / "metrics.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]

swanlab.init(experiment_name={json.dumps(experiment)}, config=cfg)
for row in metrics:
    step = row.pop("step")
    swanlab.log(row, step=step)
print(f"SwanLab 回放完成，共 {{len(metrics)}} 步")
'''
    p = out_dir / "replay.py"
    p.write_text(replay, encoding="utf-8")
    return p


def main() -> int:
    ap = argparse.ArgumentParser(description="导出 zhixun-engine 运行为 SwanLab 兼容记录")
    ap.add_argument("run_dir", help="运行产物目录（含 trace.jsonl / results.json）")
    ap.add_argument("--out", help="输出目录，默认 <run_dir>/swanlab_export")
    ap.add_argument("--name", default=None, help="SwanLab 实验名，默认取 run 目录名")
    args = ap.parse_args()

    run_dir = Path(args.run_dir).resolve()
    if not run_dir.exists():
        print(f"目录不存在: {run_dir}", file=sys.stderr)
        return 1

    trace_path = run_dir / "trace.jsonl"
    results_path = run_dir / "results.json"
    trace_rows = _load_jsonl(trace_path)
    results = _load_json(results_path)

    out_dir = Path(args.out).resolve() if args.out else run_dir / "swanlab_export"
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = extract_config(trace_rows, results)
    metrics = extract_metrics(trace_rows)
    experiment = args.name or run_dir.name

    (out_dir / "config.json").write_text(
        json.dumps(cfg, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (out_dir / "metrics.jsonl").write_text(
        "\n".join(json.dumps(m, ensure_ascii=False) for m in metrics) + "\n", encoding="utf-8"
    )
    replay = write_replay(run_dir, out_dir, experiment)

    print(f"导出完成:")
    print(f"  config   -> {out_dir / 'config.json'}")
    print(f"  metrics  -> {out_dir / 'metrics.jsonl'} ({len(metrics)} 步)")
    print(f"  replay   -> {replay}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
