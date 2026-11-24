"""报告与看板：自包含 HTML（离线可开，无 CDN 依赖）+ Markdown 复盘。

看板是答辩用的"证据面板"：分数曲线、漂移窗标记、逐窗决策链、成本对比。
"""

from __future__ import annotations

import html
import json
from typing import Any, Dict, List, Optional

POLICY_LABEL = {
    "closed_loop": "闭环 + 数字孪生预演（本方案）",
    "closed_loop_nosim": "闭环自演化（无预演）",
    "static": "静态基线（一次训练永不更新）",
    "periodic": "定期重训（固定超参）",
    "retune_all": "无差别全量重调参",
}

POLICY_COLOR = {
    "closed_loop": "#0f9b8e",
    "closed_loop_nosim": "#3b7fbe",
    "static": "#8a8f98",
    "periodic": "#d08b2c",
    "retune_all": "#b4553f",
}


def _fmt(v: Any, nd: int = 3) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def _svg_curve(results: Dict[str, Any], kinds: List[str], width: int = 900, height: int = 330) -> str:
    pad_l, pad_r, pad_t, pad_b = 58, 18, 22, 46
    pw, ph = width - pad_l - pad_r, height - pad_t - pad_b
    all_scores = []
    for r in results.values():
        all_scores += [s for s in r["scores"] if s is not None]
    if not all_scores:
        return "<svg></svg>"
    lo, hi = min(all_scores), max(all_scores)
    span = max(hi - lo, 1e-6)
    lo -= span * 0.12
    hi += span * 0.12
    n = max(len(kinds), 1)

    def x(i: int) -> float:
        return pad_l + (pw * i / max(n - 1, 1))

    def y(v: float) -> float:
        return pad_t + ph * (1 - (v - lo) / (hi - lo))

    parts: List[str] = [f'<svg viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg" role="img">']
    # 漂移背景带
    for i, k in enumerate(kinds):
        if k in ("covariate", "concept", "sudden", "gradual"):
            fill = "#f6e3c8" if k != "sudden" else "#f3d3cb"
            parts.append(
                f'<rect x="{x(i) - pw/(2*max(n-1,1)):.1f}" y="{pad_t}" width="{pw/max(n-1,1):.1f}" '
                f'height="{ph}" fill="{fill}" opacity="0.75"/>'
            )
        elif k == "empty":
            parts.append(
                f'<rect x="{x(i) - pw/(2*max(n-1,1)):.1f}" y="{pad_t}" width="{pw/max(n-1,1):.1f}" '
                f'height="{ph}" fill="#e8e8ef" opacity="0.9"/>'
            )
    # 网格 + y 轴
    for g in range(5):
        yy = pad_t + ph * g / 4
        val = hi - (hi - lo) * g / 4
        parts.append(f'<line x1="{pad_l}" y1="{yy:.1f}" x2="{width-pad_r}" y2="{yy:.1f}" stroke="#e6e8ec"/>')
        parts.append(f'<text x="{pad_l-8}" y="{yy+4:.1f}" text-anchor="end" font-size="11" fill="#7a808a">{val:.2f}</text>')
    # 曲线
    for name, r in results.items():
        pts = [(i, s) for i, s in enumerate(r["scores"]) if s is not None]
        if not pts:
            continue
        d = " ".join(f"{x(i):.1f},{y(s):.1f}" for i, s in pts)
        parts.append(
            f'<polyline points="{d}" fill="none" stroke="{POLICY_COLOR.get(name, "#333")}" '
            f'stroke-width="2.6" stroke-linejoin="round"/>'
        )
        for i, s in pts:
            parts.append(
                f'<circle cx="{x(i):.1f}" cy="{y(s):.1f}" r="3" fill="{POLICY_COLOR.get(name, "#333")}"/>'
            )
    # x 轴标签
    for i in range(n):
        parts.append(
            f'<text x="{x(i):.1f}" y="{height-pad_b+18}" text-anchor="middle" font-size="10" fill="#7a808a">W{i}</text>'
        )
    legend_x = pad_l + 6
    for j, (name, r) in enumerate(results.items()):
        lx = legend_x + j * 190
        parts.append(f'<line x1="{lx}" y1="{12}" x2="{lx+22}" y2="{12}" stroke="{POLICY_COLOR.get(name, "#333")}" stroke-width="3"/>')
        parts.append(f'<text x="{lx+28}" y="{16}" font-size="11" fill="#3b4048">{html.escape(POLICY_LABEL.get(name, name))}</text>')
    parts.append("</svg>")
    return "".join(parts)


def render_html(payload: Dict[str, Any]) -> str:
    results = payload["results"]
    kinds = payload["kinds"]
    meta = payload["meta"]
    names = [w["name"] for w in payload["windows"]]
    rows_policy = payload["comparison"]

    kpi_cards = []
    for row in rows_policy:
        color = POLICY_COLOR.get(row["policy"], "#333")
        kpi_cards.append(
            f'<div class="card"><div class="card-h" style="color:{color}">{html.escape(POLICY_LABEL.get(row["policy"], row["policy"]))}</div>'
            f'<div class="card-v">{_fmt(row["mean_score"])}</div>'
            f'<div class="card-s">均分 · 最低 {_fmt(row["worst_score"])} · 恢复 {row["recovery_rate"]}</div>'
            f'<div class="card-s">算力 {_fmt(row["minutes"], 1)} GPU·min · 试验 {row["n_trials"]}</div></div>'
        )

    detail_rows = []
    loop = results.get("closed_loop")
    if loop:
        for i, w in enumerate(loop["windows"]):
            kind = kinds[i] if i < len(kinds) else ""
            tag = {
                "stable": ('稳定', "#eef1f0", "#3c6b5f"),
                "covariate": ('协变量漂移', "#fbefdc", "#a26518"),
                "concept": ('概念漂移', "#fbefdc", "#a26518"),
                "sudden": ('突变漂移', "#f8ded6", "#a03a22"),
                "gradual": ('渐进漂移', "#fbefdc", "#a26518"),
                "empty": ('空跑包', "#ececf2", "#4a4a68"),
            }.get(kind, ("—", "#fff", "#333"))
            inc = w.get("incident", "无")
            inc_color, inc_txt = "#eef1f0", "#3c6b5f"
            if "安全" in inc:
                inc_color, inc_txt = "#f8ded6", "#a03a22"
            elif "系统" in inc:
                inc_color, inc_txt = "#fbefdc", "#a26518"
            elif "数据" in inc:
                inc_color, inc_txt = "#ececf2", "#4a4a68"
            detail_rows.append(
                "<tr>"
                f'<td>{i}</td><td class="mono">{html.escape(w["name"])}</td>'
                f'<td><span class="tag" style="background:{tag[1]};color:{tag[2]}">{tag[0]}</span></td>'
                f'<td>{html.escape(w["drift_type"])}</td>'
                f'<td><span class="tag" style="background:{inc_color};color:{inc_txt}">{html.escape(inc)}</span></td>'
                f'<td class="mono">{html.escape(w["action"])}</td>'
                f'<td>{"晋升" if w["promoted"] else ("回滚" if w.get("rollback") else "—")}</td>'
                f'<td class="num">{_fmt(w["true_score"])}</td>'
                f'<td class="num">v{w["model_version"]}</td>'
                f'<td class="num">{_fmt(w["minutes"], 2)}</td>'
                f'<td class="note">{html.escape(str(w["note"]))[:120]}</td>'
                "</tr>"
            )

    matrix_rows = []
    for i in range(len(names)):
        cells = []
        for name, r in results.items():
            v = r["scores"][i] if i < len(r["scores"]) else None
            cells.append(f'<td class="num">{_fmt(v)}</td>')
        matrix_rows.append(f"<tr><td>W{i}</td>" + "".join(cells) + "</tr>")

    headers = "".join(f"<th>{html.escape(POLICY_LABEL.get(n, n))}</th>" for n in results.keys())
    verify = payload.get("chain_verify", {})

    return f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8"/>
<title>智训引擎 · 运行看板</title>
<style>
:root{{--ink:#1d2126;--sub:#6b7280;--line:#e6e8ec;--bg:#f7f8f9;--panel:#fff;}}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--bg);color:var(--ink);
font-family:-apple-system,"Segoe UI","PingFang SC","Microsoft YaHei",sans-serif;}}
.wrap{{max-width:1060px;margin:0 auto;padding:34px 22px 60px}}
h1{{font-size:26px;margin:0 0 6px;letter-spacing:-.02em}}
.sub{{color:var(--sub);font-size:13px;margin-bottom:26px}}
.kpis{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:26px}}
.card{{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:14px 16px}}
.card-h{{font-size:12px;font-weight:600;margin-bottom:8px;line-height:1.4}}
.card-v{{font-size:26px;font-weight:650;font-variant-numeric:tabular-nums}}
.card-s{{font-size:11px;color:var(--sub);margin-top:4px;line-height:1.5}}
.panel{{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:18px 20px;margin-bottom:22px}}
.panel h2{{font-size:15px;margin:0 0 4px}}
.panel .hint{{font-size:12px;color:var(--sub);margin-bottom:14px}}
table{{width:100%;border-collapse:collapse;font-size:12.5px}}
th{{text-align:left;font-weight:600;color:var(--sub);border-bottom:1px solid var(--line);padding:8px 8px;font-size:12px}}
td{{border-bottom:1px solid #f0f2f4;padding:8px 8px;vertical-align:top}}
.num{{text-align:right;font-variant-numeric:tabular-nums}}
.mono{{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12px}}
.tag{{display:inline-block;padding:1px 7px;border-radius:4px;font-size:11px}}
.note{{color:var(--sub);max-width:300px}}
.legend{{font-size:11.5px;color:var(--sub);margin-top:8px}}
.foot{{font-size:11.5px;color:var(--sub);line-height:1.8}}
</style></head><body><div class="wrap">
<h1>智训引擎 · 工业模型自演化运行看板</h1>
<div class="sub">数据集 {html.escape(meta["dataset"])} · 任务 {html.escape(meta["task"])} · {meta["n_windows"]} 个数据窗 · seed={meta["seed"]} · run_id <span class="mono">{html.escape(meta["run_id"])}</span></div>

<div class="kpis">{"".join(kpi_cards)}</div>

<div class="panel">
<h2>逐窗得分曲线</h2>
<div class="hint">底色标出漂移发生的窗口：橙=协变量/概念/渐进漂移，红=突变，灰=空跑包（应被隔离，不计入训练）。纵轴为归一化得分（越高越好）。</div>
{_svg_curve(results, kinds)}
<div class="legend">判据：闭环方案应在漂移后更快回到漂移前水平，且在稳定段不抖动。</div>
</div>

<div class="panel">
<h2>策略对比</h2>
<div class="hint">同一数据流、同一评测口径。均分/最低分为全流窗口得分，恢复率=漂移事件中回到漂移前 90% 的比例。</div>
<table><thead><tr><th>策略</th><th class="num">均分</th><th class="num">最优窗</th><th class="num">最差窗</th>
<th class="num">稳定期波动</th><th class="num">最大回撤</th><th class="num">恢复率</th><th class="num">平均恢复步数</th>
<th class="num">算力(GPU·min)</th><th class="num">试验数</th></tr></thead><tbody>
{"".join(f'<tr><td>{html.escape(POLICY_LABEL.get(r["policy"], r["policy"]))}</td><td class="num">{_fmt(r["mean_score"])}</td><td class="num">{_fmt(r["best_score"])}</td><td class="num">{_fmt(r["worst_score"])}</td><td class="num">{_fmt(r["stability_std"])}</td><td class="num">{_fmt(r["max_drawdown"])}</td><td class="num">{r["recovery_rate"]}</td><td class="num">{_fmt(r["avg_recovery_steps"])}</td><td class="num">{_fmt(r["minutes"],1)}</td><td class="num">{r["n_trials"]}</td></tr>' for r in rows_policy)}
</tbody></table>
</div>

<div class="panel">
<h2>闭环决策链（逐窗）</h2>
<div class="hint">这是答辩时最该被追问的一张表：每一窗为什么动、动到什么程度、有没有被门槛拦住。</div>
<table><thead><tr><th>窗</th><th>数据窗</th><th>漂移判定</th><th>诊断类型</th><th>故障分级</th><th>处置动作</th><th>上线</th>
<th class="num">真实得分</th><th class="num">版本</th><th class="num">算力</th><th>备注</th></tr></thead><tbody>
{"".join(detail_rows)}
</tbody></table>
</div>

<div class="panel">
<h2>逐窗得分矩阵</h2>
<table><thead><tr><th>窗</th>{headers}</tr></thead><tbody>{"".join(matrix_rows)}</tbody></table>
</div>

<div class="foot">
版本链校验：{verify.get("n_versions", 0)} 个封版版本，{"完整" if verify.get("intact") else "发现断裂 " + str(verify.get("broken_versions"))} ·
评测口径：{html.escape(meta.get("metric", ""))} · 生成时间 {html.escape(meta.get("generated_at", ""))}<br/>
所有决策仅使用在线可得的有标签训练数据；评测窗标签只用于事后打分，未参与任何训练或上线判断。
</div>
</div></body></html>"""


def render_markdown(payload: Dict[str, Any]) -> str:
    meta = payload["meta"]
    kinds = payload["kinds"]
    rows = payload["comparison"]
    loop = payload["results"].get("closed_loop")
    lines = []
    lines.append(f"# 智训引擎运行报告 · {meta['dataset']}")
    lines.append("")
    lines.append(f"- run_id: `{meta['run_id']}`")
    lines.append(f"- 任务: {meta['task']}（{meta['task_type']}），评分口径 {meta['metric']}")
    lines.append(f"- 窗口数: {meta['n_windows']}，seed: {meta['seed']}")
    lines.append("")
    lines.append("## 一、策略对比")
    lines.append("")
    lines.append("| 策略 | 均分 | 最优窗 | 最差窗 | 稳定期波动 | 最大回撤 | 恢复率 | 平均恢复步数 | 算力(GPU·min) | 试验数 |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        lines.append(
            f"| {POLICY_LABEL.get(r['policy'], r['policy'])} | {_fmt(r['mean_score'])} | {_fmt(r['best_score'])} | "
            f"{_fmt(r['worst_score'])} | {_fmt(r['stability_std'])} | {_fmt(r['max_drawdown'])} | "
            f"{r['recovery_rate']} | {_fmt(r['avg_recovery_steps'])} | {_fmt(r['minutes'],1)} | {r['n_trials']} |"
        )
    lines.append("")
    if loop:
        lines.append("## 二、闭环逐窗决策链")
        lines.append("")
        lines.append("| 窗 | 漂移判定 | 诊断 | 故障分级 | 动作 | 上线 | 真实得分 | 版本 | 算力 | 备注 |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|")
        for i, w in enumerate(loop["windows"]):
            lines.append(
                f"| W{i} | {kinds[i]} | {w['drift_type']} | {w.get('incident','无')} | {w['action']} | "
                f"{'晋升' if w['promoted'] else ('回滚' if w.get('rollback') else '—')} | {_fmt(w['true_score'])} | "
                f"v{w['model_version']} | {_fmt(w['minutes'],2)} | {str(w['note'])[:80]} |"
            )
        lines.append("")
    lines.append("## 三、评测口径与约束")
    lines.append("")
    lines.append("- 得分定义：" + meta.get("metric", ""))
    lines.append("- 决策只用在线可得的有标签训练数据；评测窗标签仅用于事后打分，未参与训练或门槛判断。")
    lines.append(f"- 版本链：{payload.get('chain_verify', {}).get('n_versions', 0)} 个封版版本，"
                 f"{'校验完整' if payload.get('chain_verify', {}).get('intact') else '存在断裂'}。")
    lines.append("")
    return "\n".join(lines)
