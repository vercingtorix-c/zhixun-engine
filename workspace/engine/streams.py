"""数据流：把三类异构工业数据统一成"按时间到达的窗口流"。

支持：
  synthetic  内置工业仿真流（注塑/焊接尺寸偏差预测），带 5 类漂移事件与空跑包，
             无需外网即可完整演示五步闭环——评委机器上离线可复现。
  gas        UCI Gas Sensor Array Drift（真实传感器漂移，10 个批次跨 36 个月）
  cmapss     NASA CMAPSS 涡扇发动机退化数据（RUL 回归）
  xjtu       XJTU-SY 轴承全寿命振动（时域特征 + RUL 回归）

约定：盖住标签的评测窗（y_eval）只交给外部裁判打分，引擎内部不读。
"""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from .types import TaskSpec, Window, make_task_from_window

# ---------------------------------------------------------------------------
# 1) 内置工业仿真流
# ---------------------------------------------------------------------------


def _regime(kind: str, d: int = 14):
    """漂移剧本的"关系参数"：不同 regime 代表不同的工艺稳态。

    w 是线性系数，inter 是交互项系数，gain/offset 是传感器量程。
    关键设定：漂移是**永久迁移**——工业设备/工艺换到新 regime 后不再回头，
    因此静态模型无法靠"等数据恢复"自救，必须主动适应。
    """
    w_base = np.array([3.0, -2.2, 1.4, 0.8, -1.1, 0.5, 1.9, -0.7])
    gain = np.ones(d)
    offset = np.zeros(d)

    if kind in ("stable", "covariate"):
        w, inter = w_base.copy(), 2.2
        if kind == "covariate":  # 传感器增益漂移：输入分布变，映射关系不变
            gain[[2, 5]] = 1.38
            offset[[2, 5]] = 3.5
    elif kind == "concept":  # 概念漂移：输入分布不变，映射关系变（regime A→B）
        w, inter = w_base.copy(), 0.3
        w[0], w[2] = 0.6, 1.8
    elif kind in ("sudden", "steady_c"):  # 设备更换：量程重标定 + 关系跳变（regime C）
        gain = np.array([1.0, 1.25, 0.8, 1.1, 1.0, 1.3, 1.0, 0.9] + [1.0] * (d - 8))
        offset = np.array([0.0, 2.0, -1.5, 0.0, 0.0, -2.5, 0.0, 0.0] + [0.0] * (d - 8))
        w = np.array([1.2, 2.4, -1.9, 1.6, 0.6, -0.9, 2.3, 1.1])
        inter = 0.4
    elif kind == "steady_d":  # 长期老化后的最终稳态（regime D）
        w = np.array([-0.8, 1.6, 2.0, -0.6, 1.1, 0.3, -1.3, 0.9])
        inter = 1.0
        gain = np.array([1.05, 1.0, 0.95, 1.0, 1.0, 1.1, 1.0, 1.0] + [1.0] * (d - 8))
    elif kind == "empty":
        return None
    else:
        raise ValueError(kind)
    return w, inter, gain, offset


def _make_phase(kind: str, rng: np.random.Generator, n: int) -> Tuple[np.ndarray, np.ndarray]:
    """生成一个相位的数据。X 是 14 维：8 个有效工艺/传感器量 + 6 个冗余通道。"""
    d = 14
    if kind == "empty":  # 空跑包：无业务意义的纯噪声（必须被识别并丢弃）
        X = rng.uniform(0.0, 120.0, size=(n, d))
        y = rng.normal(0.0, 1.0, size=n) * 9.9
        return X, y

    w, inter, gain, offset = _regime(kind, d)
    X = rng.normal(0.0, 1.0, size=(n, d))
    X = 40.0 + 9.0 * X
    X = (X - 40.0) * gain + 40.0 + offset
    y = X[:, :8] @ w + inter * (X[:, 2] * X[:, 3]) + 1.3 * np.sin(X[:, 4]) + X[:, 6] * X[:, 7] / 9.0
    y = y / 10.0 + rng.normal(0.0, 0.18, size=n)
    return X, y


def synthetic_stream(
    n_windows: int = 12, n_train: int = 300, n_eval: int = 400, seed: int = 42
) -> Tuple[List[Window], TaskSpec]:
    """12 个窗口的完整剧本：A → 协变量 → 概念(B) → 突变(C) → 空跑 → 新稳态(C) → 渐进(D) → 稳态(D)。

    注意：漂移是永久迁移，后段的"新稳态"不再回到 regime A，静态模型无法靠等待恢复。
    """
    plan = [
        ("stable", "稳定期A"),
        ("stable", "稳定期A"),
        ("stable", "稳定期A"),
        ("covariate", "协变量漂移（传感器增益）"),
        ("concept", "概念漂移（关系A→B）"),
        ("concept", "概念漂移延续B"),
        ("sudden", "突变漂移（设备更换C）"),
        ("empty", "空跑数据包（应丢弃）"),
        ("steady_c", "突变后新稳态C"),
        ("steady_c", "突变后新稳态C"),
        ("gradual", "渐进漂移（C→D）"),
        ("steady_d", "最终稳态D"),
    ]
    if n_windows != len(plan):
        plan = plan[:n_windows] if n_windows < len(plan) else plan + [
            ("steady_d", "稳态D") for _ in range(n_windows - len(plan))
        ]

    rng = np.random.default_rng(seed)
    windows: List[Window] = []
    for i, (kind, label) in enumerate(plan):
        if kind == "gradual":
            # 半个窗口还是 regime C，后半个窗口逐步过渡到 regime D
            Xa, ya = _make_phase("steady_c", rng, n_train)
            Xb, yb = _make_phase("steady_d", rng, n_train)
            a = np.linspace(0.0, 1.0, n_train)[:, None]
            Xtr, ytr = Xa * (1 - a) + Xb * a, ya * (1 - a.ravel()) + yb * a.ravel()
            Xa, ya = _make_phase("steady_c", rng, n_eval)
            Xb, yb = _make_phase("steady_d", rng, n_eval)
            a = np.linspace(0.0, 1.0, n_eval)[:, None]
            Xev, yev = Xa * (1 - a) + Xb * a, ya * (1 - a.ravel()) + yb * a.ravel()
            meta_kind = "gradual"
        else:
            Xtr, ytr = _make_phase(kind, rng, n_train)
            Xev, yev = _make_phase(kind, rng, n_eval)
            meta_kind = kind
        windows.append(
            Window(
                index=i,
                name=f"W{i:02d}-{label}",
                X_train=Xtr.astype(np.float64),
                y_train=ytr.astype(np.float64),
                X_eval=Xev.astype(np.float64),
                y_eval=yev.astype(np.float64),
                meta={"kind": meta_kind, "label": label},
            )
        )

    task = make_task_from_window(windows[0], "synthetic-molding")
    return windows, task


# ---------------------------------------------------------------------------
# 2) UCI Gas Sensor Array Drift（真实漂移数据）
# ---------------------------------------------------------------------------


def _iter_batch_files(zf: zipfile.ZipFile) -> List[Tuple[int, str]]:
    out = []
    for name in zf.namelist():
        base = name.split("/")[-1]
        if base.lower().startswith("batch") and base.lower().endswith(".dat"):
            stem = base[5:-4]
            if stem.isdigit():
                out.append((int(stem), name))
    return sorted(out)


def _parse_dat(text: str, n_features: int = 128):
    """解析 gas 数据：支持 libsvm 稀疏格式（label 1:val 2:val ...）与普通空格分隔。"""
    labels: list = []
    feats: list = []
    libsvm = False
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.replace(",", " ").split()
        if len(parts) < 2:
            continue
        if ":" in parts[1]:  # libsvm 稀疏格式
            libsvm = True
            try:
                label = float(parts[0])
            except ValueError:
                continue
            row = np.zeros(n_features)
            for tok in parts[1:]:
                if ":" not in tok:
                    continue
                idx, val = tok.split(":", 1)
                try:
                    i, v = int(idx), float(val)
                    if 1 <= i <= n_features:
                        row[i - 1] = v
                except ValueError:
                    continue
            labels.append(label)
            feats.append(row)
        else:
            try:
                nums = [float(v) for v in parts]
            except ValueError:
                continue
            labels.append(nums[0])
            feats.append(nums[1:])

    if not labels:
        return np.empty((0, 0)), np.empty((0,))
    width = max(len(f) for f in feats)
    X = np.full((len(labels), width), np.nan)
    for i, f in enumerate(feats):
        X[i, : len(f)] = f
    return X, np.asarray(labels, dtype=np.float64)


def gas_stream(
    zip_path: Path,
    n_train: int = 500,
    n_eval: int = 500,
    seed: int = 42,
    max_batches: Optional[int] = None,
) -> Tuple[List[Window], TaskSpec]:
    """每个批次 = 一个时间窗。批次间隔约 3 个月，天然存在传感器漂移。"""
    rng = np.random.default_rng(seed)
    with zipfile.ZipFile(zip_path) as zf:
        batches = _iter_batch_files(zf)
        if not batches:
            raise RuntimeError(f"{zip_path} 里没有找到 batchN.dat")
        if max_batches:
            batches = batches[:max_batches]
        windows: List[Window] = []
        for idx, (batch_id, member) in enumerate(batches):
            X_all, y_all = _parse_dat(zf.read(member).decode("utf-8", errors="ignore"))
            if X_all.size == 0:
                continue
            # 清洗：inf -> nan（个别传感器通道存在异常读数）
            X_all = np.where(np.isfinite(X_all), X_all, np.nan)
            perm = rng.permutation(len(y_all))
            n = len(perm)
            n_tr = min(n_train, max(1, n // 2))
            n_ev = min(n_eval, n - n_tr)
            tr_idx, ev_idx = perm[:n_tr], perm[n_tr : n_tr + n_ev]
            windows.append(
                Window(
                    index=idx,
                    name=f"B{batch_id:02d}-batch{batch_id}",
                    X_train=X_all[tr_idx],
                    y_train=y_all[tr_idx],
                    X_eval=X_all[ev_idx],
                    y_eval=y_all[ev_idx],
                    meta={"batch": batch_id, "kind": "real-drift"},
                )
            )
    task = make_task_from_window(windows[0], "uci-gas-drift")
    return windows, task


# ---------------------------------------------------------------------------
# 3) NASA CMAPSS
# ---------------------------------------------------------------------------


def _read_cmapss(path: Path) -> np.ndarray:
    rows = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.split()
            if not parts:
                continue
            try:
                rows.append([float(v) for v in parts])
            except ValueError:
                continue
    if not rows:
        return np.empty((0, 26))
    return np.asarray(rows, dtype=np.float64)


def cmapss_stream(
    raw_dir: Path, n_windows: int = 10, seed: int = 42, subsets: Tuple[str, ...] = ("FD001", "FD002", "FD004")
) -> Tuple[List[Window], TaskSpec]:
    """多工况子集串接 = 天然协变量+概念漂移；目标为剩余寿命 RUL（截断 125）。"""
    files = []
    for s in subsets:
        p = raw_dir / f"train_{s}.txt"
        if p.exists():
            files.append((s, p))
    if not files:
        raise RuntimeError(f"{raw_dir} 下未找到 train_FD00*.txt")

    X_parts, y_parts, grp_parts = [], [], []
    for si, (sub, path) in enumerate(files):
        arr = _read_cmapss(path)
        if arr.size == 0:
            continue
        unit, cycle = arr[:, 0], arr[:, 1]
        feats = arr[:, 2:26]
        # 去掉常数列（CMAPSS 部分传感器恒定，没信息量）
        keep = np.nanstd(feats, axis=0) > 1e-8
        feats = feats[:, keep]
        for u in np.unique(unit):
            m = unit == u
            c = cycle[m]
            rul = (c.max() - c).clip(0, 125)
            X_parts.append(feats[m])
            y_parts.append(rul)
            grp_parts.append(np.full(m.sum(), si * 1000 + int(u)))

    X = np.vstack(X_parts)
    y = np.concatenate(y_parts)
    g = np.concatenate(grp_parts)

    # 按 (子集, 发动机, 周期) 重排，模拟"流"的到达顺序
    order = np.lexsort((np.arange(len(y)), g))
    X, y, g = X[order], y[order], g[order]

    rng = np.random.default_rng(seed)
    chunks = np.array_split(np.arange(len(y)), n_windows)
    windows: List[Window] = []
    for i, idx in enumerate(chunks):
        idx = rng.permutation(idx)
        n_tr = int(len(idx) * 0.6)
        windows.append(
            Window(
                index=i,
                name=f"C{i:02d}-{'/'.join(s for s, _ in files)}",
                X_train=X[idx[:n_tr]],
                y_train=y[idx[:n_tr]],
                X_eval=X[idx[n_tr:]],
                y_eval=y[idx[n_tr:]],
                meta={"kind": "fleet-shift"},
            )
        )
    task = make_task_from_window(windows[0], "cmapss-rul")
    return windows, task


# ---------------------------------------------------------------------------
# 4) XJTU-SY 轴承
# ---------------------------------------------------------------------------


def _bearing_features(seg: np.ndarray) -> List[float]:
    seg = np.asarray(seg, dtype=np.float64).ravel()
    seg = seg[np.isfinite(seg)]
    if seg.size == 0:
        return [0.0] * 8
    absx = np.abs(seg)
    rms = float(np.sqrt(np.mean(seg ** 2)))
    return [
        float(np.mean(absx)),                         # 平均幅值
        rms,                                          # 均方根
        float(np.max(absx)),                          # 峰值
        float(np.std(seg)),                           # 标准差
        float(np.mean(seg ** 4) / (rms ** 4 + 1e-12)),  # 峭度
        float(np.max(absx) / (rms + 1e-12)),          # 峰值因子
        float(np.mean(absx) / (rms + 1e-12)),         # 波形因子
        float(np.max(absx) / (np.mean(absx) + 1e-12)),  # 脉冲因子
    ]


def xjtu_stream(raw_dir: Path, n_windows: int = 10, seed: int = 42) -> Tuple[List[Window], TaskSpec]:
    """每个 csv = 一次全寿命试验；切成等长片段，提取 8 维时域特征，目标为剩余寿命比例。"""
    csvs = sorted(Path(raw_dir).rglob("*.csv"))
    if not csvs:
        raise RuntimeError(
            f"{raw_dir} 下未找到 XJTU-SY 的 csv。请按 README 下载后把 */*/*.csv 放到该目录。"
        )
    X_parts, y_parts = [], []
    for path in csvs:
        try:
            raw = np.genfromtxt(path, delimiter=",", dtype=np.float64)
        except Exception:
            continue
        if raw.size == 0:
            continue
        horiz = raw[:, 0] if raw.ndim == 2 and raw.shape[1] >= 2 else raw.ravel()
        seg_len = 1024
        n_seg = len(horiz) // seg_len
        if n_seg < 10:
            continue
        feats = np.asarray([_bearing_features(horiz[i * seg_len : (i + 1) * seg_len]) for i in range(n_seg)])
        rul = (n_seg - np.arange(n_seg)) / n_seg  # 0..1 的剩余寿命比例
        X_parts.append(feats)
        y_parts.append(rul)
    if not X_parts:
        raise RuntimeError("XJTU-SY csv 解析失败（文件太短或格式不符）")

    X, y = np.vstack(X_parts), np.concatenate(y_parts)
    idx_all = np.arange(len(y))
    chunks = np.array_split(idx_all, n_windows)
    rng = np.random.default_rng(seed)
    windows = []
    for i, idx in enumerate(chunks):
        idx = rng.permutation(idx)
        n_tr = int(len(idx) * 0.6)
        windows.append(
            Window(
                index=i,
                name=f"X{i:02d}-bearing",
                X_train=X[idx[:n_tr]],
                y_train=y[idx[:n_tr]],
                X_eval=X[idx[n_tr:]],
                y_eval=y[idx[n_tr:]],
                meta={"kind": "run-to-failure"},
            )
        )
    task = make_task_from_window(windows[0], "xjtu-sy-rul")
    return windows, task


# ---------------------------------------------------------------------------


def build_stream(dataset: str, raw_dir: Path, seed: int = 42, n_windows: int = 12):
    dataset = dataset.lower()
    if dataset == "synthetic":
        return synthetic_stream(n_windows=n_windows, seed=seed)
    if dataset == "gas":
        zips = sorted(Path(raw_dir).glob("*.zip"))
        if not zips:
            raise RuntimeError(f"{raw_dir} 下没有 gas 数据集 zip，请先运行 download_data.py")
        return gas_stream(zips[0], seed=seed)
    if dataset == "cmapss":
        return cmapss_stream(Path(raw_dir) / "cmapss", seed=seed)
    if dataset == "xjtu":
        return xjtu_stream(Path(raw_dir) / "xjtu-sy", seed=seed)
    raise ValueError(f"未知数据集: {dataset}")


def dataset_probe(raw_dir: Path) -> dict:
    """给 CLI 用：报告当前哪些数据集就绪。"""
    raw_dir = Path(raw_dir)
    info = {"synthetic": True}
    info["gas"] = bool(list(raw_dir.glob("*.zip")))
    info["cmapss"] = bool(list((raw_dir / "cmapss").glob("train_FD00*.txt")))
    info["xjtu"] = bool(list((raw_dir / "xjtu-sy").rglob("*.csv")))
    return info


if __name__ == "__main__":
    import sys

    d = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/raw")
    print(json.dumps(dataset_probe(d), ensure_ascii=False, indent=2))
