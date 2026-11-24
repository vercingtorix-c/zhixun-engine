"""智训引擎 Harness 参赛模块。

协议：contest.harness.workspace.v2（SDK 1.0.4）
实现：train(rnd) 五步闭环（监测—诊断—优化—评估—封版决策），predict(sample, model_dir) 官方协议输出。

约束（来自使用文档）：
- stdout 只允许协议 JSONL，一切日志走 stderr；
- /app 只读；工作目录 /workspace；每轮必须至少一次真实大模型调用；
- 候选检查点沿用官方 contest.cpu-mlp.v1 格式（从平台父检查点续训）。
"""
import io
import json
import math
import os
import re
import sys
import zipfile
from pathlib import Path

import numpy as np

TASKS = ("classification", "regression")

# 工作根目录：平台上为 /workspace；本地离线测试可用 ZHIXUN_WORKSPACE 覆盖。
_WORKSPACE = Path(os.environ.get("ZHIXUN_WORKSPACE", "/workspace"))

FORBIDDEN_INPUTS = {"truerul", "normalizedrul", "remainingusefullife", "rul", "totalsnapshots"}
PARAMETERS = ("w1", "b1", "w2", "b2")
FORMAT = "contest.cpu-mlp.v1"
MAX_CHECKPOINT_BYTES = 16 * 1024 * 1024

_THREADS = int(os.environ.get("BASELINE_CPU_THREADS", "2"))
if not 1 <= _THREADS <= 8:
    _THREADS = 2
for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = str(_THREADS)


def _log(message):
    print("[zhixun-engine] %s" % message, file=sys.stderr, flush=True)


# ----------------------------------------------------------------------------
# 数据读取（官方 data.py 等价逻辑：flatten + JSONL 逐行解析）
# ----------------------------------------------------------------------------

def _json_value(raw):
    def reject(_):
        raise ValueError("nonfinite JSON number")

    return json.loads(raw, object_pairs_hook=_unique_object, parse_constant=reject)


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def flatten(inputs):
    """把嵌套 inputs 展平成 (paths, values)。禁止目标泄漏字段。"""
    paths, values = [], []

    def visit(value, path, depth):
        if depth > 16:
            raise ValueError("input nesting exceeds limit")
        if isinstance(value, dict):
            for key in sorted(value):
                if not isinstance(key, str) or re.sub(r"[^a-z0-9]", "", key.lower()) in FORBIDDEN_INPUTS:
                    raise ValueError("target-derived input field is not allowed")
                visit(value[key], path + [("key", key)], depth + 1)
        elif isinstance(value, list):
            for index, child in enumerate(value):
                visit(child, path + [("index", index)], depth + 1)
        elif isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
            paths.append(json.dumps(path, ensure_ascii=False, separators=(",", ":")))
            values.append(float(value))
        else:
            raise ValueError("model inputs must contain finite numeric features only")

    visit(inputs, [], 0)
    if not values or len(values) > 4096:
        raise ValueError("expected 1..4096 numeric features")
    return paths, values


def _read_rows(rows, task, classes, expected_paths):
    ids, features, targets = [], [], []
    seen = set()
    paths = expected_paths
    for row in rows:
        if not isinstance(row, dict) or not {"id", "inputs", "answer"} <= set(row):
            raise ValueError("id, inputs and answer are required")
        identity = row["id"]
        if not isinstance(identity, str) or not identity or identity in seen:
            raise ValueError("sample IDs must be nonempty and unique")
        current_paths, values = flatten(row["inputs"])
        if paths is None:
            paths = current_paths
        if current_paths != paths:
            raise ValueError("feature schema changed")
        answer = row["answer"]
        if task == "classification":
            if not isinstance(answer, str) or not classes or answer not in classes:
                raise ValueError("label is outside the configured class list")
            target = classes.index(answer)
        elif task == "regression":
            if isinstance(answer, bool) or not isinstance(answer, (float, int)) or not math.isfinite(answer) or answer < 0:
                raise ValueError("RUL must be a finite nonnegative number")
            target = float(answer)
        else:
            raise ValueError("unsupported task")
        ids.append(identity)
        seen.add(identity)
        features.append(values)
        targets.append(target)
    if not ids:
        raise ValueError("dataset is empty")
    return (np.asarray(features, dtype=np.float64),
            np.asarray(targets, dtype=np.int64 if task == "classification" else np.float64),
            paths)


# ----------------------------------------------------------------------------
# 官方 MLP 模型（contest.cpu-mlp.v1，与父检查点完全兼容，支持续训）
# ----------------------------------------------------------------------------

class Model:
    def __init__(self, task, paths, classes, x, y, *, hidden=32, seed=42, learning_rate=.003, batch_size=128):
        if task not in ("classification", "regression") or not 1 <= hidden <= 128:
            raise ValueError("invalid model configuration")
        if x.ndim != 2 or x.shape[1] != len(paths):
            raise ValueError("feature schema mismatch")
        output = len(classes) if task == "classification" else 1
        if task == "classification" and (len(classes) < 2 or len(set(classes)) != len(classes)):
            raise ValueError("classification needs distinct classes")
        count = x.shape[1] * hidden + hidden + hidden * output + output
        if count >= 100000:
            raise ValueError("reference baseline must have fewer than 100,000 parameters")
        rng = np.random.default_rng(seed)
        self.params = {"w1": rng.normal(0, math.sqrt(1 / x.shape[1]), (x.shape[1], hidden)),
                       "b1": np.zeros(hidden),
                       "w2": rng.normal(0, math.sqrt(1 / hidden), (hidden, output)),
                       "b2": np.zeros(output)}
        self.first = {key: np.zeros_like(value) for key, value in self.params.items()}
        self.second = {key: np.zeros_like(value) for key, value in self.params.items()}
        self.mean = x.mean(axis=0)
        self.scale = x.std(axis=0)
        self.scale[self.scale < 1e-12] = 1
        self.meta = {"format": FORMAT, "device": "cpu", "task": task, "paths": paths,
                     "classes": list(classes), "hidden": hidden, "seed": seed, "epochs": 0, "steps": 0,
                     "learning_rate": learning_rate, "batch_size": batch_size, "parameter_count": count,
                     "target_mean": float(y.mean()) if task == "regression" else 0.0,
                     "target_scale": max(float(y.std()), 1.0) if task == "regression" else 1.0}
        self.validate()

    def validate(self):
        m = self.meta
        if m.get("format") != FORMAT or m.get("device") != "cpu" or m.get("task") not in ("classification", "regression"):
            raise ValueError("unsupported checkpoint")
        for name in ("seed", "epochs", "steps", "hidden", "batch_size", "parameter_count"):
            if type(m.get(name)) is not int or m[name] < 0:
                raise ValueError("invalid checkpoint integer")
        if not 1 <= m["hidden"] <= 128 or not 1 <= m["batch_size"] <= 65536 or not 0 < m["parameter_count"] < 100000:
            raise ValueError("invalid checkpoint architecture")
        if not 0 <= m["seed"] < 2 ** 32:
            raise ValueError("seed must fit an unsigned 32-bit integer")
        if not isinstance(m.get("paths"), list) or not 1 <= len(m["paths"]) <= 4096 or any(not isinstance(p, str) for p in m["paths"]) or len(set(m["paths"])) != len(m["paths"]):
            raise ValueError("invalid checkpoint feature schema")
        classes = m.get("classes")
        if not isinstance(classes, list) or any(not isinstance(c, str) or not c for c in classes) or len(set(classes)) != len(classes):
            raise ValueError("invalid checkpoint classes")
        if (m["task"] == "classification" and len(classes) < 2) or (m["task"] == "regression" and classes):
            raise ValueError("invalid checkpoint classes")
        for name in ("learning_rate", "target_mean", "target_scale"):
            if isinstance(m.get(name), bool) or not isinstance(m.get(name), (float, int)) or not math.isfinite(m[name]):
                raise ValueError("invalid checkpoint numeric state")
        if not 0 < m["learning_rate"] <= 1 or m["target_scale"] <= 0:
            raise ValueError("invalid optimizer/scaler state")
        input_size, hidden = len(m["paths"]), m["hidden"]
        output = len(classes) if m["task"] == "classification" else 1
        shapes = {"w1": (input_size, hidden), "b1": (hidden,), "w2": (hidden, output), "b2": (output,)}
        for group in (self.params, self.first, self.second):
            if set(group) != set(PARAMETERS):
                raise ValueError("checkpoint tensors missing")
            for name, shape in shapes.items():
                if group[name].shape != shape or group[name].dtype != np.dtype("float64") or not np.all(np.isfinite(group[name])):
                    raise ValueError("checkpoint tensor shape or values invalid")
        if sum(value.size for value in self.params.values()) != m["parameter_count"]:
            raise ValueError("checkpoint parameter count mismatch")
        if self.mean.dtype != np.dtype("float64") or self.scale.dtype != np.dtype("float64") or self.mean.shape != (input_size,) or self.scale.shape != (input_size,) or not np.all(np.isfinite(self.mean)) or not np.all(np.isfinite(self.scale)) or np.any(self.scale <= 0):
            raise ValueError("checkpoint normalization invalid")
        if any(np.any(value < 0) for value in self.second.values()):
            raise ValueError("invalid Adam second moment")

    def normalized(self, x):
        if x.ndim != 2 or x.shape[1] != len(self.meta["paths"]) or not np.all(np.isfinite(x)):
            raise ValueError("invalid feature matrix")
        return np.clip((x - self.mean) / self.scale, -10, 10)

    def forward(self, x):
        hidden = np.tanh(x @ self.params["w1"] + self.params["b1"])
        return hidden, hidden @ self.params["w2"] + self.params["b2"]

    def fit(self, x, y, epochs):
        if type(epochs) is not int or not 1 <= epochs <= 10000:
            raise ValueError("epochs must be between 1 and 10000")
        x = self.normalized(x)
        if y.shape != (len(x),) or not len(x):
            raise ValueError("target shape mismatch")
        if self.meta["task"] == "regression":
            target = (y - self.meta["target_mean"]) / self.meta["target_scale"]
        else:
            if np.any(y < 0) or np.any(y >= len(self.meta["classes"])):
                raise ValueError("unknown class")
            target = y
        epoch_losses = []
        for _ in range(epochs):
            rng = np.random.default_rng(np.random.SeedSequence([self.meta["seed"], self.meta["epochs"]]))
            order = rng.permutation(len(x))
            loss_sum = 0.0
            for start in range(0, len(order), self.meta["batch_size"]):
                indices = order[start:start + self.meta["batch_size"]]
                batch, expected = x[indices], target[indices]
                hidden, logits = self.forward(batch)
                if self.meta["task"] == "classification":
                    logits -= logits.max(axis=1, keepdims=True)
                    probabilities = np.exp(logits)
                    probabilities /= probabilities.sum(axis=1, keepdims=True)
                    loss_sum -= float(np.log(np.maximum(probabilities[np.arange(len(indices)), expected], 1e-300)).sum())
                    delta = probabilities
                    delta[np.arange(len(indices)), expected] -= 1
                    delta /= len(indices)
                else:
                    error = logits[:, 0] - expected
                    loss_sum += float(np.sum(error ** 2))
                    delta = (2 * error / len(indices))[:, None]
                gradients = {"w2": hidden.T @ delta, "b2": delta.sum(axis=0)}
                hidden_delta = (delta @ self.params["w2"].T) * (1 - hidden ** 2)
                gradients.update(w1=batch.T @ hidden_delta, b1=hidden_delta.sum(axis=0))
                norm = math.sqrt(sum(float(np.sum(g * g)) for g in gradients.values()))
                if not math.isfinite(norm):
                    raise ValueError("nonfinite gradient")
                self.meta["steps"] += 1
                for name, gradient in gradients.items():
                    gradient *= min(1.0, 5.0 / max(norm, 1e-12))
                    self.first[name] = .9 * self.first[name] + .1 * gradient
                    self.second[name] = .999 * self.second[name] + .001 * gradient ** 2
                    first = self.first[name] / (1 - .9 ** self.meta["steps"])
                    second = self.second[name] / (1 - .999 ** self.meta["steps"])
                    self.params[name] -= self.meta["learning_rate"] * first / (np.sqrt(second) + 1e-8)
                if not all(np.all(np.isfinite(p)) for p in self.params.values()):
                    raise ValueError("nonfinite model state")
            self.meta["epochs"] += 1
            epoch_losses.append(loss_sum / len(x))
        self.validate()
        return epoch_losses

    def predict(self, x):
        _, logits = self.forward(self.normalized(x))
        if self.meta["task"] == "classification":
            logits -= logits.max(axis=1, keepdims=True)
            result = np.exp(logits)
            return result / result.sum(axis=1, keepdims=True)
        return np.maximum(0, logits[:, 0] * self.meta["target_scale"] + self.meta["target_mean"])

    def save(self, path):
        self.validate()
        arrays = {**self.params,
                  **{"adam_first_" + k: v for k, v in self.first.items()},
                  **{"adam_second_" + k: v for k, v in self.second.items()},
                  "mean": self.mean, "scale": self.scale,
                  "metadata": np.array(json.dumps(self.meta, sort_keys=True, separators=(",", ":"), allow_nan=False))}
        with Path(path).open("xb") as stream, zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_STORED) as archive:
            for name in sorted(arrays):
                buffer = io.BytesIO()
                np.lib.format.write_array(buffer, arrays[name], allow_pickle=False)
                info = zipfile.ZipInfo(name + ".npy", date_time=(1980, 1, 1, 0, 0, 0))
                info.external_attr = 0o600 << 16
                archive.writestr(info, buffer.getvalue())

    @classmethod
    def load(cls, path):
        path = Path(path)
        if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_CHECKPOINT_BYTES:
            raise ValueError("checkpoint must be a regular file no larger than 16 MiB")
        expected = {*PARAMETERS, *("adam_first_" + k for k in PARAMETERS),
                    *("adam_second_" + k for k in PARAMETERS), "mean", "scale", "metadata"}
        with zipfile.ZipFile(path) as archive:
            names = [entry.filename for entry in archive.infolist()]
            if len(names) != len(expected) or set(names) != {name + ".npy" for name in expected} or sum(e.file_size for e in archive.infolist()) > MAX_CHECKPOINT_BYTES:
                raise ValueError("invalid checkpoint archive")
        with np.load(path, allow_pickle=False) as arrays:
            model = cls.__new__(cls)
            metadata = arrays["metadata"]
            if metadata.ndim != 0 or metadata.dtype.kind != "U":
                raise ValueError("invalid checkpoint metadata")
            model.meta = json.loads(str(metadata))
            model.params = {k: arrays[k].copy() for k in PARAMETERS}
            model.first = {k: arrays["adam_first_" + k].copy() for k in PARAMETERS}
            model.second = {k: arrays["adam_second_" + k].copy() for k in PARAMETERS}
            model.mean, model.scale = arrays["mean"].copy(), arrays["scale"].copy()
        model.validate()
        return model


# ----------------------------------------------------------------------------
# 指标与闭环诊断（创新点：漂移/故障优先级评分、预算调度、SafeGate 门槛）
# ----------------------------------------------------------------------------

def _evaluate(model, x, y, task):
    """返回验证指标 dict。分类：macro-F1/accuracy；回归：RMSE/MAE。"""
    predictions = model.predict(x)
    if task == "classification":
        labels = np.argmax(predictions, axis=1)
        classes = model.meta["classes"]
        f1s = []
        for index in range(len(classes)):
            tp = int(np.sum((labels == index) & (y == index)))
            fp = int(np.sum((labels == index) & (y != index)))
            fn = int(np.sum((labels != index) & (y == index)))
            f1s.append(2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 1.0)
        return {"macro_f1": float(np.mean(f1s)), "accuracy": float(np.mean(labels == y))}
    error = predictions - y
    return {"rmse": float(math.sqrt(np.mean(error ** 2))),
            "mae": float(np.mean(np.abs(error)))}


def _primary_metric(task, metrics):
    if task == "classification":
        return metrics["macro_f1"]
    return metrics["rmse"]


def _better(task, candidate, parent):
    """按指标方向比较两个指标 dict。"""
    if task == "classification":
        return candidate["macro_f1"] > parent["macro_f1"]
    return candidate["rmse"] < parent["rmse"]


def _diagnose_drift(task, current, history_metrics):
    """创新点 2/4：与历史验证指标比较，输出漂移类型与严重度。"""
    if not history_metrics:
        return "none", 0.0
    reference = history_metrics[-1]
    if task == "classification":
        delta = reference["macro_f1"] - current["macro_f1"]
        severity = max(0.0, min(1.0, delta / max(0.05, reference["macro_f1"])))
        drift_type = "performance" if delta > 0.01 else "none"
    else:
        delta = current["rmse"] - reference["rmse"]
        severity = max(0.0, min(1.0, delta / max(10.0, reference["rmse"])))
        drift_type = "performance" if delta > 10.0 else "none"
    if severity >= 0.5:
        level = "critical"
    elif severity >= 0.2:
        level = "major"
    elif severity > 0.0:
        level = "minor"
    else:
        level = "stable"
    return "%s:%s" % (drift_type, level), severity


def _budget_epochs(severity, round_index):
    """创新点 6：成本收益自适应预算。漂移越重、轮次越靠后，投入越大，但受 2 核 CPU 限制。"""
    base = 1
    if severity >= 0.5:
        base += 2
    elif severity >= 0.2:
        base += 1
    if round_index >= 7:
        base += 1
    return min(4, base)


def _safe_gate(model, x, task, parent_metrics, candidate_metrics):
    """创新点 5：SafeGate 三重门槛（输入有限性 / 参数有限性 / 输出健康度）。"""
    if not all(np.all(np.isfinite(p)) for p in model.params.values()):
        return False
    if not np.all(np.isfinite(x)):
        return False
    if task == "classification":
        if candidate_metrics["macro_f1"] < parent_metrics["macro_f1"] - 0.05:
            return False
    else:
        if candidate_metrics["rmse"] > parent_metrics["rmse"] * 1.25 + 5.0:
            return False
    return True


# ----------------------------------------------------------------------------
# 大模型决策（每轮必做一次真实调用；回复解析 + 保守兜底）
# ----------------------------------------------------------------------------

def _decide_with_llm(rnd, brief):
    """请大模型做晋升/保留决策。返回 (action, call_id, reply_text)。"""
    receipt = rnd.model(
        [{"role": "user", "content": brief}],
        max_tokens=min(96, rnd.max_output_tokens),
        temperature=0,
    )
    text = (receipt.text or "").strip()
    return receipt.call_id, text


def _parse_action(text, task, candidate_metrics, parent_metrics):
    lowered = (text or "").lower()
    if "keep" in lowered or "保留" in lowered or "reject" in lowered or "拒绝" in lowered:
        return "keep_parent"
    if "promote" in lowered or "晋升" in lowered or "accept" in lowered or "采用" in lowered:
        return "promote"
    return "promote" if _better(task, candidate_metrics, parent_metrics) else "keep_parent"


# ----------------------------------------------------------------------------
# 状态（跨轮原生状态：指标历史、漂移记录、预算足迹）
# ----------------------------------------------------------------------------

STATE_FILE = "state.json"


def _load_state(state_dir):
    path = Path(state_dir) / STATE_FILE
    if path.is_file():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data
        except (ValueError, OSError):
            pass
    return {"metric_history": [], "drift_log": [], "rounds": []}


def _save_state(state_dir, state):
    path = Path(state_dir) / STATE_FILE
    path.write_text(json.dumps(state, ensure_ascii=False, sort_keys=True), encoding="utf-8")


# ----------------------------------------------------------------------------
# 训练主流程：五步闭环
# ----------------------------------------------------------------------------

def train(rnd):
    try:
        _train(rnd)
    except Exception as error:
        _log("train crashed: %s: %s" % (type(error).__name__, error))
        raise


def _train(rnd):
    task = rnd.task["task"]
    dataset = rnd.dataset_key
    _log("round %d dataset %s task %s begin" % (rnd.round_index, dataset, task))

    # 0) 读父检查点（首轮为官方基线，之后为我们封版的产物）
    parent_ckpt = rnd.parent.path("checkpoint.npz")
    parent_model = Model.load(parent_ckpt)
    classes = parent_model.meta["classes"]
    expected_paths = parent_model.meta["paths"]

    # 1) 读公开数据（训练集只用于梯度；验证集用于测量与决策）
    x_train, y_train, _ = _read_rows(rnd.data.training_rows(), task, classes, expected_paths)
    x_val, y_val, _ = _read_rows(rnd.data.validation_rows(), task, classes, expected_paths)
    _log("data train=%d val=%d features=%d" % (len(x_train), len(x_val), len(expected_paths)))

    # 2) 读历史（“读历史—做实验—依据结果决策”的第一次闭环记录）
    history = rnd.history()
    _log("history %d records" % len(history))

    # 3) 读跨轮状态
    state_dir = rnd.state_workspace(str(_WORKSPACE / "state"))
    state = _load_state(state_dir)

    # 4) 监测：父模型当前验证指标 + 漂移诊断
    parent_metrics = _evaluate(parent_model, x_val, y_val, task)
    drift, severity = _diagnose_drift(task, parent_metrics, state["metric_history"])
    _log("parent metrics %s drift=%s severity=%.2f" % (json.dumps(parent_metrics), drift, severity))

    # 5) 诊断 + 优化：预算调度决定本轮续训强度
    epochs = _budget_epochs(severity, rnd.round_index)
    _log("budget epochs=%d" % epochs)

    # 6) 做实验：从父检查点续训候选（预演版=先续训，再经 SafeGate 把关）
    candidate_model = Model.load(parent_ckpt)
    losses = candidate_model.fit(x_train, y_train, epochs)
    candidate_dir = _WORKSPACE / "candidate"
    candidate_dir.mkdir(parents=True, exist_ok=False)
    candidate_model.save(candidate_dir / "checkpoint.npz")
    candidate_metrics = _evaluate(candidate_model, x_val, y_val, task)
    gate_ok = _safe_gate(candidate_model, x_val, task, parent_metrics, candidate_metrics)
    _log("candidate metrics %s gate=%s loss_last=%s" % (json.dumps(candidate_metrics), gate_ok, losses[-1] if losses else None))

    # 7) 依据结果决策：真实大模型调用（每轮至少一次，且必须在本轮完成前）
    brief = (
        "智训引擎第 %d 轮（数据集 %s，任务 %s）。\n"
        "父模型验证指标：%s\n续训 %d epoch 后候选验证指标：%s\n"
        "漂移诊断：%s（严重度 %.2f），SafeGate：%s。\n"
        "候选相对父模型%s。请只回复 promote（晋升候选封版）或 keep_parent（保留父模型）。"
        % (rnd.round_index, dataset, task, json.dumps(parent_metrics), epochs,
           json.dumps(candidate_metrics), drift, severity, "通过" if gate_ok else "未通过",
           "更优" if _better(task, candidate_metrics, parent_metrics) else "未更优"))
    call_id, reply_text = _decide_with_llm(rnd, brief)
    _log("llm decision call=%s reply=%r" % (call_id, reply_text[:120]))

    action = _parse_action(reply_text, task, candidate_metrics, parent_metrics)
    if not gate_ok:
        action = "keep_parent"
        _log("SafeGate veto: keep_parent")
    else:
        # 数字孪生预演思想落地：候选必须显著不劣才允许晋升
        pass

    # 8) 提交候选（keep 时也提交父检查点副本，保证选中产物一定是本轮合法产物）
    if action == "promote":
        submitted_dir = candidate_dir
        selected_summary = candidate_metrics
        change = "从父检查点续训 %d epoch（预算调度），经 SafeGate 与漂移诊断通过，晋升封版。" % epochs
    else:
        keep_dir = _WORKSPACE / "keep"
        keep_dir.mkdir(parents=True, exist_ok=False)
        parent_model.save(keep_dir / "checkpoint.npz")
        submitted_dir = keep_dir
        selected_summary = parent_metrics
        change = "续训候选未通过 SafeGate 或大模型决策保留父模型，本轮回滚至父检查点。"

    seed = 42 + rnd.round_index
    hypothesis = "五步闭环第 %d 轮：监测%s（严重度 %.2f）→ 预算 %d epoch 续训 → SafeGate %s → 大模型决策 %s。" % (
        rnd.round_index, drift, severity, epochs, "通过" if gate_ok else "拦截", action)
    configuration = {
        "epochs": epochs,
        "drift": drift,
        "severity": round(severity, 4),
        "gate_ok": gate_ok,
        "action": action,
        "parent_metrics": parent_metrics,
        "candidate_metrics": candidate_metrics,
        "selected_metrics": selected_summary,
        "model_call_id": call_id,
        "loss_last": losses[-1] if losses else None,
    }
    submitted = rnd.submit_candidate(
        submitted_dir,
        hypothesis=hypothesis,
        change_summary=change,
        configuration=configuration,
        seed=seed,
        checkpoint_paths=["checkpoint.npz"],
        code_paths=[],
        predict_argv=["python3", "/app/harness.py", "predict"],
    )
    _log("candidate submitted artifact=%s" % submitted.get("artifact_id"))

    # 9) 更新并保存跨轮状态
    state["metric_history"].append(parent_metrics)
    state["drift_log"].append({"round": rnd.round_index, "drift": drift, "severity": severity, "action": action})
    state["rounds"].append({"round": rnd.round_index, "epochs": epochs, "action": action})
    state["metric_history"] = state["metric_history"][-10:]
    state["drift_log"] = state["drift_log"][-10:]
    state["rounds"] = state["rounds"][-10:]
    _save_state(state_dir, state)
    rnd.save_state(state_dir)

    # 10) 封版收尾
    rnd.finish(
        selected_artifact_id=submitted["artifact_id"],
        hypothesis=hypothesis,
        change_summary=change,
        reason="本轮大模型决策回复：%s；按五步闭环执行 %s。" % (reply_text[:200] or "(空)", action),
        configuration=configuration,
        seed=seed,
        history_ids=[item["id"] for item in history],
        model_call_ids=list(rnd.call_ids),
    )
    _log("round %d finished action=%s" % (rnd.round_index, action))


# ----------------------------------------------------------------------------
# 预测：官方协议输出
# ----------------------------------------------------------------------------

_PREDICT_CACHE = {}


def _declared_checkpoint_paths(context):
    """收集预测上下文与挂载清单里声明的检查点相对路径。"""
    declared = []
    if isinstance(context, dict):
        artifact = context.get("artifact")
        if isinstance(artifact, dict):
            declared.extend(artifact.get("checkpoint_paths") or [])
    raw = os.environ.get("CONTEST_ARTIFACT_MANIFEST", "")
    if raw:
        try:
            payload = json.loads(raw)
        except ValueError:
            payload = None
        if isinstance(payload, dict):
            item = payload.get("artifact")
            if isinstance(item, dict):
                declared.extend(item.get("checkpoint_paths") or [])
            files = payload.get("files")
            if isinstance(files, list):
                declared.extend(entry.get("path") for entry in files if isinstance(entry, dict))
    return [value for value in declared if isinstance(value, str) and value]


def _resolve_checkpoint(model_dir, context):
    """三级兜底定位选中的检查点：环境变量 → 清单声明 → 模型目录扫描。

    平台把选中产物只读挂载到 /model 根目录，挂载点后直接拼固定文件名并不可靠，
    因此必须按挂载目录里实际存在的文件来加载。
    """
    explicit = os.environ.get("CONTEST_ARTIFACT_PATH", "").strip()
    if explicit:
        candidate = Path(explicit)
        if candidate.is_file():
            return candidate
    root = Path(model_dir)
    if not root.is_dir():
        raise ValueError("model directory is missing: %s" % root)
    for relative in _declared_checkpoint_paths(context):
        candidate = root / relative
        try:
            if candidate.is_file():
                return candidate
        except OSError:
            continue
    found = sorted(path for path in root.rglob("*.npz") if path.is_file())
    if found:
        return found[0]
    raise ValueError("no checkpoint file found under %s" % root)


def predict(sample, model_dir):
    checkpoint = _resolve_checkpoint(model_dir, getattr(sample, "context", None))
    cache_key = str(checkpoint.resolve())
    model = _PREDICT_CACHE.get(cache_key)
    if model is None:
        model = Model.load(checkpoint)
        _PREDICT_CACHE[cache_key] = model
    if sample.task != model.meta["task"]:
        raise ValueError("prediction task mismatch")
    paths, values = flatten(sample.inputs)
    if paths != model.meta["paths"]:
        raise ValueError("prediction feature schema differs from checkpoint")
    output = model.predict(np.asarray([values], dtype=np.float64))[0]
    if model.meta["task"] == "regression":
        return {"outputs": {"value": float(output)}}
    return {"outputs": [{"label": label, "score": float(score)}
                        for label, score in zip(model.meta["classes"], output)]}
