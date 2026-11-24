# 智训引擎（工业模型自演化系统）

第二届北京市大学生"人工智能+"创新大赛 · 赛题一《智训引擎》参赛作品。
面向工业场景的可续训模型自演化闭环系统，按官方 contest-harness SDK 1.0.4 协议接入评测平台。

## 系统总览

从平台提供的工业模型检查点出发，在 CPU 环境下持续自演化：

```
读历史 → 监测（验证指标） → 诊断（漂移/故障） → 优化（预算调度续训） → 评估（SafeGate） → 大模型决策封版
```

六个创新点：

1. 数字孪生仿真场 + 算力双模式
2. 漂移/故障优先级评分 + 故障分类归纳
3. 可解释根因归因
4. 漂移前瞻预警（趋势 / 预计恶化时间 / 预漂移）
5. SafeGate 三重拒答（输入越界 / 低置信 / 输出越界）
6. 成本收益自适应预算调度

## 仓库结构

```
├── Dockerfile               # Harness 提交镜像构建（入口 /app/harness.py）
├── contest_submission.py    # 平台参赛模块：train(rnd) 五步闭环 + predict(sample, model_dir)
├── harness.py               # 官方入口模板（describe / train / predict）
└── workspace/               # 本地完整五步闭环系统（含 engine/ 全部模块、docs、run.py）
```

- `contest_submission.py` 内的模型为官方 `contest.cpu-mlp.v1` 检查点格式（单隐层 MLP + Adam），
  从平台父检查点严格续训，格式与协议完全兼容。
- 每轮 `train(rnd)` 必须至少调用一次平台大模型做「晋升 / 保留」决策，决策、指标与配置
  全部如实写入 `round_complete` 记录。
- 合规红线：梯度更新只用平台公开训练集；验证集只用于测量与决策；不获取隐藏测试标签。

## 构建 Harness 提交镜像

```bash
# 构建上下文需包含 SDK 解压目录（contest-harness-sdk-1.0.4/）与 submission/
docker build -t zhixun-engine:harness-v2 .
docker run --rm zhixun-engine:harness-v2 describe   # 应输出 workspace.v2 协议描述
docker save -o zhixun-engine-harness-image.tar zhixun-engine:harness-v2
```

## 本地离线测试

```bash
pip install .   # 在 SDK 解压目录安装 contest_harness
python -m unittest tests.test_sdk tests.test_contract_regressions tests.test_transport_regressions -v
```

## 本地完整系统复现

```bash
cd workspace
python -m venv .venv && .venv/Scripts/activate   # Windows
pip install -r requirements.txt
python run.py --dataset official-a --windows 5 --out out/a
python run.py --dataset official-b --windows 5 --out out/b
python run.py --dataset official-c --windows 5 --out out/c
```

官方数据集按赛事提供的 JSONL 放在 `official-data/{A-gas,B-bearing,C-engine}/` 下。

## 实验基线（本地数据包口径）

| 数据集 | 指标 | 官方基线 | 接入闭环后 |
|---|---|---|---|
| A 气体识别 | macro-F1 | 0.7140 | 保持/提升，见 runs 记录 |
| B 轴承健康 | RMSE / MAE | 681.16 / 449.63 | 572.94 / 449.63 |
| C 发动机寿命 | RMSE / MAE | 52.87 / 38.23 | 51.89 / 38.23 |
