# 智训引擎 Harness 提交镜像（SDK 1.0.4 协议）
# 构建上下文：harness-sdk/ 目录
FROM python:3.12-slim-bookworm

# 运行时库：libgomp1 供 numpy/OpenBLAS 使用；libatomic1 与 ca-certificates 按官方示例补齐
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 libatomic1 ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# 安装 contest_harness SDK（零依赖，纯标准库）
WORKDIR /opt/contest-harness
COPY contest-harness-sdk-1.0.4/contest_harness ./contest_harness
COPY contest-harness-sdk-1.0.4/pyproject.toml ./pyproject.toml
RUN pip install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple . numpy scikit-learn \
    && rm -rf /opt/contest-harness

# 固定入口与选手模块（/app 只读，平台挂载）
WORKDIR /app
COPY contest-harness-sdk-1.0.4/template/harness.py /app/harness.py
COPY submission/contest_submission.py /app/contest_submission.py

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
USER 65534:65534
ENTRYPOINT ["python3", "/app/harness.py"]
