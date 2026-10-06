# ---------- 构建阶段：装依赖（利用缓存层，改了代码不必重装依赖）----------
FROM python:3.11-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build
# CPU 版 torch 单独装，避免默认拉 CUDA 巨包
RUN pip install --prefix=/install "torch>=2.2.0" "sentence-transformers>=3.0.0" \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        -i https://pypi.tuna.tsinghua.edu.cn/simple

COPY requirements-core.txt ./
RUN pip install --prefix=/install -r requirements-core.txt \
        -i https://pypi.tuna.tsinghua.edu.cn/simple

# ---------- 运行阶段 ----------
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    HF_HOME=/app/.cache/huggingface \
    HF_ENDPOINT=https://hf-mirror.com \
    TOKENIZERS_PARALLELISM=false

WORKDIR /app

# jieba 在没有 wheel 时需要编译器；装上先备着（slim 镜像默认没有）
RUN apt-get update \
 && apt-get install -y --no-install-recommends gcc g++ \
 && rm -rf /var/lib/apt/lists/*

COPY --from=builder /install /usr/local
COPY . .

# 非 root 运行
RUN useradd -m -u 10001 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD python -c "import httpx,sys; sys.exit(0 if httpx.get('http://127.0.0.1:8000/api/health',timeout=4).status_code==200 else 1)"

CMD ["python", "-m", "scripts.serve", "--host", "0.0.0.0", "--port", "8000"]
