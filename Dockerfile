FROM python:3.11-slim-bookworm AS builder
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ libffi-dev libssl-dev python3-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# 过滤 Windows-only + GUI/桌面依赖（Linux 容器里通常用不到）
RUN grep -v -E "^(pywin32|uiautomation|PyAutoGUI|pygame)([<=>~! ].*)?$" requirements.txt \
    > requirements-docker.txt

RUN pip install --no-cache-dir -r requirements-docker.txt


FROM python:3.11-slim-bookworm
WORKDIR /app

# 运行时常见依赖（按需增减）
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /usr/local /usr/local
COPY . .

RUN mkdir -p data logs data/config data/avatars

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TZ=Asia/Shanghai \
    PYTHONPATH=/app/src:/app

EXPOSE 16667 8502
CMD ["python", "run_config_web.py"]
