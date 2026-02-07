# syntax=docker/dockerfile:1.7

# --- Builder: resolve and build wheels using uv ---
FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim AS builder

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

# 1) Copy only dependency manifests to maximize layer cache reuse
COPY uv.lock pyproject.toml ./

# 2) Build wheels for all production dependencies (fast to install in runtime)
RUN --mount=type=cache,target=/root/.cache/uv \
    uv export --frozen --no-dev --no-editable -o requirements.lock

RUN --mount=type=cache,target=/root/.cache/uv \
    python -m pip wheel --no-deps -r requirements.lock -w /wheels


# --- Runtime: minimal python image + non-root user ---
FROM python:3.13-slim AS runtime

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TOKENIZERS_PARALLELISM=false \
    HF_HOME=/app/.cache/huggingface \
    TRANSFORMERS_CACHE=/app/.cache/huggingface/hub \
    SENTENCE_TRANSFORMERS_HOME=/app/.cache/sentence-transformers

# Create an unprivileged user
RUN useradd --create-home --uid 10001 appuser

# Create and populate a venv from prebuilt wheels (no network needed here)
RUN python -m venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH"

COPY --from=builder /wheels /wheels
COPY --from=builder /app/requirements.lock /app/requirements.lock
RUN python -m pip install --no-cache-dir --no-index --find-links=/wheels -r /app/requirements.lock \
    && rm -rf /wheels

# Now copy app source (kept late to avoid busting dependency layers)
COPY --chown=appuser:appuser app.py api_schema.py base_model.py external_model.py similarity_search.py utils.py ./
COPY --chown=appuser:appuser prompt_template.txt prompt_judge_template.txt ./

# Ensure cache dirs are writable
RUN mkdir -p /app/.cache/huggingface /app/.cache/sentence-transformers \
    && chown -R appuser:appuser /app/.cache

USER appuser

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers"]
