# syntax=docker/dockerfile:1.7

# --- Builder: resolve and build dependencies using uv ---
FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim AS builder

WORKDIR /app

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

# 1) Copy only dependency manifests to maximize layer cache reuse
COPY uv.lock pyproject.toml ./

# 2) Install dependencies into a virtual environment
# --frozen: strict usage of lock file
# --no-dev: exclude dev dependencies
# --no-install-project: strictly install dependencies, not the app itself
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project


# --- Runtime: minimal python image + non-root user ---
FROM python:3.13-slim AS runtime

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TOKENIZERS_PARALLELISM=false \
    HF_HOME=/app/.cache/huggingface \
    TRANSFORMERS_CACHE=/app/.cache/huggingface/hub \
    SENTENCE_TRANSFORMERS_HOME=/app/.cache/sentence-transformers \
    PATH="/app/.venv/bin:$PATH"

# Create an unprivileged user
RUN useradd --create-home --uid 10001 appuser

# Copy the environment from builder
COPY --from=builder /app/.venv /app/.venv

# Now copy app source (kept late to avoid busting dependency layers)
COPY --chown=appuser:appuser app.py api_schema.py base_model.py external_model.py similarity_search.py utils.py ./
COPY --chown=appuser:appuser prompt_template.txt prompt_judge_template.txt ./

# Ensure cache dirs are writable
RUN mkdir -p /app/.cache/huggingface /app/.cache/sentence-transformers \
    && chown -R appuser:appuser /app/.cache

USER appuser

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers"]
