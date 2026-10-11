FROM python:3.13-slim
COPY --from=ghcr.io/astral-sh/uv:0.11.28 /uv /bin/uv
WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PROJECT_ENVIRONMENT=/opt/venv
# EXTRAS=local-asr bundles faster-whisper for transcribing video audio without Azure
ARG EXTRAS=""
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project ${EXTRAS:+--extra $EXTRAS}
COPY src ./src
COPY policies ./policies
RUN uv sync --frozen --no-dev ${EXTRAS:+--extra $EXTRAS}
ENV PATH="/opt/venv/bin:$PATH" RAG_DIR=/data/rag_storage REVIEW_DB_PATH=/data/queue/reviews.db \
    HF_HOME=/data/queue/hf-cache
RUN useradd --create-home --uid 1000 app && mkdir -p /data/queue /data/rag_storage && chown -R app /data
USER app
EXPOSE 8000
CMD ["compliance-agent", "serve", "--host", "0.0.0.0", "--port", "8000"]
