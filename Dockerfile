FROM python:3.13-slim
COPY --from=ghcr.io/astral-sh/uv:0.11.28 /uv /bin/uv
WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PROJECT_ENVIRONMENT=/opt/venv
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project
COPY src ./src
COPY policies ./policies
RUN uv sync --frozen --no-dev
ENV PATH="/opt/venv/bin:$PATH" RAG_DIR=/data/rag_storage REVIEW_QUEUE_PATH=/data/queue/review_queue.jsonl
RUN useradd --create-home --uid 1000 app && mkdir -p /data/queue /data/rag_storage && chown -R app /data
USER app
EXPOSE 8000
CMD ["compliance-agent", "serve", "--host", "0.0.0.0", "--port", "8000"]
