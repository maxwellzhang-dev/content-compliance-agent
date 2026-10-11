"""Settings from the environment (and `.env`). Azure OpenAI is used when
AZURE_OPENAI_ENDPOINT is set; otherwise the public OpenAI API."""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

REASONING_PREFIXES = ("gpt-5", "o1", "o3", "o4")


def _float(name: str, default: float) -> float:
    return float(os.getenv(name, default))


@dataclass(frozen=True)
class Settings:
    provider: str  # "azure" | "openai"
    judge_model: str  # model or Azure deployment that makes the decision
    index_model: str  # model LightRAG uses for entity extraction at indexing time
    embedding_model: str
    azure_endpoint: str | None
    api_version: str | None
    openai_base_url: str | None
    rag_dir: str
    samples: int
    min_verdict_probability: float
    min_agreement: float
    min_groundedness: float
    review_db_path: str
    review_ui_url: str  # where the /review page is reachable, for links in answers
    api_key: str | None  # bearer token the agent's own API expects, if set
    speech_key: str | None
    speech_region: str | None
    speech_language: str
    allow_private_video_urls: bool

    @property
    def judge_is_reasoning(self) -> bool:
        """Reasoning models reject `temperature` and do not return logprobs."""
        override = os.getenv("JUDGE_IS_REASONING")
        if override is not None:
            return override.lower() in {"1", "true", "yes"}
        return self.judge_model.lower().startswith(REASONING_PREFIXES)

    @classmethod
    def from_env(cls) -> Settings:
        load_dotenv()
        azure = bool(os.getenv("AZURE_OPENAI_ENDPOINT"))
        default_chat = (
            os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4.1-mini") if azure else "gpt-4.1-mini"
        )
        judge = os.getenv("JUDGE_MODEL", default_chat)
        return cls(
            provider="azure" if azure else "openai",
            judge_model=judge,
            index_model=os.getenv("INDEX_MODEL", judge),
            embedding_model=os.getenv(
                "AZURE_EMBEDDING_DEPLOYMENT" if azure else "EMBEDDING_MODEL",
                "text-embedding-3-small",
            ),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2025-04-01-preview"),
            openai_base_url=os.getenv("OPENAI_BASE_URL"),
            rag_dir=os.getenv("RAG_DIR", "./rag_storage"),
            samples=int(os.getenv("CONSISTENCY_SAMPLES", "5")),
            min_verdict_probability=_float("MIN_VERDICT_PROBABILITY", 0.85),
            min_agreement=_float("MIN_AGREEMENT", 0.6),
            min_groundedness=_float("MIN_GROUNDEDNESS", 0.75),
            review_db_path=os.getenv("REVIEW_DB_PATH", "./reviews.db"),
            review_ui_url=os.getenv("REVIEW_UI_URL", "http://localhost:8000/review"),
            api_key=os.getenv("AGENT_API_KEY") or None,
            speech_key=os.getenv("AZURE_SPEECH_KEY") or None,
            speech_region=os.getenv("AZURE_SPEECH_REGION") or None,
            speech_language=os.getenv("SPEECH_LANGUAGE", "en-US"),
            allow_private_video_urls=os.getenv("ALLOW_PRIVATE_VIDEO_URLS", "").lower()
            in {"1", "true", "yes"},
        )


def make_async_client(settings: Settings):
    from openai import AsyncAzureOpenAI, AsyncOpenAI

    if settings.provider == "azure":
        return AsyncAzureOpenAI(
            azure_endpoint=settings.azure_endpoint,
            api_key=os.environ["AZURE_OPENAI_API_KEY"],
            api_version=settings.api_version,
            max_retries=6,  # small student-tier quotas hit 429 under eval load
        )
    return AsyncOpenAI(base_url=settings.openai_base_url, max_retries=6)
