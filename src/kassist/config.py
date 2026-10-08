"""Typed application configuration.

Sources, highest precedence first: environment variables, `.env`, `configs/default.yaml`
(path overridable with KASSIST_CONFIG), then the defaults below. Secrets are only ever read
from the environment and are held as `SecretStr` so they never appear in logs or reprs.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, SecretStr
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)

# Repo root in development; set KASSIST_HOME when the package is installed elsewhere (Docker).
PROJECT_ROOT = Path(os.environ.get("KASSIST_HOME", Path(__file__).resolve().parents[2]))
DEFAULT_CONFIG_FILE = PROJECT_ROOT / "configs" / "default.yaml"


class LLMSettings(BaseModel):
    provider: Literal["openai", "azure", "ollama"] = "openai"
    fallback_to_ollama: bool = True
    router_model: str = "gpt-4.1-mini"
    synth_model: str = "gpt-4.1"
    # Azure OpenAI addresses models by *deployment name* (chosen when deploying in Azure AI Foundry)
    azure_router_deployment: str | None = None
    azure_synth_deployment: str | None = None
    azure_api_version: str = "2024-10-21"
    ollama_router_model: str = "qwen2.5:7b-instruct"
    ollama_synth_model: str = "qwen2.5:7b-instruct"
    # Sent to OpenAI/Azure only for non-reasoning models: GPT-5 / o-series reject any value but the
    # default, so it is omitted for them automatically (see llm/client.py). null = never send it.
    temperature: float | None = 0.0
    timeout_s: float = 60
    max_retries: int = 2


class EmbeddingSettings(BaseModel):
    dense_model: str = "BAAI/bge-small-en-v1.5"
    sparse_model: str = "Qdrant/bm25"
    reranker_model: str = "Xenova/ms-marco-MiniLM-L-6-v2"
    batch_size: int = 32
    cache_dir: str | None = None


class ChunkingSettings(BaseModel):
    max_tokens: int = 380
    overlap_tokens: int = 60
    min_tokens: int = 50


class RetrievalSettings(BaseModel):
    prefetch_k: int = 30
    candidate_k: int = 20
    rerank_max_chars: int | None = None  # truncate passages fed to the cross-encoder (CPU latency)
    top_k: int = 6
    min_relevance: float = 0.10
    keep_relevance: float = 0.02


class AgentSettings(BaseModel):
    max_subtasks: int = 4
    max_iterations: int = 2
    max_tool_calls: int = 8
    max_evidence_chunks: int = 12
    max_history_turns: int = 4  # previous turns given to the follow-up rewriter
    follow_up_count: int = 3  # suggested follow-up questions per answer
    follow_up_min_confidence: float = 0.4  # below this blended confidence, no follow-ups are suggested


class ConversationSettings(BaseModel):
    ttl_hours: float = 24  # conversations idle longer than this are deleted
    max_turns: int = 20  # turns kept per conversation


class APISettings(BaseModel):
    max_question_chars: int = 2000
    rate_limit: str = "30/minute"
    cors_origins: list[str] = Field(default_factory=list)
    index_refresh_s: float = 15  # how often the API checks whether ingestion changed the index


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        extra="ignore",
    )

    app_env: Literal["dev", "test", "prod"] = "dev"
    log_level: str = "INFO"
    log_json: bool = True
    # Log raw question text? Off by default: questions may contain personal data.
    log_questions: bool = False

    # Secrets / endpoints (environment only)
    openai_api_key: SecretStr | None = None
    openai_base_url: str | None = None
    azure_openai_endpoint: str | None = None  # https://<resource>.openai.azure.com/
    azure_openai_api_key: SecretStr | None = None
    ollama_base_url: str = "http://localhost:11434"
    api_key: SecretStr | None = Field(default=None, description="If set, /v1 requires X-API-Key")

    qdrant_url: str | None = None  # e.g. http://qdrant:6333; if unset, embedded mode is used
    qdrant_path: str = str(PROJECT_ROOT / ".qdrant")  # embedded (local) mode storage
    qdrant_api_key: SecretStr | None = None
    collection: str = "knowledge_base"
    corpus_dir: str = str(PROJECT_ROOT / "data" / "corpus")
    parse_cache_dir: str | None = str(PROJECT_ROOT / "data" / ".parse_cache")
    conversation_db_path: str = str(PROJECT_ROOT / "data" / "conversations.sqlite")

    llm: LLMSettings = LLMSettings()
    embeddings: EmbeddingSettings = EmbeddingSettings()
    chunking: ChunkingSettings = ChunkingSettings()
    retrieval: RetrievalSettings = RetrievalSettings()
    agent: AgentSettings = AgentSettings()
    conversations: ConversationSettings = ConversationSettings()
    api: APISettings = APISettings()

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        yaml_file = Path(os.environ.get("KASSIST_CONFIG", DEFAULT_CONFIG_FILE))
        sources: list[PydanticBaseSettingsSource] = [init_settings, env_settings, dotenv_settings]
        if yaml_file.exists():
            sources.append(YamlConfigSettingsSource(settings_cls, yaml_file=yaml_file))
        sources.append(file_secret_settings)
        return tuple(sources)


@lru_cache
def get_settings() -> Settings:
    return Settings()
