"""LLM access behind a small interface.

* Two tiers: "fast" (routing, planning, reflection) and "strong" (final grounded answer).
  Cheap model for control decisions, strong model only where answer quality is visible.
* Provider is config-driven: OpenAI (api.openai.com), Azure OpenAI (deployment names), or
  Ollama for local / air-gapped use.
* With fallback_to_ollama=true, each OpenAI/Azure call transparently retries on Ollama if the
  primary fails (outage, rate limit, network). Same prompts, same schema.
* All calls return validated Pydantic objects (structured output), never free text to parse.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from typing import Any, Literal, Protocol, TypeVar

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.runnables import Runnable, RunnableConfig
from pydantic import BaseModel

from kassist.config import LLMSettings, Settings
from kassist.observability import get_logger

log = get_logger(__name__)

Tier = Literal["fast", "strong"]
# Reasoning models (GPT-5 family, o1/o3/o4) only accept the default temperature.
_REASONING_MODEL = re.compile(r"^(gpt-5|o[1-9])", re.I)
T = TypeVar("T", bound=BaseModel)


class LLMUnavailableError(RuntimeError):
    """Every configured provider failed for a call."""


class LLMStreamInterruptedError(LLMUnavailableError):
    """The provider failed after part of the answer had already been streamed."""


class LLMConfigError(ValueError):
    """The selected provider is missing required settings."""


class LLMClient(Protocol):
    def structured(self, schema: type[T], messages: list[BaseMessage], tier: Tier,
                   config: RunnableConfig | None = None) -> T: ...

    def stream_text(self, messages: list[BaseMessage], tier: Tier,
                    config: RunnableConfig | None = None) -> Iterator[str]: ...

    def describe(self) -> dict[str, str]: ...


class LangChainLLM:
    def __init__(self, settings: Settings):
        self.cfg: LLMSettings = settings.llm
        self._settings = settings
        self._models: dict[Tier, list[tuple[str, BaseChatModel]]] = {
            tier: self._build_chain(tier) for tier in ("fast", "strong")
        }
        self._structured_cache: dict[tuple[Tier, type[BaseModel]], Runnable] = {}

    # ---- construction ----
    def _openai(self, model: str) -> BaseChatModel:
        from langchain_openai import ChatOpenAI

        key = self._settings.openai_api_key
        return ChatOpenAI(
            model=model,
            **self._sampling(model),
            timeout=self.cfg.timeout_s,
            max_retries=self.cfg.max_retries,
            stream_usage=True,
            api_key=key,
            base_url=self._settings.openai_base_url,
        )

    def _sampling(self, model_or_deployment: str) -> dict[str, Any]:
        """Temperature kwarg, omitted for reasoning models (they reject non-default values).
        Azure deployment names are user-chosen, so this relies on them containing the model name;
        otherwise set llm.temperature to null."""
        if self.cfg.temperature is None or _REASONING_MODEL.match(model_or_deployment):
            return {}
        return {"temperature": self.cfg.temperature}

    def _azure(self, deployment: str) -> BaseChatModel:
        from langchain_openai import AzureChatOpenAI

        return AzureChatOpenAI(
            azure_endpoint=self._settings.azure_openai_endpoint,
            azure_deployment=deployment,
            api_version=self.cfg.azure_api_version,
            api_key=self._settings.azure_openai_api_key,
            stream_usage=True,
            **self._sampling(deployment),
            timeout=self.cfg.timeout_s,
            max_retries=self.cfg.max_retries,
        )

    def _check_azure_config(self) -> None:
        required = {
            "AZURE_OPENAI_ENDPOINT": self._settings.azure_openai_endpoint,
            "AZURE_OPENAI_API_KEY": self._settings.azure_openai_api_key,
            "LLM__AZURE_ROUTER_DEPLOYMENT": self.cfg.azure_router_deployment,
            "LLM__AZURE_SYNTH_DEPLOYMENT": self.cfg.azure_synth_deployment,
        }
        missing = [name for name, value in required.items() if not value]
        if missing:
            raise LLMConfigError(f"LLM__PROVIDER=azure but these settings are empty: {', '.join(missing)}")

    def _ollama(self, model: str) -> BaseChatModel:
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=model,
            temperature=self.cfg.temperature,
            base_url=self._settings.ollama_base_url,
            client_kwargs={"timeout": self.cfg.timeout_s * 3},  # local CPU inference is slow
        )

    def _build_chain(self, tier: Tier) -> list[tuple[str, BaseChatModel]]:
        fast = tier == "fast"
        openai_model = self.cfg.router_model if fast else self.cfg.synth_model
        ollama_model = self.cfg.ollama_router_model if fast else self.cfg.ollama_synth_model
        chain: list[tuple[str, BaseChatModel]] = []
        if self.cfg.provider == "azure":
            self._check_azure_config()
            deployment = self.cfg.azure_router_deployment if fast else self.cfg.azure_synth_deployment
            assert deployment  # guaranteed by _check_azure_config
            chain.append((f"azure:{deployment}", self._azure(deployment)))
        elif self.cfg.provider == "openai" and self._settings.openai_api_key:
            chain.append((f"openai:{openai_model}", self._openai(openai_model)))
        elif self.cfg.provider == "openai":
            log.warning("llm.no_openai_key_using_ollama")
        if not chain or self.cfg.fallback_to_ollama:
            chain.append((f"ollama:{ollama_model}", self._ollama(ollama_model)))
        return chain

    def describe(self) -> dict[str, str]:
        return {tier: " -> ".join(name for name, _ in chain) for tier, chain in self._models.items()}

    # ---- calls ----
    def _runnable(self, schema: type[T], tier: Tier) -> Runnable:
        key = (tier, schema)
        if key not in self._structured_cache:
            runnables = [
                model.with_structured_output(
                    schema, method="json_schema" if name.startswith("ollama") else "function_calling"
                )
                for name, model in self._models[tier]
            ]
            primary, *fallbacks = runnables
            self._structured_cache[key] = primary.with_fallbacks(fallbacks) if fallbacks else primary
        return self._structured_cache[key]

    def structured(self, schema: type[T], messages: list[BaseMessage], tier: Tier,
                   config: RunnableConfig | None = None) -> T:
        try:
            # config carries per-request callbacks (token usage accounting, tracing)
            result = self._runnable(schema, tier).invoke(messages, config=config)
        except Exception as exc:
            log.error("llm.all_providers_failed", tier=tier, schema=schema.__name__,
                      error=type(exc).__name__)
            raise LLMUnavailableError(f"LLM call failed ({type(exc).__name__})") from exc
        if result is None:  # model refused or returned nothing parseable
            raise LLMUnavailableError("LLM returned no structured output")
        return result

    def stream_text(self, messages: list[BaseMessage], tier: Tier,
                    config: RunnableConfig | None = None) -> Iterator[str]:
        """Stream plain text. Falls back to the next provider only if a provider fails before its
        first token; a failure mid-answer raises, because half an answer must not be continued by a
        different model."""
        last_error: Exception | None = None
        for name, model in self._models[tier]:
            started = False
            try:
                for chunk in model.stream(messages, config=config):
                    text = chunk.content if isinstance(chunk.content, str) else ""
                    if text:
                        started = True
                        yield text
                return
            except Exception as exc:
                if started:
                    log.error("llm.stream_interrupted", tier=tier, provider=name, error=type(exc).__name__)
                    raise LLMStreamInterruptedError(f"answer stream interrupted ({type(exc).__name__})") from exc
                log.warning("llm.provider_failed", tier=tier, provider=name, error=type(exc).__name__)
                last_error = exc
        raise LLMUnavailableError(f"LLM call failed ({type(last_error).__name__})") from last_error
