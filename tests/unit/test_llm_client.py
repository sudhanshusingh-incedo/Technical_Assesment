"""Provider selection and fallback wiring (constructs clients only; no network calls)."""

import pytest
from pydantic import SecretStr

from kassist.config import Settings
from kassist.llm.client import LangChainLLM, LLMConfigError


def make(settings: Settings, **llm_overrides) -> Settings:
    return settings.model_copy(update={"llm": settings.llm.model_copy(update=llm_overrides)})


@pytest.fixture
def azure(settings: Settings) -> Settings:
    s = make(settings, provider="azure", azure_router_deployment="mini-dep", azure_synth_deployment="big-dep")
    return s.model_copy(update={"azure_openai_endpoint": "https://example.openai.azure.com/",
                                "azure_openai_api_key": SecretStr("azure-key")})


def test_azure_uses_deployment_names_per_tier_with_ollama_fallback(azure):
    desc = LangChainLLM(azure).describe()
    assert desc["fast"] == "azure:mini-dep -> ollama:qwen2.5:7b-instruct"
    assert desc["strong"] == "azure:big-dep -> ollama:qwen2.5:7b-instruct"


def test_azure_without_fallback(azure):
    desc = LangChainLLM(make(azure, fallback_to_ollama=False)).describe()
    assert desc == {"fast": "azure:mini-dep", "strong": "azure:big-dep"}


def test_azure_missing_settings_fail_fast_with_names(settings):
    s = make(settings, provider="azure", azure_router_deployment="mini-dep")
    with pytest.raises(LLMConfigError) as exc:
        LangChainLLM(s)
    msg = str(exc.value)
    assert "AZURE_OPENAI_ENDPOINT" in msg and "AZURE_OPENAI_API_KEY" in msg
    assert "LLM__AZURE_SYNTH_DEPLOYMENT" in msg and "ROUTER" not in msg


def test_openai_with_key(settings):
    s = settings.model_copy(update={"openai_api_key": SecretStr("sk-test")})
    assert LangChainLLM(s).describe()["strong"] == "openai:gpt-4.1 -> ollama:qwen2.5:7b-instruct"


def test_openai_without_key_uses_ollama_only(settings):
    assert LangChainLLM(settings).describe() == {"fast": "ollama:qwen2.5:7b-instruct",
                                                 "strong": "ollama:qwen2.5:7b-instruct"}


def _primary(llm: LangChainLLM, tier: str):
    return llm._models[tier][0][1]


def test_temperature_omitted_for_reasoning_models_only(azure):
    s = make(azure, azure_router_deployment="gpt-5-mini", azure_synth_deployment="gpt-4o")
    llm = LangChainLLM(s)
    assert _primary(llm, "fast").temperature is None      # gpt-5 family: provider default
    assert _primary(llm, "strong").temperature == 0.0     # classic model: deterministic


def test_temperature_null_disables_it_everywhere(azure):
    llm = LangChainLLM(make(azure, temperature=None, azure_synth_deployment="gpt-4o"))
    assert _primary(llm, "strong").temperature is None


# ---- streaming fallback semantics (fake chat models; no network) --------------------------------
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel  # noqa: E402
from langchain_core.messages import AIMessage, HumanMessage  # noqa: E402

from kassist.llm.client import LLMStreamInterruptedError, LLMUnavailableError  # noqa: E402


class _FailingModel(GenericFakeChatModel):
    """Streams `ok_tokens` words, then raises (0 = fails before the first token)."""

    ok_tokens: int = 0

    def _stream(self, *args, **kwargs):
        for i, chunk in enumerate(super()._stream(*args, **kwargs)):
            if i >= self.ok_tokens:
                raise ConnectionError("provider down")
            yield chunk


def _llm_with(settings, *models) -> LangChainLLM:
    llm = LangChainLLM(settings)
    chain = [(f"m{i}", m) for i, m in enumerate(models)]
    llm._models = {"fast": chain, "strong": chain}
    return llm


def _fake(text: str, cls=GenericFakeChatModel, **kw):
    return cls(messages=iter([AIMessage(content=text)]), **kw)


def test_stream_falls_back_when_primary_fails_before_first_token(settings):
    llm = _llm_with(settings, _fake("primary", _FailingModel, ok_tokens=0), _fake("fallback answer"))
    assert "".join(llm.stream_text([HumanMessage("q")], tier="strong")) == "fallback answer"


def test_stream_failure_mid_answer_is_not_continued_by_another_model(settings):
    llm = _llm_with(settings, _fake("one two three", _FailingModel, ok_tokens=2), _fake("fallback answer"))
    received = []
    with pytest.raises(LLMStreamInterruptedError):
        for chunk in llm.stream_text([HumanMessage("q")], tier="strong"):
            received.append(chunk)
    assert "".join(received) == "one " and "fallback" not in "".join(received)


def test_stream_all_providers_down(settings):
    llm = _llm_with(settings, _fake("a", _FailingModel), _fake("b", _FailingModel))
    with pytest.raises(LLMUnavailableError):
        list(llm.stream_text([HumanMessage("q")], tier="strong"))
