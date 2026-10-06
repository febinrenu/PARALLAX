from pathlib import Path

from pydantic import BaseModel, Field

# Model ids and URLs live here (and in .env overrides), never in call sites.
SERVICE_KEY_ENV = {
    "extract": "GROQ_KEY_EXTRACT",
    "report": "GROQ_KEY_REPORT",
    "judge": "GROQ_KEY_JUDGE",
    "audio": "GROQ_KEY_AUDIO",
}

DEFAULT_MODELS = {
    "extract": "openai/gpt-oss-20b",
    "report": "openai/gpt-oss-120b",
    "judge": "openai/gpt-oss-20b",
}

# Confirmed present in GET /models on 2026-10-06 (the rate-limit docs omit the prefix).
PROMPT_GUARD_MODEL = "meta-llama/llama-prompt-guard-2-86m"


class LLMConfig(BaseModel):
    base_url: str = "https://api.groq.com/openai/v1"
    models: dict[str, str] = Field(default_factory=lambda: dict(DEFAULT_MODELS))
    key_env: dict[str, str] = Field(default_factory=lambda: dict(SERVICE_KEY_ENV))
    cache_dir: Path = Path(".cache/llm")
    prompts_dir: Path = Path(__file__).resolve().parent / "prompts"

    # Free tier is 30 RPM / 8K TPM / 1K RPD per org and model; stay under it.
    rpm_limit: int = 25
    tpm_soft_limit: int = 6500
    rpd_limit: int = 1000
    rpd_warn_fraction: float = 0.8
    max_request_tokens: int = 3000
    max_output_tokens: int = 800
    max_audio_bytes: int = 25 * 1024 * 1024  # Groq's free-tier upload limit
    audio_model: str = "whisper-large-v3-turbo"

    max_attempts: int = 4
    deadline_s: float = 20.0
    backoff_base_s: float = 0.5
    request_timeout_s: float = 15.0

    # "json_object" + schema text in the system prompt; "json_schema" uses strict structured output.
    json_mode: str = "json_object"
    reasoning_effort: str | None = "low"
