"""Groq chat client shared by extraction, report and judge stages.

Per-service keys, an on-disk response cache, retry/backoff that honours
`retry-after`, a token budgeter that keeps requests under the free-tier ceiling,
schema-validated JSON with one repair turn, and an explicit offline fallback.
"""

import hashlib
import json
import logging
import random
import re
import time
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Generic, TypeVar

import httpx
from diskcache import Cache  # type: ignore[import-untyped]
from pydantic import BaseModel, ValidationError

from medproof.llm.config import LLMConfig
from medproof.llm.errors import BudgetExceeded, LLMError, LLMUnavailable, SchemaError

log = logging.getLogger("medproof.llm")

T = TypeVar("T", bound=BaseModel)

_HEADER = re.compile(r"\A---\n.*?\n---\n", re.DOTALL)
_FENCE = re.compile(r"\A\s*```(?:json)?\s*(.*?)\s*```\s*\Z", re.DOTALL)
_TAG = re.compile(r"</?\s*note_data\s*>", re.IGNORECASE)
_DROPPABLE_PARAMS = ("reasoning_effort", "response_format")


@dataclass
class PoolResult(Generic[T]):
    ok: bool
    data: T | None
    source: str  # "live" | "cache" | "fallback"
    model: str
    attempts: int = 0
    tokens: int = 0
    warning: str | None = None


def quote_data(text: str) -> str:
    """Wrap untrusted text as inert data; it cannot close or reopen the fence."""
    safe = _TAG.sub(lambda m: m.group(0).replace("<", "&lt;").replace(">", "&gt;"), text)
    return f"<note_data>\n{safe}\n</note_data>"


def load_prompt(config: LLMConfig, name: str) -> str:
    raw = (config.prompts_dir / f"{name}.md").read_text(encoding="utf-8")
    return _HEADER.sub("", raw, count=1).strip()


def estimate_tokens(text: str) -> int:
    return int(len(text) / 3.5) + 1


class _Window:
    """Sliding 60 s window of (timestamp, amount) entries."""

    def __init__(self) -> None:
        self.entries: deque[list[float]] = deque()

    def purge(self, now: float) -> None:
        while self.entries and now - self.entries[0][0] >= 60.0:
            self.entries.popleft()

    def total(self) -> float:
        return sum(e[1] for e in self.entries)


class GroqPool:
    def __init__(
        self,
        config: LLMConfig | None = None,
        *,
        transport: httpx.BaseTransport | None = None,
        env: Mapping[str, str] | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        rng: random.Random | None = None,
    ) -> None:
        import os

        self.config = config or LLMConfig()
        self._env = os.environ if env is None else env
        self._clock = clock
        self._sleep = sleep
        self._rng = rng or random.Random(0)
        self._client = httpx.Client(
            base_url=self.config.base_url,
            transport=transport,
            timeout=self.config.request_timeout_s,
        )
        self.config.cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache = Cache(str(self.config.cache_dir))
        self._tokens: dict[tuple[str, str], _Window] = {}
        self._requests: dict[tuple[str, str], _Window] = {}

    def close(self) -> None:
        self._client.close()
        self._cache.close()

    # ------------------------------------------------------------------ public

    def chat(
        self,
        service: str,
        prompt_name: str,
        user: str,
        schema: type[T],
        *,
        model: str | None = None,
        fallback: Callable[[], T] | None = None,
    ) -> PoolResult[T]:
        model = model or self.config.models.get(service)
        if not model:
            raise LLMError(f"no model configured for service '{service}'")
        try:
            return self._chat(service, prompt_name, user, schema, model)
        except LLMUnavailable as exc:
            if fallback is None:
                raise
            log.warning("llm fallback used: service=%s reason=%s", service, exc)
            return PoolResult(
                ok=False, data=fallback(), source="fallback", model=model, warning=str(exc)
            )

    # ---------------------------------------------------------------- internals

    def _chat(
        self, service: str, prompt_name: str, user: str, schema: type[T], model: str
    ) -> PoolResult[T]:
        system = (
            f"{load_prompt(self.config, prompt_name)}\n\n"
            "Reply with a single JSON object matching this JSON Schema:\n"
            f"{json.dumps(schema.model_json_schema(), sort_keys=True)}"
        )
        messages: list[dict[str, str]] = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        prompt_tokens = estimate_tokens(system) + estimate_tokens(user)
        if prompt_tokens > self.config.max_request_tokens:
            raise BudgetExceeded(
                f"request ~{prompt_tokens} tokens exceeds cap {self.config.max_request_tokens}"
            )

        cache_key = hashlib.sha256(
            json.dumps([model, messages, schema.__name__], sort_keys=True).encode()
        ).hexdigest()
        hit = self._cache.get(cache_key)
        if hit is not None:
            return PoolResult(
                ok=True, data=schema.model_validate(hit), source="cache", model=model
            )

        key = self._env.get(self.config.key_env.get(service, ""), "")
        if not key:
            raise LLMUnavailable(f"no API key set for service '{service}'")
        label = f"{service}:{hashlib.sha256(key.encode()).hexdigest()[:6]}"

        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": 0,
            "max_tokens": self.config.max_output_tokens,
            "response_format": {"type": "json_object"},
        }
        if self.config.reasoning_effort:
            payload["reasoning_effort"] = self.config.reasoning_effort

        start = self._clock()
        attempts = 0
        repaired = False
        total_tokens = 0
        last_error = "no attempt made"

        while attempts < self.config.max_attempts:
            self._check_daily(label, model)
            reserve = estimate_tokens(json.dumps(payload["messages"])) + self.config.max_output_tokens
            slot = self._reserve(label, model, reserve, start)
            attempts += 1
            try:
                resp = self._client.post(
                    "/chat/completions",
                    json=payload,
                    headers={"Authorization": f"Bearer {key}"},
                )
            except httpx.TransportError as exc:
                slot[1] = 0
                last_error = f"transport error: {type(exc).__name__}"
                log.warning("groq transport error: key=%s attempt=%d", label, attempts)
                self._backoff(attempts, None, start)
                continue
            self._count_daily(label, model)

            if resp.status_code == 200:
                body = resp.json()
                used = int(body.get("usage", {}).get("total_tokens") or reserve)
                slot[1] = used
                total_tokens += used
                content = body["choices"][0]["message"].get("content") or ""
                try:
                    data = self._parse(content, schema)
                except (ValueError, ValidationError) as exc:
                    last_error = f"reply failed validation: {exc}"
                    if repaired:
                        raise SchemaError(last_error) from exc
                    repaired = True
                    payload["messages"] = [
                        *messages,
                        {"role": "assistant", "content": content},
                        {
                            "role": "user",
                            "content": "Your reply failed schema validation: "
                            f"{str(exc)[:300]}. Reply again with only the corrected JSON object.",
                        },
                    ]
                    continue
                self._cache.set(cache_key, data.model_dump(mode="json"))
                return PoolResult(
                    ok=True,
                    data=data,
                    source="live",
                    model=model,
                    attempts=attempts,
                    tokens=total_tokens,
                )

            slot[1] = 0
            if resp.status_code == 429 or resp.status_code >= 500:
                last_error = f"http {resp.status_code}"
                log.warning("groq http %d: key=%s attempt=%d", resp.status_code, label, attempts)
                self._backoff(attempts, resp.headers.get("retry-after"), start)
                continue
            if resp.status_code == 400 and self._drop_param(resp, payload):
                last_error = "dropped unsupported parameter"
                continue
            raise LLMUnavailable(f"http {resp.status_code} from Groq")

        raise LLMUnavailable(f"gave up after {attempts} attempts: {last_error}")

    @staticmethod
    def _parse(content: str, schema: type[T]) -> T:
        text = content.strip()
        fenced = _FENCE.match(text)
        if fenced:
            text = fenced.group(1)
        return schema.model_validate(json.loads(text))

    @staticmethod
    def _drop_param(resp: httpx.Response, payload: dict[str, Any]) -> bool:
        try:
            message = str(resp.json().get("error", {}).get("message", "")).lower()
        except ValueError:
            return False
        for name in _DROPPABLE_PARAMS:
            if name in message and name in payload:
                del payload[name]
                return True
        return False

    def _remaining(self, start: float) -> float:
        return self.config.deadline_s - (self._clock() - start)

    def _wait(self, seconds: float, start: float, why: str) -> None:
        if seconds > self._remaining(start):
            raise LLMUnavailable(f"{why}: would wait {seconds:.1f}s, past the deadline")
        if seconds > 0:
            self._sleep(seconds)

    def _backoff(self, attempt: int, retry_after: str | None, start: float) -> None:
        if retry_after:
            try:
                wait = float(retry_after)
            except ValueError:
                wait = self.config.backoff_base_s
        else:
            base = self.config.backoff_base_s * (2 ** (attempt - 1))
            wait = base + self._rng.uniform(0, base * 0.25)
        self._wait(wait, start, "rate limited")

    def _reserve(self, label: str, model: str, tokens: int, start: float) -> list[float]:
        """Block until the request fits under both the RPM and soft TPM limits."""
        slot_key = (label, model)
        tw = self._tokens.setdefault(slot_key, _Window())
        rw = self._requests.setdefault(slot_key, _Window())
        while True:
            now = self._clock()
            tw.purge(now)
            rw.purge(now)
            wait = 0.0
            if len(rw.entries) >= self.config.rpm_limit:
                wait = max(wait, rw.entries[0][0] + 60.0 - now)
            if tw.entries and tw.total() + tokens > self.config.tpm_soft_limit:
                wait = max(wait, tw.entries[0][0] + 60.0 - now)
            if wait <= 0:
                break
            self._wait(wait, start, "budget")
        rw.entries.append([now, 1])
        entry = [now, float(tokens)]
        tw.entries.append(entry)
        return entry

    def _daily_key(self, label: str, model: str) -> str:
        day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        return f"rpd:{label}:{model}:{day}"

    def _check_daily(self, label: str, model: str) -> None:
        used = int(self._cache.get(self._daily_key(label, model), 0))
        if used >= self.config.rpd_limit:
            raise LLMUnavailable(f"daily request cap reached for {model}")

    def _count_daily(self, label: str, model: str) -> None:
        key = self._daily_key(label, model)
        used = int(self._cache.get(key, 0)) + 1
        self._cache.set(key, used, expire=86400)
        if used == int(self.config.rpd_limit * self.config.rpd_warn_fraction):
            log.warning("groq daily usage at %d/%d: key=%s", used, self.config.rpd_limit, label)
