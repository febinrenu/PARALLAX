class LLMError(Exception):
    """Base class for failures raised by the LLM pool."""


class LLMUnavailable(LLMError):
    """The service cannot answer (no key, rate limited past the deadline, outage)."""


class SchemaError(LLMUnavailable):
    """The model's reply never validated against the requested schema."""


class BudgetExceeded(LLMError):
    """The request is larger than the per-request token cap."""
