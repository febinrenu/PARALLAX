"""Startup warm-up: load the heavy models once, in the background, so the first study is not the slow one.

A cold chest read takes about 17 s, the precedent embedder 10 to 45 s on a CPU, and a study that has to load them
while the machine is short of memory can hit the stage time limit and return nothing. Warming them right after the
server starts moves that cost out of the first doctor's wait. The server answers requests while it warms; a failure
to load one model is recorded and never stops the server or the other loads.

`MEDPROOF_WARMUP` chooses what to load: `1` (default, everything), `0` (nothing), or a list such as `cxr,precedents`.
The second reader (MedGemma) is a separate process that loads its model on its first request, so warming it means sending
one tiny read; without `MEDGEMMA_URL` that target is reported as not loaded.
On a laptop short of memory, load only what the demo needs: each reader and the embedder hold their weights for the
life of the process.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable, Mapping
from typing import Any

READERS = ("cxr", "bone_xray", "skin_dermoscopy", "brain_mri")
ALL_TARGETS = (*READERS, "precedents", "llm", "second_read")


def parse_targets(value: str | None) -> tuple[str, ...]:
    """The targets named by a setting. None means on; empty or `0` means off; unknown names are ignored."""
    if value is None:
        return ALL_TARGETS
    text = value.strip().lower()
    if text in ("", "0", "off", "false", "no", "none"):
        return ()
    if text in ("1", "on", "true", "yes", "all"):
        return ALL_TARGETS
    names = [part.strip() for part in text.replace(";", ",").split(",")]
    return tuple(n for n in names if n in ALL_TARGETS)


def targets_from_env() -> tuple[str, ...]:
    return parse_targets(os.environ.get("MEDPROOF_WARMUP"))


def _service_transport() -> Any:
    """Replaced in tests; None means the real network."""
    return None


def default_loaders(app_state: Any) -> dict[str, Callable[[], object]]:
    """One loader per target, each importing lazily so a missing optional package only fails its own target."""
    config = app_state.config

    def reader(modality: str) -> Callable[[], object]:
        def load() -> object:
            from medproof.pipeline import _load_reader

            return _load_reader(modality, config)

        return load

    def precedents() -> object:
        from medproof.reasoning_stages import _embedder

        return _embedder()

    def llm() -> object:
        from medproof.reasoning_stages import _default_llm_pool

        return _default_llm_pool()

    def second_read() -> object:
        import numpy as np

        from medproof.intake.decode import DecodedImage
        from medproof.readers.generalist import GeneralistReader

        client = GeneralistReader(cache_dir=config.cache_dir / "medgemma", transport=_service_transport(), timeout=240.0)
        try:
            if not client.url:
                raise LookupError("MEDGEMMA_URL is not set")
            blank = np.full((64, 64), 128, np.uint8)
            read = client.read(DecodedImage(display=blank, analysis=blank.astype(np.float32), sha256=f"warm-up-{time.time_ns()}", source_format="png"), "other")  # never a cache hit: the point is to reach the service
            if not read.ok:
                raise RuntimeError("; ".join(read.warnings) or "the service gave no read")
            return read
        finally:
            client.close()

    return {**{m: reader(m) for m in READERS}, "precedents": precedents, "llm": llm, "second_read": second_read}


def warm_up(
    targets: tuple[str, ...], *, loaders: Mapping[str, Callable[[], object]], log: Callable[[str], None] = print
) -> dict[str, dict[str, Any]]:
    """Load each target in order. Returns {target: {ok, seconds, error?}}; never raises."""
    results: dict[str, dict[str, Any]] = {}
    for target in targets:
        t0 = time.perf_counter()
        loader = loaders.get(target)
        try:
            if loader is None:
                raise LookupError("no loader for this target")
            loader()
            results[target] = {"ok": True, "seconds": round(time.perf_counter() - t0, 1)}
            log(f"warm-up: {target} ready in {results[target]['seconds']} s")
        except Exception as exc:  # noqa: BLE001 - one model failing to load must not stop the others
            results[target] = {"ok": False, "seconds": round(time.perf_counter() - t0, 1), "error": f"{type(exc).__name__}: {exc}"}
            log(f"warm-up: {target} not loaded ({results[target]['error']})")
    return results
