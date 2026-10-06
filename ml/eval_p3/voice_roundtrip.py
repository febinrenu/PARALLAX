"""Round-trip demo for voice dictation: audio file -> Whisper on Groq -> injection guard -> facts.

    python -m ml.eval_p3.voice_roundtrip path/to/dictation.wav

Live Groq (GROQ_KEY_AUDIO for Whisper, GROQ_KEY_EXTRACT / GROQ_KEY_JUDGE for the text stages).
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT))

from medproof.context.voice import process_dictation  # noqa: E402
from medproof.llm.config import LLMConfig  # noqa: E402
from medproof.llm.groq_pool import GroqPool  # noqa: E402

from ml.eval_p3.eval_context import load_env  # noqa: E402


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        print(__doc__)
        return 2
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    load_env()
    path = Path(argv[0])
    pool = GroqPool(LLMConfig(cache_dir=ROOT / ".cache" / "llm", deadline_s=120.0, max_attempts=5))
    voice, guard, ext = process_dictation(path.read_bytes(), path.name, pool, pool, note_id="voice_1")
    print(f"transcription ok={voice.ok} language={voice.language} duration={voice.duration_s}s source={voice.source}")
    print("transcript:", voice.text)
    print("guard flagged:", guard.flagged, [(f.source, f.rule, guard.text_of(f)) for f in guard.flags])
    print("facts:")
    for f in ext.facts:
        print(f"  {f.fact_type:12s} {f.polarity:10s} {f.quote!r} @ {f.span}")
    print("dropped:", ext.dropped, "warnings:", voice.warnings + ext.warnings)
    pool.close()
    return 0 if voice.ok else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
