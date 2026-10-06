"""Voice dictation (D19): audio in, a note out, then the same guard and extraction as a typed note.

A spoken instruction is as untrusted as a typed one, so the transcript goes through the injection
guard before extraction. Every failure degrades to "no voice input" with a warning; nothing raises.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Protocol

from medproof.context.extract import ExtractionResult, extract
from medproof.context.injection_guard import GuardResult, InjectionGuard
from medproof.llm.config import LLMConfig
from medproof.llm.errors import LLMError
from medproof.llm.groq_pool import TranscribeResult

ALLOWED_EXTENSIONS = frozenset({"wav", "mp3", "m4a", "mp4", "mpeg", "mpga", "ogg", "flac", "webm"})
# Biases Whisper towards clinical vocabulary; static text, so it can be cached and carries no patient data.
CLINICAL_PROMPT = (
    "Clinical dictation by a doctor. Terms: dyspnea, pleural effusion, consolidation, pneumothorax, "
    "atelectasis, cardiomegaly, nodule, fracture, hemoptysis, orthopnea, c/o, h/o, s/p, "
    "patient's right, patient's left."
)


@dataclass
class VoiceNote:
    ok: bool
    text: str = ""
    language: str = ""
    duration_s: float | None = None
    source: str = ""
    warnings: list[str] = field(default_factory=list)


class _AudioPool(Protocol):
    def transcribe(
        self, service: str, model: str, filename: str, data: bytes, *, language: str | None = ..., prompt: str | None = ...
    ) -> TranscribeResult: ...


def _extension(filename: str) -> str:
    base = filename.replace("\\", "/").rsplit("/", 1)[-1]
    return base.rsplit(".", 1)[1].lower() if "." in base else ""


def transcribe_dictation(audio: bytes, filename: str, pool: _AudioPool | None) -> VoiceNote:
    if pool is None:
        return VoiceNote(ok=False, warnings=["voice input not configured: no audio service"])
    if _extension(filename) not in ALLOWED_EXTENSIONS or ".." in filename:
        return VoiceNote(ok=False, warnings=[f"unsupported audio file type: {filename!r}"])
    if not audio:
        return VoiceNote(ok=False, warnings=["the audio file is empty"])
    try:
        result = pool.transcribe("audio", LLMConfig().audio_model, filename, audio, prompt=CLINICAL_PROMPT)
    except LLMError as exc:
        return VoiceNote(ok=False, warnings=[f"voice transcription unavailable: {exc}"])
    text = re.sub(r"\s+", " ", result.text).strip()
    if not text:
        return VoiceNote(ok=False, language=result.language, duration_s=result.duration_s, warnings=["no speech detected in the audio"])
    return VoiceNote(ok=True, text=text, language=result.language, duration_s=result.duration_s, source=result.source)


def process_dictation(
    audio: bytes, filename: str, audio_pool: _AudioPool | None, text_pool: Any, *, note_id: str
) -> tuple[VoiceNote, GuardResult, ExtractionResult]:
    """Transcribe, guard, extract: the voice path into the context engine."""
    voice = transcribe_dictation(audio, filename, audio_pool)
    if not voice.ok:
        return voice, GuardResult(text=""), ExtractionResult(ok=True, warnings=list(voice.warnings))
    guard = InjectionGuard(text_pool).check(voice.text)
    return voice, guard, extract(voice.text, note_id, text_pool, guard)
