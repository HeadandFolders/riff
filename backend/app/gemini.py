"""Thin Vertex AI wrapper.

Everything that constitutes a judgement in riff goes through here, so the
reasoning model is swappable in one place and every call is structured.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Sequence, TypeVar

from google import genai
from google.genai import types
from pydantic import BaseModel

from .config import settings

T = TypeVar("T", bound=BaseModel)


@lru_cache
def client() -> genai.Client:
    cfg = settings()
    return genai.Client(
        vertexai=True, project=cfg.project_id, location=cfg.vertex_location
    )


def structured(
    contents: Any,
    schema: type[T],
    *,
    system_instruction: str | None = None,
    temperature: float = 0.2,
) -> T:
    """Generate a validated instance of ``schema``.

    Raises if the model returns something unparseable, rather than silently
    handing back a half-populated object.
    """
    cfg = settings()
    response = client().models.generate_content(
        model=cfg.reasoning_model,
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=temperature,
            response_mime_type="application/json",
            response_schema=schema,
        ),
    )
    parsed = response.parsed
    if not isinstance(parsed, schema):
        raise ValueError(f"{schema.__name__} not returned by {cfg.reasoning_model}")
    return parsed


def audio_part(data: bytes, mime_type: str = "audio/webm") -> types.Part:
    """Spoken answers go straight to Gemini; there is no transcription step."""
    return types.Part.from_bytes(data=data, mime_type=mime_type)


def pdf_part(data: bytes) -> types.Part:
    return types.Part.from_bytes(data=data, mime_type="application/pdf")


def embed(texts: Sequence[str]) -> list[list[float]]:
    cfg = settings()
    response = client().models.embed_content(
        model=cfg.embedding_model,
        contents=list(texts),
        config=types.EmbedContentConfig(
            output_dimensionality=cfg.embedding_dimensions
        ),
    )
    return [list(item.values or []) for item in (response.embeddings or [])]
