"""Ollama embedding client for the vector-memory server.

Exposes :func:`embed` (single text) and :func:`embed_many` (batch). Both try
the modern ``client.embed()`` API first and fall back to the legacy
``client.embeddings()`` API for older Ollama servers.
"""

from __future__ import annotations

import os
from typing import Any

import ollama

_DEFAULTS: dict[str, str] = {
    "ollama_url": "http://192.168.1.103:11434",
    "embed_model": "qwen3-embedding:8b",
}

OLLAMA_URL = os.environ.get("OLLAMA_URL", _DEFAULTS["ollama_url"])
EMBED_MODEL = os.environ.get("EMBED_MODEL", _DEFAULTS["embed_model"])

# Single shared Ollama client; tests replace this attribute.
ollama_client: Any = ollama.Client(host=OLLAMA_URL)


def embed(text: str) -> list[float]:
    """Return the embedding vector for *text*.

    Tries the modern ``client.embed()`` API first and falls back to the
    legacy ``client.embeddings()`` for older ollama servers.
    """
    try:
        res = ollama_client.embed(model=EMBED_MODEL, input=text)
        return res["embeddings"][0]
    except (AttributeError, KeyError, TypeError):
        res = ollama_client.embeddings(model=EMBED_MODEL, prompt=text)
        return res["embedding"]


def embed_many(texts: list[str]) -> list[list[float]]:
    """Embed a list of texts in a single Ollama call.

    ``client.embed()`` natively accepts ``input: Sequence[str]``.
    """
    try:
        res = ollama_client.embed(model=EMBED_MODEL, input=texts)
        return res["embeddings"]
    except (AttributeError, KeyError, TypeError):
        # Legacy fallback: one call per text.
        return [ollama_client.embeddings(model=EMBED_MODEL, prompt=t)["embedding"] for t in texts]


# Back-compat alias used by earlier revisions and the live smoke test.
embed_batch = embed_many
