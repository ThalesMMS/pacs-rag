from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Protocol
from urllib import request


class EmbeddingProvider(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]:
        ...


@dataclass
class HashEmbeddingProvider:
    dim: int = 64

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            vector = [0.0] * self.dim
            tokens = [token for token in text.lower().split() if token]
            for token in tokens:
                digest = hashlib.md5(token.encode("utf-8")).digest()
                idx = int.from_bytes(digest[:4], "little") % self.dim
                vector[idx] += 1.0
            norm = math.sqrt(sum(value * value for value in vector))
            if norm:
                vector = [value / norm for value in vector]
            vectors.append(vector)
        return vectors


@dataclass
class OllamaEmbeddingProvider:
    base_url: str
    model: str

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            payload = json.dumps({"model": self.model, "prompt": text}).encode("utf-8")
            req = request.Request(
                url=f"{self.base_url.rstrip('/')}/api/embeddings",
                data=payload,
                headers={"Content-Type": "application/json"},
            )
            with request.urlopen(req, timeout=30) as response:
                body = json.loads(response.read().decode("utf-8"))
            embedding = body.get("embedding")
            if not isinstance(embedding, list):
                raise ValueError("Invalid embedding response")
            vectors.append([float(value) for value in embedding])
        return vectors


def build_embedder(
    provider: str,
    model: str | None,
    base_url: str | None,
    dim: int,
) -> EmbeddingProvider:
    provider_norm = provider.strip().lower() if provider else "hash"
    if provider_norm == "ollama":
        if not model or not base_url:
            raise ValueError("Ollama embedder requires model and base_url")
        return OllamaEmbeddingProvider(base_url=base_url, model=model)
    return HashEmbeddingProvider(dim=dim)
