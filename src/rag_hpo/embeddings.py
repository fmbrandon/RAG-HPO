from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import numpy as np

DEFAULT_SAPBERT_MODEL = "pritamdeka/SapBERT-mnli-snli-scinli-scitail-mednli-stsb"
DEFAULT_SAPBERT_REVISION = "ad34b857369884acb59a6e67f69008f533c8ca3e"  # pragma: allowlist secret
DEFAULT_BGE_MODEL = "BAAI/bge-small-en-v1.5"


class EmbeddingBackend(Protocol):
    name: str
    model_id: str
    revision: str

    def encode(self, texts: Sequence[str]) -> np.ndarray: ...


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    output = np.asarray(matrix, dtype=np.float32).copy()
    norms = np.linalg.norm(output, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("embedding backend returned a zero-length vector")
    output /= norms
    return output


class SapBERTBackend:
    name = "sapbert"

    def __init__(
        self,
        model_id: str = DEFAULT_SAPBERT_MODEL,
        revision: str = DEFAULT_SAPBERT_REVISION,
        *,
        local_files_only: bool = False,
    ) -> None:
        from sentence_transformers import SentenceTransformer

        self.model_id = model_id
        self.revision = revision
        model_options: dict[str, object] = {"revision": revision}
        if local_files_only:
            model_options["local_files_only"] = True
        self._model = SentenceTransformer(model_id, **model_options)

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        values = self._model.encode(
            list(texts),
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return normalize_rows(np.asarray(values))


class FastEmbedBackend:
    name = "fastembed"
    revision = "fastembed-managed"

    def __init__(self, model_id: str = DEFAULT_BGE_MODEL) -> None:
        try:
            from fastembed import TextEmbedding
        except ImportError as exc:
            raise RuntimeError("install rag-hpo[fastembed] to use this backend") from exc
        self.model_id = model_id
        self._model = TextEmbedding(model_name=model_id)

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        return normalize_rows(np.asarray(list(self._model.embed(list(texts)))))


def create_backend(name: str, *, offline: bool = False) -> EmbeddingBackend:
    if name == "sapbert":
        return SapBERTBackend(local_files_only=offline)
    if name == "fastembed":
        return FastEmbedBackend()
    raise ValueError(f"unknown embedding backend: {name}")
