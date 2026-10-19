"""Small, dependency-light capability checks used by the health endpoint."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _resolve_project_path(value: str | None, default: str) -> Path:
    path = Path(value or default)
    return path if path.is_absolute() else PROJECT_ROOT / path


def get_rag_status() -> dict[str, object]:
    """Report whether the existing knowledge base has a usable retrieval path."""
    directory = _resolve_project_path(os.getenv("EMBEDDING_DIR"), "embedding")
    embeddings_path = directory / "embeddings.pkl"
    file_info_path = directory / "file_info.pkl"
    data_available = all(
        path.is_file() and path.stat().st_size > 0
        for path in (embeddings_path, file_info_path)
    )
    semantic_dependency = importlib.util.find_spec("FlagEmbedding") is not None
    requested_backend = os.getenv("RAG_BACKEND", "auto").strip().lower()

    if requested_backend not in {"auto", "semantic", "lexical"}:
        return {
            "available": False,
            "backend": None,
            "data_available": data_available,
            "semantic_dependency": semantic_dependency,
            "path": str(directory),
            "error": "RAG_BACKEND 必须是 auto、semantic 或 lexical。",
        }

    backend = (
        "lexical"
        if requested_backend == "lexical" or not semantic_dependency
        else "semantic"
    )
    dependency_error = requested_backend == "semantic" and not semantic_dependency
    error = None
    if not data_available:
        error = "知识库文件 embeddings.pkl 或 file_info.pkl 缺失。"
    elif dependency_error:
        error = "RAG_BACKEND=semantic，但未安装 FlagEmbedding。"

    return {
        "available": data_available and not dependency_error,
        "backend": backend,
        "data_available": data_available,
        "semantic_dependency": semantic_dependency,
        "degraded": backend == "lexical" and data_available,
        "path": str(directory),
        "error": error,
    }
