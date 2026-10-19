"""Restrict downloads to generated results, including resolved-path checks."""

from pathlib import Path

ALLOWED_RESULT_DIRECTORIES = frozenset({"simulation_results", "plot_results"})


def resolve_generated_file(file_path: str, root: Path) -> Path:
    normalized = file_path.replace("\\", "/")
    parts = normalized.split("/")
    if (
        not normalized
        or "\x00" in normalized
        or ":" in normalized
        or parts[0] not in ALLOWED_RESULT_DIRECTORIES
        or any(part in {"", ".", ".."} for part in parts)
    ):
        raise PermissionError("只能访问生成结果目录中的文件")
    resolved_root = root.resolve()
    expected_root = resolved_root / parts[0]
    allowed_root = expected_root.resolve()
    candidate = (resolved_root / normalized).resolve()
    # A result directory must not itself be a junction to api/ or the root.
    if allowed_root != expected_root or not candidate.is_relative_to(allowed_root):
        raise PermissionError("文件路径超出允许目录")
    if not candidate.is_file():
        raise FileNotFoundError("结果文件不存在")
    return candidate


def generated_output_directory(
    root: Path, default: str, configured: str | None = None
) -> str:
    """Keep new artifacts in the same directories accepted by the download route."""
    if default not in ALLOWED_RESULT_DIRECTORIES:
        raise ValueError("未知结果目录")
    selected = configured if configured in ALLOWED_RESULT_DIRECTORIES else default
    expected = root.resolve() / selected
    if expected.resolve() != expected:
        raise PermissionError("结果目录不能重定向到其他路径")
    expected.mkdir(parents=True, exist_ok=True)
    return selected
