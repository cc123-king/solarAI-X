"""无需启动大模型和服务器的项目契约测试。"""

from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _decorator_attribute(decorator: ast.expr) -> tuple[str, str] | None:
    expression = decorator.func if isinstance(decorator, ast.Call) else decorator
    if not isinstance(expression, ast.Attribute):
        return None
    if not isinstance(expression.value, ast.Name):
        return None
    return expression.value.id, expression.attr


def test_all_python_files_have_valid_syntax() -> None:
    python_files = [
        *ROOT.joinpath("api").rglob("*.py"),
        *ROOT.joinpath("web").rglob("*.py"),
        *ROOT.joinpath("scripts").rglob("*.py"),
    ]

    assert python_files
    for path in python_files:
        ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))


def test_expected_mcp_tools_are_registered() -> None:
    source = (ROOT / "api" / "mcpserver.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    tool_names = {
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and any(
            _decorator_attribute(decorator) == ("mcp", "tool")
            for decorator in node.decorator_list
        )
    }

    assert tool_names == {
        "draw_curve",
        "draw_table",
        "simulate_solar_cell",
        "batch_simulate_solar_cell",
        "search_embedded_text",
        "process_directory_for_embedding",
        "predict_perovskite_aging",
        "predict_perovskite_parameters",
        "predict_perovskite_bandgap",
    }


def test_expected_fastapi_routes_exist() -> None:
    source = (ROOT / "api" / "main.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    routes = set()

    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for decorator in node.decorator_list:
            route = _decorator_attribute(decorator)
            if route is None or route[0] != "app":
                continue
            if not isinstance(decorator, ast.Call) or not decorator.args:
                continue
            path_argument = decorator.args[0]
            if isinstance(path_argument, ast.Constant):
                routes.add((route[1].upper(), path_argument.value))

    assert ("GET", "/api/health") in routes
    assert ("POST", "/api/chat/send") in routes
    assert ("POST", "/api/solar/predict") in routes
    assert ("POST", "/api/aging/predict") in routes
    assert ("POST", "/api/perovskite/predict") in routes
    assert ("POST", "/api/perovskite/predict-bandgap") in routes


def test_configuration_template_is_complete() -> None:
    content = (ROOT / ".env.example").read_text(encoding="utf-8")
    for variable in (
        "DEEPSEEK_API_KEY",
        "DEEPSEEK_BASE_URL",
        "MCP_SERVER_URL",
        "MODEL_DIR",
        "PSKNA_DIR",
        "PSKWD_DIR",
        "EMBEDDING_DIR",
        "SOLAR_API_HOST",
        "SOLAR_API_PORT",
        "SOLAR_API_BASE_URL",
    ):
        assert f"{variable}=" in content


def test_required_local_assets_exist() -> None:
    required_assets = (
        ROOT / "api" / "quxian_high500.pth",
        ROOT / "api" / "quxian_shiyan.xlsx",
        ROOT / "embedding" / "embeddings.pkl",
        ROOT / "embedding" / "file_info.pkl",
    )

    assert all(path.is_file() and path.stat().st_size > 0 for path in required_assets)


def test_readme_matches_the_actual_frontend() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "Gradio" in readme
    assert "React + TypeScript + Ant Design" not in readme
