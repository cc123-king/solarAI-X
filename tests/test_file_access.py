import asyncio
import sys
from pathlib import Path

import pytest
import httpx
from fastapi import HTTPException
from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "api"))
import main  # noqa: E402
from file_access import resolve_generated_file  # noqa: E402


@pytest.mark.parametrize(
    "path",
    [
        "plot_results/../.env",
        "plot_results/../../secret.txt",
        "plot_results\\..\\secret.txt",
        "C:/secret.txt",
        "/plot_results/a.png",
        "plot_results_bad/a.png",
        "plot_results//a.png",
        "plot_results/a:stream",
        "plot_results/\x00a.png",
    ],
)
def test_unsafe_result_paths_are_rejected(tmp_path, path):
    with pytest.raises(PermissionError):
        resolve_generated_file(path, tmp_path)


def test_allowed_file_is_resolved_and_missing_file_is_not_served(tmp_path, monkeypatch):
    directory = tmp_path / "plot_results"
    directory.mkdir()
    target = directory / "curve.png"
    target.write_bytes(b"png-test")
    monkeypatch.setattr(main, "GENERATED_FILES_ROOT", tmp_path)
    response = asyncio.run(main.get_file("plot_results/curve.png"))
    assert Path(response.path) == target
    assert response.media_type == "image/png"
    with pytest.raises(HTTPException) as error:
        asyncio.run(main.get_file("plot_results/../secret.txt"))
    assert error.value.status_code == 403
    with pytest.raises(HTTPException) as error:
        asyncio.run(main.get_file("plot_results/missing.png"))
    assert error.value.status_code == 404


def test_empty_chat_messages_are_rejected_before_streaming():
    with pytest.raises(ValidationError):
        main.ChatRequest(messages=[])


@pytest.mark.parametrize("destination", ["", "api", "elsewhere"])
def test_result_directory_redirect_cannot_expose_other_project_files(
    tmp_path, monkeypatch, destination
):
    original_resolve = Path.resolve
    redirected = tmp_path / destination

    def resolve(path, *args, **kwargs):
        if path == tmp_path / "plot_results":
            return redirected
        return original_resolve(path, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", resolve)
    with pytest.raises(PermissionError):
        resolve_generated_file("plot_results/secret.txt", tmp_path)


def test_download_route_returns_bytes_and_rejects_non_result_files(
    tmp_path, monkeypatch
):
    directory = tmp_path / "plot_results"
    directory.mkdir()
    (directory / "curve.png").write_bytes(b"png-fixture")
    monkeypatch.setattr(main, "GENERATED_FILES_ROOT", tmp_path)

    async def scenario():
        # ASGI transport does not start the real MCP lifespan/network connection.
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=main.app), base_url="http://test"
        ) as client:
            response = await client.get("/api/files/plot_results/curve.png")
            assert response.status_code == 200
            assert response.content == b"png-fixture"
            assert (await client.get("/api/files/api/secret.txt")).status_code == 403
            assert (
                await client.get("/api/files/plot_results/missing.png")
            ).status_code == 404

    asyncio.run(scenario())


@pytest.mark.parametrize("configured", [None, "outputs", "../outside", "C:/elsewhere"])
def test_generated_directory_stays_downloadable(tmp_path, configured):
    from file_access import generated_output_directory
    selected = generated_output_directory(tmp_path, "plot_results", configured)
    assert selected == "plot_results"
    target = tmp_path / selected / "chart.png"
    target.write_bytes(b"test")
    assert resolve_generated_file(selected + "/chart.png", tmp_path) == target


def test_generated_directory_preserves_an_allowed_setting(tmp_path):
    from file_access import generated_output_directory
    assert generated_output_directory(tmp_path, "plot_results", "simulation_results") == "simulation_results"
