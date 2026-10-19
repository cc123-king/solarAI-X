import copy
import importlib.util
import json
from pathlib import Path

import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "agent_eval", ROOT / "scripts/evaluate_agent_live.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
CASES = json.loads((ROOT / "evaluation/agent_tasks.json").read_text(encoding="utf-8"))


def record(case):
    return {
        "status": "completed",
        "result": json.dumps({"results": case["expected_results"]}),
        "events": [
            {"kind": "tool_completed", "payload": copy.deepcopy(call)}
            for call in case["expected_calls"]
        ],
    }


def test_single_requires_actual_outputs_and_parameter_association():
    case = CASES[0]
    valid = record(case)
    assert module.score_task(case, valid)["passed"]
    for section, key, value in [
        ("arguments", "Br_percentage", 1),
        ("result", "bandgap", 2.3),
        ("result", "unit", "meV"),
    ]:
        wrong = copy.deepcopy(valid)
        wrong["events"][0]["payload"][section][key] = value
        assert not module.score_task(case, wrong)["passed"]
    missing = copy.deepcopy(valid)
    del missing["events"][0]["payload"]["result"]
    assert not module.score_task(case, missing)["passed"]


@pytest.mark.parametrize(
    "answer",
    [
        "不是1.64 eV，实际是2.3 eV",
        "1.64 eV",
        '{"results":[{"Br_percentage":0.5,"bandgap":1.64,"unit":"meV"}]}',
        '{"results":[{"Br_percentage":1,"bandgap":1.64,"unit":"eV"}]}',
    ],
)
def test_answer_cannot_pass_by_containing_the_expected_number(answer):
    item = record(CASES[0])
    item["result"] = answer
    assert not module.score_task(CASES[0], item)["passed"]


def curve_record(tmp_path):
    case = CASES[1]
    item = record(case)
    Image.new("RGB", (20, 20), "white").save(tmp_path / "curve.png")
    item["result"] = json.dumps(
        {"results": case["expected_results"], "chart_file": "curve.png"}
    )
    item["events"].append(
        {
            "kind": "tool_completed",
            "payload": {
                "tool": "draw_curve",
                "arguments": copy.deepcopy(case["expected_curve"]),
                "result": {"file_path": "curve.png"},
            },
        }
    )
    return item


def test_curve_checks_every_point_and_real_png(tmp_path):
    case = CASES[1]
    item = curve_record(tmp_path)
    assert module.score_task(case, item, tmp_path)["passed"]
    swapped = copy.deepcopy(item)
    swapped["events"][-1]["payload"]["arguments"]["y_data"].reverse()
    assert not module.score_task(case, swapped, tmp_path)["passed"]
    (tmp_path / "curve.png").unlink()
    assert not module.score_task(case, item, tmp_path)["passed"]
    (tmp_path / "curve.png").write_bytes(b"\x89PNG\r\n\x1a\ninvalid")
    assert not module.score_task(case, item, tmp_path)["passed"]


def test_curve_rejects_missing_results_or_artifacts_outside_root(tmp_path):
    item = curve_record(tmp_path)
    wrong = copy.deepcopy(item)
    wrong["events"].pop(0)
    assert not module.score_task(CASES[1], wrong, tmp_path)["passed"]
    item["events"][-1]["payload"]["result"]["file_path"] = "../elsewhere.png"
    assert not module.score_task(CASES[1], item, tmp_path)["passed"]


def test_failed_task_is_never_successful():
    item = record(CASES[0])
    item["status"] = "failed"
    assert not module.score_task(CASES[0], item)["passed"]


@pytest.mark.parametrize("removed", [4, 8, 12, 16, 20, 24])
def test_truncated_png_never_passes_or_crashes(tmp_path, removed):
    item = curve_record(tmp_path)
    path = tmp_path / "curve.png"
    data = path.read_bytes()
    path.write_bytes(data[:-removed])
    scored = module.score_task(CASES[1], item, tmp_path)
    assert not scored["passed"] and not scored["chart_artifact_verified"]
    assert scored["chart_artifact_error"]


def test_png_crc_and_pixel_stream_are_both_checked(tmp_path):
    import struct
    import zlib

    item = curve_record(tmp_path)
    path = tmp_path / "curve.png"
    original = path.read_bytes()
    start = original.index(b"IDAT")
    size = struct.unpack(">I", original[start - 4 : start])[0]
    broken = bytearray(original)
    broken[start + 4] ^= 0xFF
    path.write_bytes(broken)
    scored = module.score_task(CASES[1], item, tmp_path)
    assert not scored["passed"] and scored["chart_artifact_error"] == "SyntaxError"
    # Recalculate CRC so chunk verification succeeds, but compressed pixels remain invalid.
    crc = zlib.crc32(broken[start : start + 4 + size]) & 0xFFFFFFFF
    broken[start + 4 + size : start + 8 + size] = struct.pack(">I", crc)
    path.write_bytes(broken)
    with Image.open(path) as image:
        image.verify()
    scored = module.score_task(CASES[1], item, tmp_path)
    assert not scored["passed"] and scored["chart_artifact_error"] == "OSError"


def test_one_task_exception_is_redacted_and_next_task_still_runs(tmp_path):
    import asyncio
    from types import SimpleNamespace

    class Store:
        def create(self, _):
            return "synthetic-task"

        def get(self, _):
            return {**record(CASES[0]), "elapsed_ms": 1}

    attempts = []

    async def stream(*_):
        attempts.append(1)
        if len(attempts) == 1:
            raise SyntaxError("sensitive provider detail")
        yield 'data: {"type":"content"}'

    fake = SimpleNamespace(
        state_store=Store(),
        ChatRequest=lambda **kw: kw,
        Message=lambda **kw: kw,
        tracked_stream=stream,
        GENERATED_FILES_ROOT=tmp_path,
    )

    async def run():
        return [
            await module.evaluate_case(fake, CASES[0]),
            await module.evaluate_case(fake, CASES[0]),
        ]

    first, second = asyncio.run(run())
    assert first["error_type"] == "SyntaxError" and not first["passed"]
    assert "sensitive" not in json.dumps(first)
    assert second["passed"] and len(attempts) == 2
