import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "api"))
import asyncio
import time
from types import SimpleNamespace

from async_stream import close_stream, iter_stream, open_stream
from task_store import TaskStore


def test_blocking_legacy_stream_does_not_block_event_loop():
    order = []

    def create():
        time.sleep(0.05)
        order.append("created")
        return iter([1, 2])

    async def scenario():
        ticks = []

        async def tick():
            for _ in range(3):
                await asyncio.sleep(0.005)
                ticks.append(1)
                order.append("tick")

        async def consume():
            s = await open_stream(create)
            return [x async for x in iter_stream(s)]

        values, _ = await asyncio.gather(consume(), tick())
        assert values == [1, 2] and len(ticks) == 3
        assert order[0] == "tick"

    asyncio.run(scenario())


def test_native_async_stream_closes():
    class Stream:
        closed = False

        async def close(self):
            self.closed = True

        async def __aiter__(self):
            yield 3

    async def scenario():
        stream = Stream()

        async def create():
            return stream

        s = await open_stream(create)
        assert [x async for x in iter_stream(s)] == [3]
        await close_stream(s)
        assert stream.closed

    asyncio.run(scenario())


def test_tasks_and_conversations_survive_restart(tmp_path):
    path = tmp_path / "state.db"
    first = TaskStore(path)
    task_id = first.create("c1")
    first.update(task_id, "running")
    first.event(task_id, "tool_completed", {"tool": "calculate"})
    first.save_conversation(
        {
            "id": "c1",
            "messages": [
                {"role": "assistant", "content": "ok", "reasoning_content": "transient"}
            ],
        }
    )
    second = TaskStore(path)
    second.recover()
    assert second.get(task_id)["status"] == "interrupted"
    assert second.get(task_id)["events"][0]["payload"]["tool"] == "calculate"
    assert second.conversations()["c1"]["messages"] == [
        {"role": "assistant", "content": "ok"}
    ]


def test_tracked_stream_cancellation_records_status_and_closes(monkeypatch, tmp_path):
    import main

    store = TaskStore(tmp_path / "cancel.db")
    monkeypatch.setattr(main, "state_store", store)

    class DelayedStream:
        closed = False

        async def __aiter__(self):
            await asyncio.sleep(5)
            yield SimpleNamespace(choices=[])

        async def close(self):
            self.closed = True

    stream = DelayedStream()

    async def create(**kwargs):
        return stream

    monkeypatch.setattr(
        main,
        "client",
        SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create))
        ),
    )
    monkeypatch.setattr(main.mcp_client, "session", None)

    async def scenario():
        task_id = store.create("missing-conversation")
        request = main.ChatRequest(
            messages=[main.Message(role="user", content="hello")]
        )

        async def consume():
            async for _ in main.tracked_stream(request, asyncio.Event(), task_id):
                pass

        task = asyncio.create_task(consume())
        await asyncio.sleep(0.03)
        await main.cancel_task(task_id)
        try:
            await task
        except asyncio.CancelledError:
            pass
        assert store.get(task_id)["status"] == "cancelled"
        assert stream.closed

    asyncio.run(scenario())
