"""Nonblocking model streams, including compatibility with offline test doubles."""

import asyncio
import inspect

_END = object()


def _next(iterator):
    return next(iterator, _END)


async def open_stream(create, **kwargs):
    if inspect.iscoroutinefunction(create):
        return await create(**kwargs)
    result = await asyncio.to_thread(create, **kwargs)
    return await result if inspect.isawaitable(result) else result


async def iter_stream(stream):
    if hasattr(stream, "__aiter__"):
        async for chunk in stream:
            yield chunk
    else:
        iterator = iter(stream)
        while True:
            chunk = await asyncio.to_thread(_next, iterator)
            if chunk is _END:
                return
            yield chunk


async def close_stream(stream):
    close = getattr(stream, "close", None)
    if not callable(close):
        return
    if inspect.iscoroutinefunction(close):
        await close()
    else:
        result = await asyncio.to_thread(close)
        if inspect.isawaitable(result):
            await result
