"""Real OpenTelemetry SDK spans with an in-memory exporter; no network export."""

import sys
from pathlib import Path
from types import SimpleNamespace

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'api'))
from runtime_telemetry import RuntimeTelemetry


def test_task_model_tool_have_one_trace_and_no_user_content():
    provider, exporter = TracerProvider(), InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    observer = RuntimeTelemetry(tracer=provider.get_tracer('test'))
    with observer.task() as trace:
        span = observer.model_start()
        observer.usage(SimpleNamespace(prompt_tokens=10, completion_tokens=4))
        observer.model_finish(span, 'stream_completed')
        observer.tool_event('tool_started', 'draw_curve')
        observer.tool_event('tool_completed', 'draw_curve')
        identifier = trace['id']
    spans = exporter.get_finished_spans()
    assert {s.name for s in spans} == {'agent.task', 'agent.model', 'agent.tool'}
    assert {f'{s.context.trace_id:032x}' for s in spans} == {identifier}
    root = next(s for s in spans if s.name == 'agent.task')
    assert all(s.parent.span_id == root.context.span_id for s in spans if s != root)
    assert not any(s.events for s in spans)
    model = next(s for s in spans if s.name == 'agent.model')
    assert model.attributes['input_tokens'] == 10 and model.attributes['output_tokens'] == 4
    assert all(set(s.attributes) <= {'outcome', 'tool', 'input_tokens', 'output_tokens'} for s in spans)
    provider.shutdown()
