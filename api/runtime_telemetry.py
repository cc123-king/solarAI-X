"""Content-free Prometheus counters/histograms with optional OpenTelemetry export."""

import math
import threading
from collections import Counter, defaultdict
from contextlib import contextmanager
from contextvars import ContextVar
from importlib import import_module
from time import perf_counter
from uuid import uuid4

TOOLS = {'draw_curve', 'draw_table', 'simulate_solar_cell', 'batch_simulate_solar_cell',
         'search_embedded_text', 'process_directory_for_embedding', 'predict_perovskite_aging',
         'predict_perovskite_parameters', 'predict_perovskite_bandgap'}
BOUNDS = (0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0)
TRACE = ContextVar('solar_runtime_trace', default=None)
FAILURES = {'timeout', 'invalid_json', 'invalid_arguments', 'schema', 'unauthorized_tool',
            'tool_result', 'connection', 'model_stream', 'answer_validation', 'internal'}


class RuntimeTelemetry:
    def __init__(self, endpoint='', tracer=None):
        self.lock = threading.Lock()
        self.counters, self.durations = Counter(), defaultdict(list)
        self.tracer, self.provider = tracer, None
        if endpoint:
            sdk = import_module('opentelemetry.sdk.trace')
            resources = import_module('opentelemetry.sdk.resources')
            exporters = import_module('opentelemetry.exporter.otlp.proto.http.trace_exporter')
            processors = import_module('opentelemetry.sdk.trace.export')
            self.provider = sdk.TracerProvider(resource=resources.Resource.create({'service.name': 'solarai-x'}))
            self.provider.add_span_processor(processors.BatchSpanProcessor(exporters.OTLPSpanExporter(endpoint=endpoint)))
            self.tracer = self.provider.get_tracer('solarai-x.runtime')

    def increment(self, name, label, value=1):
        allowed = {'tasks': {'completed', 'failed', 'cancelled'},
                   'failures': {f'{stage}:{kind}' for stage in ['model', 'tool', 'protocol', 'answer'] for kind in FAILURES},
                   'model_calls': {'started', 'stream_completed', 'stream_failed'},
                   'tokens': {'input', 'output'}, 'usage_observations': {'reported'},
                   'tool_events': {f'{tool}:{kind}' for tool in TOOLS
                                   for kind in ['tool_started', 'tool_completed', 'tool_failed']}}
        if name not in allowed or label not in allowed[name] or not isinstance(value, int) or value < 0:
            raise ValueError('Unknown metric label or invalid value')
        with self.lock:
            self.counters[(name, label)] += value

    def duration(self, operation, seconds):
        if operation not in {'task', 'model', 'tool'} or not math.isfinite(seconds) or seconds < 0:
            raise ValueError('Invalid duration observation')
        with self.lock:
            # Aggregate fixed buckets; never retain individual requests or user identifiers.
            bins = self.durations[operation]
            if not bins:
                bins.extend([0.0] * (len(BOUNDS)+2))
            for i, upper in enumerate(BOUNDS):
                if seconds <= upper:
                    bins[i] += 1
            bins[-2] += 1
            bins[-1] += seconds

    def failure(self, stage, kind):
        """Only fixed labels, never exception messages, user IDs or argument values."""
        self.increment('failures', f'{stage}:{kind}')
        trace = TRACE.get()
        if trace and trace.get('model_span'):
            trace['model_span'].set_attribute('failure.category', kind)

    @contextmanager
    def task(self):
        trace = {'id': uuid4().hex, 'tools': {}}
        token = TRACE.set(trace)
        try:
            if self.tracer:
                with self.tracer.start_as_current_span('agent.task', record_exception=False, set_status_on_exception=False) as span:
                    trace['id'] = f'{span.get_span_context().trace_id:032x}'
                    try:
                        yield trace
                    except BaseException:
                        trace['outcome'] = 'interrupted_or_failed'
                        raise
                    finally:
                        span.set_attribute('outcome', trace.get('outcome', 'completed'))
            else:
                yield trace
        finally:
            for item in trace['tools'].values():
                self.duration('tool', perf_counter()-item[0])
                if item[1]:
                    item[1].set_attribute('outcome', 'interrupted')
                    item[1].end()
            TRACE.reset(token)

    def tool_event(self, kind, tool):
        if tool not in TOOLS or kind not in {'tool_started', 'tool_completed', 'tool_failed'}:
            return
        self.increment('tool_events', f'{tool}:{kind}')
        trace = TRACE.get()
        if trace is not None:
            if kind == 'tool_started':
                span = self.tracer.start_span('agent.tool', attributes={'tool': tool}) if self.tracer else None
                trace['tools'][tool] = (perf_counter(), span)
            elif tool in trace['tools']:
                started, span = trace['tools'].pop(tool)
                self.duration('tool', perf_counter()-started)
                if span:
                    span.set_attribute('outcome', kind)
                    span.end()

    def usage(self, usage):
        if usage is None:
            return
        values = [getattr(usage, key, None) for key in ['prompt_tokens', 'completion_tokens']]
        if any(type(v) is not int or v < 0 for v in values):
            return
        self.increment('usage_observations', 'reported')
        for key, value in zip(['input', 'output'], values):
            self.increment('tokens', key, value)
            trace = TRACE.get()
            if trace and trace.get('model_span'):
                trace['model_span'].set_attribute(key + '_tokens', value)

    def model_start(self):
        span = self.tracer.start_span('agent.model') if self.tracer else None
        trace = TRACE.get()
        if trace is not None:
            trace['model_span'] = span
        return span

    def model_finish(self, span, outcome):
        if span:
            span.set_attribute('outcome', outcome)
            span.end()
        trace = TRACE.get()
        if trace is not None:
            trace.pop('model_span', None)

    def render(self):
        with self.lock:
            lines = []
            for (name, label), value in sorted(self.counters.items()):
                lines.append(f'solar_{name}_total{{kind="{label}"}} {value}')
            for operation, bins in sorted(self.durations.items()):
                for i, bound in enumerate(BOUNDS):
                    lines.append(f'solar_duration_seconds_bucket{{operation="{operation}",le="{bound}"}} {bins[i]}')
                lines.extend([f'solar_duration_seconds_bucket{{operation="{operation}",le="+Inf"}} {bins[-2]}',
                              f'solar_duration_seconds_count{{operation="{operation}"}} {bins[-2]}',
                              f'solar_duration_seconds_sum{{operation="{operation}"}} {bins[-1]}'])
        return '\n'.join(lines) + '\n'

    def close(self):
        if self.provider:
            self.provider.shutdown()
