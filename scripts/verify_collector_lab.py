"""Send content-free spans over real OTLP/HTTP to an isolated collector lab."""

import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'api'))
from runtime_telemetry import RuntimeTelemetry

observer = RuntimeTelemetry(endpoint='http://127.0.0.1:14318/v1/traces')
with observer.task() as trace:
    span = observer.model_start()
    observer.failure('model', 'timeout')
    observer.model_finish(span, 'stream_failed')
    observer.tool_event('tool_started', 'draw_curve')
    observer.tool_event('tool_completed', 'draw_curve')
    trace_id = trace['id']
assert observer.provider.force_flush(timeout_millis=10000)
observer.close()
log, deadline = '', time.monotonic() + 30
while trace_id not in log and time.monotonic() < deadline:
    time.sleep(1)
    log = subprocess.check_output(['docker', 'logs', 'solar-otel-lab'], stderr=subprocess.STDOUT, text=True)
delivered = trace_id in log and all(name in log for name in ['agent.task', 'agent.model', 'agent.tool'])
Path('reports').mkdir(exist_ok=True)
Path('reports/collector-lab.json').write_text(json.dumps({
    'trace_id': trace_id, 'exporter_flushed': True, 'spans_expected': 3,
    'collector_delivery_verified': delivered, 'collector_log': log,
    'destination': 'isolated_ci_otlp_http_collector_debug_backend',
    'enterprise_monitoring_verified': False, 'paid_api_calls': 0}) + '\n')
print(trace_id)
assert delivered, 'Collector did not acknowledge the expected correlated spans'
