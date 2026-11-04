"""Check real SDK stream compatibility and usage delivery without loading weights."""

from fastapi.testclient import TestClient
from openai import OpenAI

from scripts.serve_cpu_lora import create_app


def test_cpu_stream_reports_usage_once_only_when_requested():
    class Runtime:
        aliases = ('agent-base',)

        def complete(self, request):
            return {'id': 'synthetic-test', 'model': request.model, 'choices': [{
                'index': 0, 'message': {'role': 'assistant', 'content': 'Hello'}, 'finish_reason': 'stop'}],
                'usage': {'prompt_tokens': 13, 'completion_tokens': 2, 'total_tokens': 15}}

    with TestClient(create_app(Runtime())) as http, OpenAI(base_url='http://testserver/v1', api_key='synthetic', http_client=http) as sdk:
            for enabled in [False, True]:
                chunks = list(sdk.chat.completions.create(model='agent-base', messages=[{'role': 'user', 'content': 'Hello'}],
                    stream=True, stream_options={'include_usage': enabled}))
                usage = [chunk.usage for chunk in chunks if chunk.usage is not None]
                assert len(usage) == int(enabled)
                if enabled:
                    assert usage[0].prompt_tokens == 13 and usage[0].total_tokens == 15
                    assert chunks[-1].choices == []
                assert chunks[0].choices[0].delta.content == 'Hello'
