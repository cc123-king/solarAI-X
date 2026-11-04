"""Loopback-only Transformers/PEFT CPU experiment service. This is not vLLM."""

import argparse
import json
import re
from contextlib import contextmanager, nullcontext
from pathlib import Path
from threading import Lock
from time import perf_counter
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from jsonschema import Draft202012Validator, ValidationError
from pydantic import BaseModel, ConfigDict, Field, field_validator

ENGINE = "transformers-peft-cpu"


def parse_tool_calls(text, tools):
    matches = list(re.finditer(r"<tool_call>\s*(.*?)\s*</tool_call>", text, re.DOTALL))
    if ("<tool_call>" in text or "</tool_call>" in text) and not matches:
        raise ValueError("Incomplete model-generated tool call")
    catalog = {tool["function"]["name"]: tool["function"] for tool in tools}
    calls = []
    for match in matches:
        payload = json.loads(match.group(1))
        if not isinstance(payload, dict) or set(payload) != {"name", "arguments"}:
            raise ValueError("Invalid tool-call envelope")
        if payload["name"] not in catalog or not isinstance(payload["arguments"], dict):
            raise ValueError("Unknown tool or invalid arguments")
        json.dumps(payload["arguments"], allow_nan=False)
        schema = catalog[payload["name"]]["parameters"]

        def local_references(node):
            if isinstance(node, dict):
                for key in ["$ref", "$dynamicRef", "$recursiveRef"]:
                    if key in node and not node[key].startswith("#"):
                        raise ValueError("External schema references are not accepted")
                for value in node.values():
                    local_references(value)
            elif isinstance(node, list):
                for value in node:
                    local_references(value)

        local_references(schema)
        Draft202012Validator.check_schema(schema)
        try:
            Draft202012Validator(schema).validate(payload["arguments"])
        except ValidationError as error:
            raise ValueError("Model tool arguments violate schema") from error
        calls.append(
            {
                "id": "call_" + uuid4().hex[:16],
                "type": "function",
                "function": {
                    "name": payload["name"],
                    "arguments": json.dumps(payload["arguments"], ensure_ascii=False),
                },
            }
        )
    content = re.sub(r"<tool_call>\s*.*?\s*</tool_call>", "", text, flags=re.DOTALL).strip()
    if "<tool_call>" in content or "</tool_call>" in content:
        raise ValueError("Incomplete extra tool-call envelope")
    return content or None, calls


class StreamOptions(BaseModel):
    include_usage: bool = False


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")
    model: str
    messages: list[dict[str, Any]] = Field(min_length=1, max_length=64)
    tools: list[dict[str, Any]] = Field(default_factory=list, max_length=16)
    max_tokens: int = Field(default=256, ge=1, le=512)
    temperature: float = Field(default=0, ge=0, le=2)
    stream: bool = False
    stream_options: StreamOptions | None = None
    response_format: dict[str, Any] | None = None
    tool_choice: Any = None

    @field_validator('tools', mode='before')
    @classmethod
    def normalize_optional_tools(cls, value):
        # OpenAI-compatible callers may encode an absent catalog as null.
        return [] if value is None else value


class CPURuntime:
    def __init__(self, base_model, career_adapter, solar_adapter, threads=4):
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        torch.set_num_threads(threads)
        torch.set_num_interop_threads(1)
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model, local_files_only=True, trust_remote_code=False
        )
        base = AutoModelForCausalLM.from_pretrained(
            base_model,
            local_files_only=True,
            trust_remote_code=False,
            torch_dtype=torch.float32,
        )
        self.model = PeftModel.from_pretrained(
            base, solar_adapter, adapter_name="solar-tools", is_trainable=False
        )
        if career_adapter is not None:
            self.model.load_adapter(career_adapter, adapter_name="career-extract", is_trainable=False)
        self.model.eval()
        self.lock = Lock()
        self.aliases = ["agent-base", "solar-tools"] + (["career-extract"] if career_adapter is not None else [])
        self.model_id = "Qwen/Qwen2.5-0.5B-Instruct"

    @contextmanager
    def admission(self):
        if not self.lock.acquire(blocking=False):
            raise HTTPException(429, "CPU experiment service is busy; no request queue")
        try:
            yield
        finally:
            self.lock.release()

    def loss(self, request):
        if request.model not in self.aliases:
            raise HTTPException(404, "Unknown local model alias")
        if request.messages[-1].get("role") != "assistant":
            raise HTTPException(
                400, "Loss verification requires a reference assistant response"
            )
        options = {"tokenize": True, "add_generation_prompt": False}
        if request.tools:
            options["tools"] = request.tools
        full = self.tokenizer.apply_chat_template(request.messages, **options)
        prompt = self.tokenizer.apply_chat_template(
            request.messages[:-1], **dict(options, add_generation_prompt=True)
        )
        if full[: len(prompt)] != prompt or not len(prompt) < len(full) <= 2048:
            raise HTTPException(400, "Invalid completion loss mask")
        inputs = self.torch.tensor([full], dtype=self.torch.long)
        labels = inputs.clone()
        labels[:, : len(prompt)] = -100
        with self.admission():
            context = (
                self.model.disable_adapter()
                if request.model == "agent-base"
                else nullcontext()
            )
            if request.model != "agent-base":
                self.model.set_adapter(request.model)
            with context, self.torch.inference_mode():
                output = self.model(
                    input_ids=inputs,
                    attention_mask=self.torch.ones_like(inputs),
                    labels=labels,
                    use_cache=False,
                )
        return {
            "engine": ENGINE,
            "model": request.model,
            "completion_loss": float(output.loss.item()),
            "assistant_tokens": len(full) - len(prompt),
            "sequence_tokens": len(full),
            "method": "teacher_forced_next_token_loss_assistant_only_not_task_success",
        }

    def complete(self, request):
        if request.model not in self.aliases:
            raise HTTPException(404, "Unknown local model alias")
        if any(
            message.get("role") not in {"system", "user", "assistant", "tool"}
            for message in request.messages
        ):
            raise HTTPException(400, "Unsupported message role")
        options = {
            "tokenize": True,
            "add_generation_prompt": True,
            "return_tensors": "pt",
        }
        if request.tools:
            options["tools"] = request.tools
        inputs = self.tokenizer.apply_chat_template(request.messages, **options)
        if inputs.shape[-1] > 4096:
            raise HTTPException(400, "CPU experiment input exceeds 4096 tokens")
        generation = {
            "max_new_tokens": request.max_tokens,
            "do_sample": request.temperature > 0,
            "pad_token_id": self.tokenizer.eos_token_id,
            "use_cache": True,
            "max_time": 20.0,
        }
        if request.temperature > 0:
            generation["temperature"] = request.temperature
        with self.admission():
            context = (
                self.model.disable_adapter()
                if request.model == "agent-base"
                else nullcontext()
            )
            if request.model != "agent-base":
                self.model.set_adapter(request.model)
            start = perf_counter()
            with context, self.torch.inference_mode():
                output = self.model.generate(
                    inputs, attention_mask=self.torch.ones_like(inputs), **generation
                )
            elapsed = (perf_counter() - start) * 1000
        if elapsed >= 20000:
            raise HTTPException(504, "CPU generation time budget reached")
        generated = output[0, inputs.shape[-1] :]
        raw = self.tokenizer.decode(generated, skip_special_tokens=True).strip()
        try:
            content, calls = parse_tool_calls(raw, request.tools)
        except (ValueError, TypeError, KeyError) as error:
            raise HTTPException(
                502,
                {
                    "code": "invalid_model_tool_output",
                    "error_type": type(error).__name__,
                    "raw_model_output": raw,
                    "engine": ENGINE,
                },
            ) from error
        message = {"role": "assistant", "content": content}
        if calls:
            message["tool_calls"] = calls
        return {
            "id": "cpu-" + uuid4().hex,
            "object": "chat.completion",
            "created": 0,
            "model": request.model,
            "choices": [
                {
                    "index": 0,
                    "message": message,
                    "finish_reason": "tool_calls" if calls else "stop",
                }
            ],
            "usage": {
                "prompt_tokens": int(inputs.shape[-1]),
                "completion_tokens": int(generated.numel()),
                "total_tokens": int(inputs.shape[-1] + generated.numel()),
            },
            "local_verification": {
                "engine": ENGINE,
                "engine_ms": round(elapsed, 3),
                "base_model": self.model_id,
                "adapter": None if request.model == "agent-base" else request.model,
                "json_grammar_enforced": False,
                "streaming": "buffered_sse",
            },
        }


def create_app(runtime):
    app = FastAPI(title="CPU LoRA experiment service — Transformers/PEFT")

    @app.get("/health")
    def health():
        return {
            "status": "ready",
            "engine": ENGINE,
            "models": runtime.aliases,
            "vllm": False,
            "max_output_tokens": 512,
            "generation_time_budget_seconds": 20,
            "queue": "disabled_busy_requests_rejected",
        }

    @app.get("/v1/models")
    def models():
        return {
            "object": "list",
            "data": [
                {
                    "id": name,
                    "object": "model",
                    "created": 0,
                    "owned_by": "local-cpu-experiment",
                }
                for name in runtime.aliases
            ],
        }

    @app.post("/verification/completion-loss")
    def completion_loss(request: ChatRequest):
        return runtime.loss(request)

    @app.post("/v1/chat/completions")
    def completions(request: ChatRequest):
        result = runtime.complete(request)
        if not request.stream:
            return result

        def events():
            choice = result["choices"][0]
            delta = {"role": "assistant"}
            if choice["message"].get("content") is not None:
                delta["content"] = choice["message"]["content"]
            if choice["message"].get("tool_calls"):
                delta["tool_calls"] = [
                    dict(call, index=i)
                    for i, call in enumerate(choice["message"]["tool_calls"])
                ]
            for item in [
                {"index": 0, "delta": delta, "finish_reason": None},
                {"index": 0, "delta": {}, "finish_reason": choice["finish_reason"]},
            ]:
                yield (
                    "data: "
                    + json.dumps(
                        {
                            "id": result["id"],
                            "object": "chat.completion.chunk",
                            "created": 0,
                            "model": result["model"],
                            "choices": [item],
                        },
                        ensure_ascii=False,
                    )
                    + "\n\n"
                )
            if request.stream_options and request.stream_options.include_usage and 'usage' in result:
                yield 'data: ' + json.dumps({'id': result['id'], 'object': 'chat.completion.chunk',
                    'created': 0, 'model': result['model'], 'choices': [], 'usage': result['usage']}) + '\n\n'
            yield "data: [DONE]\n\n"

        return StreamingResponse(events(), media_type="text/event-stream")

    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--career-adapter", type=Path, help='Optional separate career project adapter')
    parser.add_argument("--solar-adapter", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8018)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    for adapter in [args.career_adapter, args.solar_adapter]:
        if adapter is not None and not (adapter / "training_manifest.json").is_file():
            parser.error("Missing actual completed training manifest")
    import uvicorn

    runtime = CPURuntime(
        args.base_model, args.career_adapter, args.solar_adapter, args.threads
    )
    uvicorn.run(
        create_app(runtime),
        host="127.0.0.1",
        port=args.port,
        workers=1,
        log_level="info",
    )


if __name__ == "__main__":
    main()
