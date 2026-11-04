"""Launch a loopback-only local LoRA/MCP/API/UI stack without modifying .env."""

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from contextlib import ExitStack
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def port(value):
    parsed = int(value)
    if not 1024 <= parsed <= 65535:
        raise argparse.ArgumentTypeError('Use a port from 1024 to 65535')
    return parsed


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-model', type=Path, required=True, help='Existing local Qwen model directory')
    parser.add_argument('--solar-adapter', type=Path, required=True, help='Completed local PEFT adapter directory')
    parser.add_argument('--encoder', type=Path, help='Optional existing SentenceTransformer directory')
    parser.add_argument('--encoder-profile', choices=['symmetric', 'e5'], default='symmetric')
    parser.add_argument('--tool-routing', choices=['all', 'lexical', 'semantic'])
    parser.add_argument('--python', type=Path, default=Path(sys.executable), help='API/MCP/UI Python environment')
    parser.add_argument('--model-python', type=Path, default=Path(sys.executable), help='Transformers/PEFT Python environment')
    parser.add_argument('--model-port', type=port, default=18018)
    parser.add_argument('--mcp-port', type=port, default=18346)
    parser.add_argument('--api-port', type=port, default=18001)
    parser.add_argument('--ui-port', type=port, default=15173)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--max-output-tokens', type=int, default=256, help='CPU completion budget, 1..512 tokens')
    parser.add_argument('--startup-timeout', type=int, default=300, help='Readiness deadline per service in seconds')
    parser.add_argument('--state-dir', type=Path, default=ROOT / '.local' / 'resume-stack')
    parser.add_argument('--namespace', default='local-stack')
    parser.add_argument('--dry-run', action='store_true', help='Print the plan without starting processes or writing files')
    return parser, parser.parse_args()


def validate(parser, args):
    for name in ['base_model', 'solar_adapter', 'encoder', 'python', 'model_python', 'state_dir']:
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, value.expanduser().resolve())
    for name in ['python', 'model_python']:
        if not getattr(args, name).is_file():
            parser.error(f'{name} must point to an existing Python executable')
    for directory, filename in [(args.base_model, 'config.json'), (args.solar_adapter, 'adapter_config.json'),
                                (args.solar_adapter, 'training_manifest.json')]:
        if not (directory / filename).is_file():
            parser.error(f'Missing local asset: {directory / filename}')
    if args.encoder is not None and not args.encoder.is_dir():
        parser.error('--encoder must be an existing local directory')
    args.tool_routing = args.tool_routing or ('semantic' if args.encoder else 'all')
    if args.tool_routing == 'semantic' and not args.encoder:
        parser.error('Semantic routing requires --encoder')
    ports = [args.model_port, args.mcp_port, args.api_port, args.ui_port]
    if len(set(ports)) != 4:
        parser.error('The four service ports must be distinct')
    if not 1 <= args.threads <= 32 or not 5 <= args.startup_timeout <= 1800:
        parser.error('Use 1..32 CPU threads and a 5..1800 second readiness deadline')
    if not 1 <= args.max_output_tokens <= 512:
        parser.error('CPU model output budget must be 1..512 tokens')
    if not 1 <= len(args.namespace) <= 80:
        parser.error('Memory namespace must contain 1..80 characters')
    if args.state_dir.exists() and not args.state_dir.is_dir():
        parser.error('--state-dir must be a directory')


def environment(args):
    env = dict(os.environ)
    env.update({
        'LLM_PROVIDER': 'local-transformers-peft', 'LLM_API_KEY': 'local-loopback',
        'LLM_BASE_URL': f'http://127.0.0.1:{args.model_port}/v1',
        'LLM_DEFAULT_MODEL': 'solar-tools', 'LLM_MODEL_NAME': '本地 LoRA',
        'LLM_MODELS': 'solar-tools|本地 LoRA,agent-base|本地基座',
        'LLM_REASONING_MODELS': '', 'LLM_STREAM_USAGE': 'true',
        'LLM_MAX_OUTPUT_TOKENS': str(args.max_output_tokens),
        'LLM_MAX_RETRIES': '0',
        'SOLAR_TOOL_DESCRIPTIONS': 'summary',
        'MCP_SERVER_URL': f'http://127.0.0.1:{args.mcp_port}/sse', 'MCP_DEBUG': 'false',
        'SOLAR_API_HOST': '127.0.0.1', 'SOLAR_API_PORT': str(args.api_port),
        'SOLAR_API_BASE_URL': f'http://127.0.0.1:{args.api_port}/api',
        'SOLAR_UI_HOST': '127.0.0.1', 'SOLAR_UI_PORT': str(args.ui_port),
        'SOLAR_CORS_ORIGINS': f'http://127.0.0.1:{args.ui_port},http://localhost:{args.ui_port}',
        'SOLAR_STATE_DB': str(args.state_dir / 'state.sqlite3'), 'SOLAR_MEMORY_NAMESPACE': args.namespace,
        'SOLAR_ENCODER_PATH': str(args.encoder) if args.encoder else '',
        'SOLAR_ENCODER_PROFILE': args.encoder_profile, 'SOLAR_TOOL_ROUTING': args.tool_routing,
        'SOLAR_OTLP_ENDPOINT': '',  # An external collector must be configured in a separate deployment.
        'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1', 'HF_HUB_DISABLE_TELEMETRY': '1',
        'GRADIO_ANALYTICS_ENABLED': 'false', 'TOKENIZERS_PARALLELISM': 'false',
        'OPENBLAS_NUM_THREADS': str(args.threads), 'OMP_NUM_THREADS': str(args.threads),
        'MKL_NUM_THREADS': str(args.threads), 'PYTHONUTF8': '1',
    })
    # Keep generated result directories under the existing file-download contract.
    # No environment file, external model directory or shared database is written.
    return env


def commands(args):
    return [
        ('model', [str(args.model_python), str(ROOT / 'scripts/serve_cpu_lora.py'),
                   '--base-model', str(args.base_model), '--solar-adapter', str(args.solar_adapter),
                   '--port', str(args.model_port), '--threads', str(args.threads)],
         f'http://127.0.0.1:{args.model_port}/v1/models'),
        ('mcp', [str(args.python), str(ROOT / 'api/mcpserver.py'), '--host', '127.0.0.1', '--port', str(args.mcp_port)],
         f'http://127.0.0.1:{args.mcp_port}/sse'),
        ('api', [str(args.python), str(ROOT / 'api/main.py')], f'http://127.0.0.1:{args.api_port}/api/health'),
        ('ui', [str(args.python), str(ROOT / 'web/app.py')], f'http://127.0.0.1:{args.ui_port}/'),
    ]


def preflight(args):
    for number in [args.model_port, args.mcp_port, args.api_port, args.ui_port]:
        with socket.socket() as listener:
            if os.name == 'nt':
                listener.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
            listener.bind(('127.0.0.1', number))
    requirements = [(args.python, ['fastapi', 'uvicorn', 'gradio', 'mcp', 'openai', 'torch', 'pandas', 'matplotlib']),
                    (args.model_python, ['torch', 'transformers', 'peft', 'fastapi', 'uvicorn', 'jsonschema'])]
    if args.encoder:
        requirements[0][1].append('sentence_transformers')
    for executable, modules in requirements:
        script = ('import importlib.util,json; print(json.dumps([m for m in ' + repr(modules)
                  + ' if importlib.util.find_spec(m) is None]))')
        options = {'creationflags': subprocess.CREATE_NO_WINDOW} if os.name == 'nt' else {}
        result = subprocess.run([str(executable), '-c', script], capture_output=True, text=True, timeout=30, check=False, **options)
        if result.returncode:
            raise RuntimeError(f'Unable to inspect dependencies with {executable}')
        missing = json.loads(result.stdout)
        if missing:
            raise RuntimeError(f'Missing dependencies in {executable}: {", ".join(missing)}; install the documented requirements first')


def wait_ready(name, process, url, timeout):
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f'{name} exited with code {process.returncode}; inspect its local log')
        try:
            headers = {'Accept': 'text/event-stream'} if name == 'mcp' else {}
            with opener.open(urllib.request.Request(url, headers=headers), timeout=2) as response:
                if response.status == 200:
                    if name == 'api':
                        health = json.load(response)
                        if not health.get('mcp_connected') or not health.get('llm_configured'):
                            raise RuntimeError('API started without the local MCP/model connection; inspect api.log')
                    elif name == 'model':
                        models = {item['id'] for item in json.load(response).get('data', [])}
                        if not {'agent-base', 'solar-tools'} <= models:
                            raise RuntimeError('Local model service is missing the base/LoRA aliases')
                    return
        except (urllib.error.URLError, TimeoutError, OSError):
            pass
        time.sleep(0.5)
    raise RuntimeError(f'{name} readiness timed out; inspect its local log')


def stop(process):
    if process.poll() is not None:
        return
    if os.name == 'nt':
        subprocess.run(['taskkill', '/PID', str(process.pid), '/T', '/F'], stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL, creationflags=subprocess.CREATE_NO_WINDOW, timeout=15, check=False)
    else:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        if os.name != 'nt':
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        else:
            process.kill()
        process.wait(timeout=10)


def main():
    parser, args = arguments()
    validate(parser, args)
    plan = commands(args)
    print(json.dumps({'services': [{'name': name, 'command': command, 'readiness': url}
                                  for name, command, url in plan], 'state_dir': str(args.state_dir),
                      'memory_namespace': args.namespace, 'tool_routing': args.tool_routing,
                      'downloads': False, 'mode': 'dry_run' if args.dry_run else 'local_stack'}, ensure_ascii=False, indent=2), flush=True)
    if args.dry_run:
        return
    processes = []
    args.state_dir.mkdir(parents=True, exist_ok=True)
    lock_path = args.state_dir / 'launcher.lock'
    try:
        lock_fd = os.open(lock_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        parser.error('State directory is already locked. Inspect launcher.lock and its PID before removing a stale lock, or select another --state-dir.')
    try:
        with os.fdopen(lock_fd, 'w') as lock:
            lock.write(str(os.getpid()))
        preflight(args)
        with ExitStack() as files:
            try:
                for name, command, url in plan:
                    log = files.enter_context((args.state_dir / f'{name}.log').open('ab'))
                    options = ({'creationflags': subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW}
                               if os.name == 'nt' else {'start_new_session': True})
                    process = subprocess.Popen(command, cwd=ROOT, env=environment(args), stdin=subprocess.DEVNULL,
                                               stdout=log, stderr=subprocess.STDOUT, **options)
                    processes.append(process)
                    wait_ready(name, process, url, args.startup_timeout)
                    print(f'{name} ready: {url}', flush=True)
                print(f'Open http://127.0.0.1:{args.ui_port}/ ; Ctrl+C stops this launcher\'s four services.', flush=True)
                while all(process.poll() is None for process in processes):
                    time.sleep(1)
                raise RuntimeError('A local service stopped; inspect the logs')
            finally:
                for process in reversed(processes):
                    try:
                        stop(process)
                    except (OSError, subprocess.SubprocessError):
                        print(f'Unable to stop owned process {process.pid}; inspect it manually.', file=sys.stderr)
    except KeyboardInterrupt:
        print('Local stack stopped.', flush=True)
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        parser.exit(1, str(error) + '\n')
    finally:
        lock_path.unlink(missing_ok=True)


if __name__ == '__main__':
    main()
