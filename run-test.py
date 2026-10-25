"""Run the prepared local test and restore the original ComfyUI service."""
import json
import pathlib
import subprocess
import time
import urllib.error
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parent
EVIDENCE = ROOT / 'test-evidence'
BASE = 'http://127.0.0.1:8192'


def request(path, data=None, base=BASE):
    payload = None if data is None else json.dumps(data).encode()
    req = urllib.request.Request(base + path, data=payload,
                                 headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=15) as response:
        return json.load(response)


def wait_ready(base, timeout=120):
    until = time.monotonic() + timeout
    while time.monotonic() < until:
        try:
            return request('/system_stats', base=base)
        except (OSError, urllib.error.URLError):
            time.sleep(1)
    raise TimeoutError(f'Server not ready: {base}')


def run():
    queue = request('/queue', base='http://127.0.0.1:8191')
    if queue['queue_running'] or queue['queue_pending']:
        raise RuntimeError('Original ComfyUI queue is no longer empty')
    server = None
    stopped = False
    log = None
    result = {'started': time.strftime('%Y-%m-%dT%H:%M:%S%z')}
    try:
        subprocess.run(['systemctl', '--user', 'stop', 'comfyui-qwen21.service'], check=True)
        stopped = True
        print('Original ComfyUI stopped; starting isolated GPU test.', flush=True)
        log = (EVIDENCE / 'gpu-server.log').open('w')
        server = subprocess.Popen(['bash', str(ROOT / 'start-test.sh')], cwd=ROOT,
                                  stdout=log, stderr=subprocess.STDOUT)
        stats = wait_ready(BASE)
        if stats['devices'][0]['type'] != 'cuda':
            raise RuntimeError('Test server is not using CUDA')
        (EVIDENCE / 'gpu-system-stats.json').write_text(json.dumps(stats, indent=2))
        prompt = json.loads((ROOT / 'user/default/fasth3-test-api.json').read_text())
        start = time.monotonic()
        submitted = request('/prompt', {'prompt': prompt, 'client_id': 'fasth3-spark-test'})
        result['submission'] = submitted
        prompt_id = submitted['prompt_id']
        print(f'Test submitted: {prompt_id}', flush=True)
        next_report = 0
        while time.monotonic() - start < 1800:
            elapsed = time.monotonic() - start
            history = request('/history/' + prompt_id)
            if prompt_id in history:
                result['elapsed_seconds'] = round(elapsed, 3)
                result['history'] = history[prompt_id]
                print(f'Test ended after {elapsed:.1f} s: {history[prompt_id]["status"]["status_str"]}', flush=True)
                break
            if elapsed >= next_report:
                print(f'Generation running: {elapsed:.0f} seconds.', flush=True)
                next_report = elapsed + 30
            time.sleep(2)
        else:
            request('/interrupt', {})
            raise TimeoutError('Generation exceeded 30 minutes')
    except Exception as error:
        result['error'] = str(error)
        print(f'Test failed: {error}', flush=True)
    finally:
        if server is not None and server.poll() is None:
            server.terminate()
            try:
                server.wait(timeout=30)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait(timeout=10)
        if log is not None:
            log.close()
        if stopped:
            subprocess.run(['systemctl', '--user', 'start', 'comfyui-qwen21.service'], check=True)
            restored = wait_ready('http://127.0.0.1:8191')
            result['original_service_restored'] = restored['devices'][0]['type'] == 'cuda'
            print('Original ComfyUI restored.', flush=True)
        (EVIDENCE / 'gpu-test-result.json').write_text(json.dumps(result, indent=2))


if __name__ == '__main__':
    run()
