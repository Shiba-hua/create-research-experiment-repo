"""Run one command with a timeout, resource samples, and durable exit evidence."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time


def sample_resources(pid):
    record = {'unix_time': time.time(), 'pid': pid}
    try:
        status = Path(f'/proc/{pid}/status').read_text().splitlines()
        record['process_memory_kb'] = {line.split(':')[0]: int(line.split()[1]) for line in status
                                       if line.startswith(('VmRSS:', 'VmHWM:', 'VmSize:'))}
    except FileNotFoundError:
        record['process_memory_kb'] = None
    gpu = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,power.draw,temperature.gpu',
                          '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=15)
    if gpu.returncode == 0:
        fields = gpu.stdout.strip().split(',')
        record['gpu'] = dict(zip(['utilization_percent', 'memory_used_mib', 'power_watts', 'temperature_c'],
                                 [float(value.strip()) for value in fields]))
    else:
        record['gpu_query_error'] = gpu.stderr[-1000:]
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--timeout', type=int, required=True)
    parser.add_argument('--interval', type=int, default=30)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command or args.timeout <= 0:
        raise ValueError('Need command and positive timeout')
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'job.json').exists():
        raise FileExistsError('Use a new job output directory for each attempt')
    revision = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True, check=True).stdout.strip()
    dirty = subprocess.run(['git', 'status', '--porcelain', '--untracked-files=no'], capture_output=True, text=True, check=True).stdout.strip()
    if dirty:
        raise RuntimeError('Commit source changes before launching a recorded job')
    started = time.monotonic()
    with (output / 'stdout.log').open('w') as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        manifest = {'command': command, 'pid': process.pid, 'started_unix': time.time(),
                    'timeout_seconds': args.timeout, 'cwd': os.getcwd(), 'status': 'running',
                    'git_commit': revision}
        (output / 'job.json').write_text(json.dumps(manifest, indent=2)+'\n')
        timed_out = False
        while process.poll() is None:
            if time.monotonic() - started >= args.timeout:
                timed_out = True
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                break
            try:
                sample = sample_resources(process.pid)
            except (subprocess.SubprocessError, ValueError) as error:
                sample = {'unix_time': time.time(), 'resource_sampling_error': str(error)}
            with (output / 'resources.jsonl').open('a') as handle:
                handle.write(json.dumps(sample, allow_nan=False)+'\n')
            try:
                process.wait(timeout=min(args.interval, max(1, args.timeout - (time.monotonic()-started))))
            except subprocess.TimeoutExpired:
                pass
        code = process.wait()
    manifest.update(exit_code=code, timed_out=timed_out, elapsed_seconds=time.monotonic()-started,
                    status='completed' if code == 0 else 'failed')
    (output / 'job.json').write_text(json.dumps(manifest, indent=2)+'\n')
    (output / 'exit_code').write_text(str(code)+'\n')
    raise SystemExit(124 if timed_out else code)


if __name__ == '__main__':
    main()
