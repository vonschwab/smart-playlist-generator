import json
import os
import sys
import time
import subprocess
from pathlib import Path

TIMEOUT = 15


def _start_worker():
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
    return subprocess.Popen(
        [sys.executable, "-u", "-m", "playlist_gui.worker"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=str(Path(__file__).resolve().parents[1]),
        env=env,
    )


def test_worker_protocol_ping_and_doctor():
    proc = _start_worker()
    try:
        assert proc.stdout is not None
        assert proc.stdin is not None

        secret = "SHOULD_NOT_APPEAR"
        requests = [
            {"cmd": "ping", "request_id": "req-ping", "protocol_version": 1, "token": secret},
            {"cmd": "doctor", "request_id": "req-doctor", "base_config_path": "config.example.yaml"},
        ]
        for req in requests:
            proc.stdin.write(json.dumps(req) + "\n")
        proc.stdin.flush()
        proc.stdin.close()

        seen_done = set()
        seen_requests = set()
        start = time.time()
        lines = []
        doctor_finished = False
        while time.time() - start < TIMEOUT and not doctor_finished:
            line = proc.stdout.readline()
            if not line:
                if proc.poll() is not None:
                    break
                time.sleep(0.05)
                continue
            lines.append(line)
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            req_id = event.get("request_id")
            if req_id:
                assert req_id in {"req-ping", "req-doctor"}
                seen_requests.add(req_id)
                if req_id == "req-doctor":
                    doctor_finished = True
            if "protocol_version" in event:
                assert event["protocol_version"] == 1
            if event.get("result_type") == "doctor":
                doctor_finished = True
            if event.get("type") == "done":
                if req_id:
                    seen_done.add(req_id)
                if event.get("cmd") == "doctor" or req_id == "req-doctor":
                    doctor_finished = True

        assert doctor_finished
        assert "req-ping" in seen_requests
        joined = "\n".join(lines)
        assert secret not in joined
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            proc.kill()
