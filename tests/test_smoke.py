# tests/test_smoke.py
import os, subprocess, sys, time, requests
from pathlib import Path

PORT = int(os.getenv("PORT", "8501"))
URL = f"http://127.0.0.1:{PORT}/"

def _cmd():
    return [
        sys.executable, "-m", "streamlit", "run", "app.py",
        "--server.port", str(PORT),
        "--server.headless", "true",
        "--browser.gatherUsageStats", "false",
    ]

def wait(url, timeout=45):
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = requests.get(url, timeout=2)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(0.5)
    return False

def test_streamlit_starts_and_serves_root():
    repo = Path(__file__).resolve().parents[1]
    env = {**os.environ, "PORT": str(PORT)}
    proc = subprocess.Popen(_cmd(), cwd=str(repo), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        assert wait(URL, 45), "Streamlit did not become ready"
        r = requests.get(URL, timeout=3)
        assert r.status_code == 200
    finally:
        proc.terminate()
        try:
            proc.wait(10)
        except subprocess.TimeoutExpired:
            proc.kill()
