import json
import os
import shutil
import subprocess
import sys
 
NAMESPACE = os.environ.get("AUTORESEARCH_NAMESPACE", "user-irtalama")
POD = os.environ.get("AUTORESEARCH_POD", "autoresearch-debug-quadrant")
POD_TRAIN_PATH = "/workspace/autoresearch/quadrant_classifier/quadrant_classifier.py"
LOG_PATH = "/root/.cache/autoresearch/run.log"
 
RUN_TIMEOUT = 7200
 
 
def _run(cmd, timeout):
    return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, timeout=timeout)
 
 
def crash(msg, log=None):
    out = {"status": "crash", "metrics": {}, "error": msg}
    if log:
        out["log_tail"] = log[-2000:]
    print(json.dumps(out))
    sys.exit(0)
 
 
def parse_metrics(text: str) -> dict:
    for line in reversed(text.strip().splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict) and "metrics" in parsed:
            return parsed
    return {}
 
# ---------------------------------------------------------------#
# Run section
# ---------------------------------------------------------------#
 
 
def detect_runner() -> str:
    choice = os.environ.get("AUTORESEARCH_RUNNER", "").strip().lower()
    if choice in ("local", "k8s"):
        return choice
    if shutil.which("kubectl"):
        try:
            r = _run(
                [
                    "kubectl",
                    "get",
                    "pod",
                    POD,
                    "-n",
                    NAMESPACE,
                    "-o",
                    "jsonpath={.status.phase}",
                ],
                timeout=30,
            )
            if r.returncode == 0 and r.stdout.strip() == "Running":
                return "k8s"
        except Exception:
            pass
 
    return "local"
 
# ------------------------------------------------------------------#
# local backend
# ------------------------------------------------------------------#
 
 
def run_local(artifact: str) -> dict:
    artifact = os.path.abspath(artifact)
    if not os.path.isfile(artifact):
        crash(f"artifact not found: {artifact}")
    repo_dir = os.path.dirname(artifact)
 
    try:
        proc = subprocess.run(
            [sys.executable, os.path.basename(artifact)],
            cwd=repo_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            timeout=RUN_TIMEOUT,
        )
    except subprocess.TimeoutExpired as e:
        out = e.stdout if isinstance(e.stdout, str) else ""
        crash("training run exceeded timeout", log=out or "")
 
    output = (proc.stdout or "") + (proc.stderr or "")
 
    if proc.returncode != 0:
        crash(f"nonzero exit code {proc.returncode}", log=output)
 
    result = parse_metrics(output)
    if not result:
        crash(
            "no parseable metrics JSON in output (the run did not produce valid output)",
            log=output,
        )
    if result.get("status") != "ok":
        crash(f"training reported non-ok status: {result.get('reason', 'unknown')}", log=output)
    return result["metrics"]
 
# ------------------------------------------------------------------#
# k8s backend
# ------------------------------------------------------------------#
 
 
def _log_tail(n=50):
    try:
        r = _run(["kubectl", "exec", "-n", NAMESPACE, POD, "--",
                   "tail", "-n", str(n), LOG_PATH], timeout=60)
        return r.stdout[-2000:]
    except Exception as e:
        return f"(could not read log: {e})"
 
 
def run_k8s(artifact: str) -> dict:
    try:
        _run(["kubectl", "exec", "-n", NAMESPACE, POD, "--",
              "pkill", "-9", "-f", "quadrant_classifier.py"], timeout=30)
    except Exception:
        pass

    try:
        r = _run(
            [
                "kubectl",
                "get",
                "pod",
                POD,
                "-n",
                NAMESPACE,
                "-o",
                "jsonpath={.status.phase}",
            ],
            timeout=60,
        )
    except subprocess.TimeoutExpired:
        crash("kubectl get pod timed out")
        return {}
    if r.stdout.strip() != "Running":
        crash(f"debug pod not running (phase={r.stdout.strip()!r})")
 
    try:
        r = _run(
            [
                "kubectl",
                "cp",
                artifact,
                f"{NAMESPACE}/{POD}:{POD_TRAIN_PATH}",
            ],
            timeout=120,
        )
    except subprocess.TimeoutExpired:
        crash("kubectl cp timed out")
        return {}
    if r.returncode != 0:
        crash(f"kubectl cp failed: {r.stderr.strip()}")
 
    try:
        _run(
            [
                "kubectl",
                "exec",
                "-n",
                NAMESPACE,
                POD,
                "--",
                "bash",
                "-c",
                f"cd /workspace/autoresearch/quadrant_classifier && python3 quadrant_classifier.py > {LOG_PATH} 2>&1",
            ],
            timeout=RUN_TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        crash("training run exceeded timeout", log=_log_tail())
 
    try:
        r = _run(["kubectl", "exec", "-n", NAMESPACE, POD, "--", "cat", LOG_PATH], timeout=60)
    except subprocess.TimeoutExpired:
        crash("reading log timed out", log=_log_tail())
        return {}
 
    result = parse_metrics(r.stdout)
    if not result:
        crash("no parseable metrics JSON in log (run did not produce valid output)", log=_log_tail())
    if result.get("status") != "ok":
        crash(f"training reported non-ok status: {result.get('reason', 'unknown')}", log=_log_tail())
 
    return result["metrics"]
 
 
def main():
    if len(sys.argv) < 2:
        crash("usage: scorer.py <path-to-quadrant_classifier.py>")
        return
    artifact = sys.argv[1]
 
    runner = detect_runner()
    metrics = run_k8s(artifact) if runner == "k8s" else run_local(artifact)
 
    print(json.dumps({"status": "ok", "metrics": metrics}))
 
 
if __name__ == "__main__":
    main()
