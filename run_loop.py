import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
import urllib.error
 
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CLASSIFIER_PATH = os.path.join(SCRIPT_DIR, "class.py")
PROGRAM_MD_PATH = os.path.join(SCRIPT_DIR, "Program.md")
SCORER_PATH = os.path.join(SCRIPT_DIR, "Scorer.py")
LOG_PATH = os.path.join(SCRIPT_DIR, "run_loop_log.jsonl")
BEST_PATH = os.path.join(SCRIPT_DIR, "class_best.py")
 
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "").rstrip("/")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "")
 
 
def die(msg):
    print("ERROR: " + msg, file=sys.stderr)
    sys.exit(1)
 
 
def read_file(path):
    with open(path, "r") as f:
        return f.read()
 
 
def write_file(path, content):
    with open(path, "w") as f:
        f.write(content)
 
 
def log_event(event):
    event = dict(event)
    event["ts"] = time.time()
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(event) + "\n")
    print(json.dumps(event))
 
 
def call_llm(system_prompt, user_prompt, max_retries=3, retry_delay=30):
    """Call an OpenAI-compatible /chat/completions endpoint, with retries
    on transient connection failures (e.g. a shared inference gateway
    briefly refusing connections) instead of killing the whole loop."""
    if not OPENAI_BASE_URL or not OPENAI_API_KEY or not OPENAI_MODEL:
        die("OPENAI_BASE_URL, OPENAI_API_KEY, and OPENAI_MODEL must all be set")

    url = OPENAI_BASE_URL + "/chat/completions"
    payload = {
        "model": OPENAI_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.8,
    }
    data = json.dumps(payload).encode("utf-8")

    last_error = None
    for attempt in range(1, max_retries + 1):
        req = urllib.request.Request(url, data=data, method="POST")
        req.add_header("Content-Type", "application/json")
        req.add_header("Authorization", "Bearer " + OPENAI_API_KEY)

        try:
            resp = urllib.request.urlopen(req, timeout=180)
            body = resp.read().decode("utf-8")
            result = json.loads(body)
            try:
                return result["choices"][0]["message"]["content"]
            except (KeyError, IndexError):
                die("Unexpected LLM response shape: {}".format(body[:500]))
                return ""
        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8", errors="replace")
            die("LLM request failed ({}): {}".format(e.code, err_body))
            return ""
        except urllib.error.URLError as e:
            last_error = e.reason
            print("[call_llm] attempt {}/{} failed: {}. Retrying in {}s...".format(
                attempt, max_retries, last_error, retry_delay))
            if attempt < max_retries:
                time.sleep(retry_delay)

    die("Could not reach LLM endpoint after {} attempts: {}".format(max_retries, last_error))
    return ""
 
 
def extract_code(llm_output):
    """Pull a python code block out of the LLM's response, if fenced."""
    text = llm_output.strip()
    if "```" in text:
        parts = text.split("```")
        # parts alternate: text, code, text, code, ...
        for i in range(1, len(parts), 2):
            block = parts[i]
            if block.startswith("python"):
                block = block[len("python"):]
            block = block.lstrip("\n")
            if "def main" in block or "import" in block:
                return block
    return text
 
 
def run_scorer(candidate_path):
    """Invoke Scorer.py on a candidate script and return its parsed JSON result."""
    try:
        proc = subprocess.run(
            [sys.executable, SCORER_PATH, candidate_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            timeout=7500,
        )
    except subprocess.TimeoutExpired:
        return {"status": "crash", "error": "scorer timed out"}
 
    out_lines = proc.stdout.strip().splitlines()
    for line in reversed(out_lines):
        line = line.strip()
        if not line:
            continue
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue
    return {
        "status": "crash",
        "error": "no parseable scorer output",
        "raw_stdout": proc.stdout[-1000:],
        "raw_stderr": proc.stderr[-1000:],
    }
 
 
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iterations", type=int, default=5)
    args = ap.parse_args()
 
    if not os.path.isfile(CLASSIFIER_PATH):
        die("class.py not found at " + CLASSIFIER_PATH)
    if not os.path.isfile(PROGRAM_MD_PATH):
        die("Program.md not found at " + PROGRAM_MD_PATH)
 
    program_md = read_file(PROGRAM_MD_PATH)
 
    print("Scoring current class.py as baseline...")
    baseline = run_scorer(CLASSIFIER_PATH)
    if baseline.get("status") != "ok":
        die("Baseline class.py failed to score: {}".format(baseline))
    best_val_acc = baseline["metrics"].get("val_acc", 0.0)
    shutil.copy(CLASSIFIER_PATH, BEST_PATH)
    log_event({"iteration": 0, "event": "baseline", "val_acc": best_val_acc})
 
    history = [{"iteration": 0, "val_acc": best_val_acc, "kept": True, "note": "baseline"}]
 
    for i in range(1, args.iterations + 1):
        current_code = read_file(BEST_PATH)
 
        system_prompt = program_md
        user_prompt = (
            "Here is the current best-performing version of quadrant_classifier.py "
            "(current best val_acc = {:.4f}):\n\n"
            "```python\n{}\n```\n\n"
            "Here is the run history so far (iteration, val_acc, whether it was kept):\n"
            "{}\n\n"
            "Propose ONE modified, complete, self-contained version of this script "
            "that you believe will improve val_acc. Return ONLY the full python code "
            "in a single ```python code block, nothing else."
        ).format(best_val_acc, current_code, json.dumps(history[-5:]))
 
        print("[iter {}] Requesting proposal from LLM...".format(i))
        llm_output = call_llm(system_prompt, user_prompt)
        candidate_code = extract_code(llm_output)
 
        candidate_path = os.path.join(SCRIPT_DIR, "class_candidate_{}.py".format(i))
        write_file(candidate_path, candidate_code)
 
        print("[iter {}] Scoring candidate...".format(i))
        result = run_scorer(candidate_path)
 
        if result.get("status") == "ok":
            val_acc = result["metrics"].get("val_acc", 0.0)
            kept = val_acc > best_val_acc
            if kept:
                shutil.copy(candidate_path, BEST_PATH)
                shutil.copy(candidate_path, CLASSIFIER_PATH)
                best_val_acc = val_acc
            log_event({
                "iteration": i, "event": "scored", "val_acc": val_acc,
                "kept": kept, "best_val_acc": best_val_acc,
            })
            history.append({"iteration": i, "val_acc": val_acc, "kept": kept})
        else:
            log_event({
                "iteration": i, "event": "crash",
                "error": result.get("error", "unknown"), "best_val_acc": best_val_acc,
            })
            history.append({"iteration": i, "val_acc": None, "kept": False, "note": "crashed"})
 
    print("Done. Best val_acc = {:.4f}. Best version saved at {}".format(best_val_acc, BEST_PATH))
 
 
if __name__ == "__main__":
    main()
