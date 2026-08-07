# Quadrant Classifier + Autoresearch on Kubernetes (Intel Gaudi)

This repo demonstrates using [autoresearch-gaudi](https://github.com/thediymaker/autoresearch-gaudi) — an AI-agent-driven iterative training loop, ported to Intel Gaudi 2 HPUs — to automatically improve a custom image classifier, orchestrated via Kubernetes.

## What this is

A 4-class image classifier that sorts spectrogram images into quadrants (++, --, +-, -+) based on color intensity, running on a Kubernetes pod with an Intel Gaudi 2 HPU. An AI agent (`run_loop.py`) iteratively edits the classifier's code, retrains it, and keeps changes only if they improve validation accuracy — fully logged and reproducible at every step.

## The point of this demo

The goal here isn't chasing a specific accuracy number — it's showing that the tool makes real, measurable, traceable improvements to a working model with no manual tuning involved. Every result in this repo has a matching saved code file, checkpoint, and log entry.

## Prerequisites

- A Kubernetes cluster with a Gaudi-enabled pod already running (see `yaml/` for the manifests used here)
- `kubectl` access to that cluster/namespace
- An OpenAI-compatible API endpoint + key (used by `run_loop.py` to drive the agent's decisions)

## Environment setup

```bash
export AUTORESEARCH_RUNNER=k8s
export AUTORESEARCH_NAMESPACE=<your-namespace>
export AUTORESEARCH_POD=<current-live-pod-name>  # changes if the pod is recreated
export OPENAI_BASE_URL=<your-openai-compatible-endpoint>
export OPENAI_API_KEY=<your-key>
export OPENAI_MODEL=<your-model>
```

**Note:** `AUTORESEARCH_POD` goes stale whenever the pod is rescheduled. Always verify with `kubectl get pods -n $AUTORESEARCH_NAMESPACE` before a run — a stale pod name is the most common failure mode (`"debug pod not running"`).

## Files

| File | What it is |
|---|---|
| `baseline_0795_code.py` | Verified baseline classifier code (EPOCHS=8), val_acc **0.795** — matching code, checkpoint, and Grad-CAM images all included |
| `run_loop.py` | The orchestrator: scores baseline, calls the LLM agent, trains candidates, keeps improvements |
| `Scorer.py` | Scoring logic — runs the classifier in the pod and parses the result |
| `Program.md` | Instructions given to the AI agent (what it can/can't modify) |
| `gradcam_report.py` | Generates Grad-CAM heatmap visualizations for a given model checkpoint |
| `run_history_full.jsonl` | Complete raw log across every session run against this classifier, including crashes/timeouts — included for full transparency |
| `baseline_0795_gradcam/` | Grad-CAM images for the verified baseline, using a fixed sample set |
| `yaml/` | Kubernetes manifests for the debug pod and PVC |

## How to reproduce

1. Deploy the pod: `kubectl apply -f yaml/<your-debug-pod>.yaml`
2. Set the environment variables above, matching your actual pod/namespace names
3. Copy the classifier code into the pod:
   ```bash
   kubectl cp baseline_0795_code.py <namespace>/<pod>:/workspace/autoresearch/quadrant_classifier/quadrant_classifier.py
   ```
4. Run the loop:
   ```bash
   nohup python3 run_loop.py --iterations 5 > run_loop_stdout.log 2>&1 &
   ```
5. Monitor progress:
   ```bash
   tail -f run_loop_log.jsonl        # structured results, one line per iteration
   tail -f run_loop_stdout.log       # raw orchestrator output
   kubectl exec -n <namespace> <pod> -- tail -f /root/.cache/autoresearch/run.log  # live epoch progress
   ```

## Results

Starting from a verified baseline of **0.795** validation accuracy (see `baseline_0795_code.py` and its matching Grad-CAM images), autoresearch was run for multiple iterations. See `run_history_full.jsonl` for the complete, unedited log of every attempt — including any that didn't improve on the baseline, which the agent correctly discarded.

## Credits

- Original autoresearch concept: [Andrej Karpathy](https://github.com/karpathy)
- Gaudi (HPU) port: [thediymaker](https://github.com/thediymaker/autoresearch-gaudi)
- Classifier, Kubernetes orchestration, and this use case: Isaiah Talamantes
