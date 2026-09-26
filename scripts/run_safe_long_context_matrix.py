#!/usr/bin/env python3
"""Run a bounded long-context matrix without hiding failures.

The harness records server-reported usage, TTFT, Prefill, Decode and wall time.
It never prints prompts or completions.  A failed request is retained in the
JSON output and, by default, stops the matrix so a CUDA failure is not masked
by later requests.  When run on the GPU host, --telemetry samples nvidia-smi.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import threading
import time
import urllib.error
import urllib.request


WORDS = (
    "system kernel memory buffer thread process socket packet register cache "
    "pointer allocate schedule interrupt virtual physical address translate "
    "compile execute branch predict pipeline vector matrix tensor gradient "
    "cluster network storage device driver module segment offset boundary"
).split()


def make_prompt(target_tokens: int, seed: int) -> str:
    rng = random.Random(seed)
    body = " ".join(rng.choice(WORDS) for _ in range(max(1, target_tokens - 8)))
    return f"[safe-context-{seed}] {body}"


def request(base_url: str, model: str, prompt: str, max_tokens: int, headers: dict[str, str]) -> dict[str, object]:
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0,
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/v1/completions",
        json.dumps(payload).encode("utf-8"),
        {"Content-Type": "application/json", **headers},
    )
    started = time.perf_counter()
    first = last = None
    prompt_tokens = completion_tokens = 0
    with urllib.request.urlopen(req, timeout=3600) as response:
        for raw in response:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data: "):
                continue
            item = line[6:]
            if item == "[DONE]":
                break
            data = json.loads(item)
            usage = data.get("usage") or {}
            prompt_tokens = max(prompt_tokens, int(usage.get("prompt_tokens") or 0))
            completion_tokens = max(completion_tokens, int(usage.get("completion_tokens") or 0))
            choices = data.get("choices") or []
            if choices and ((choices[0].get("text") or "") or (choices[0].get("delta") or {}).get("content")):
                now = time.perf_counter()
                first = first or now
                last = now
    finished = time.perf_counter()
    ttft = (first - started) if first else None
    decode_seconds = (last - first) if first and last else None
    return {
        "promptTokens": prompt_tokens,
        "completionTokens": completion_tokens,
        "ttftSeconds": round(ttft, 3) if ttft is not None else None,
        "prefillTokensPerSecond": round(prompt_tokens / ttft, 2) if ttft else None,
        "decodeTokensPerSecond": round((completion_tokens - 1) / decode_seconds, 2)
        if decode_seconds and completion_tokens > 1 else None,
        "wallSeconds": round(finished - started, 3),
        "complete": completion_tokens == max_tokens,
    }


def read_telemetry() -> list[list[float]]:
    try:
        raw = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,temperature.gpu,power.draw,utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return []
    rows = []
    for line in raw.splitlines():
        try:
            rows.append([float(value.strip()) for value in line.split(",")])
        except ValueError:
            continue
    return rows


def summarize_telemetry(samples: list[list[float]]) -> dict[str, object]:
    grouped: dict[str, list[list[float]]] = {}
    for row in samples:
        if len(row) >= 5:
            grouped.setdefault(str(int(row[0])), []).append(row)
    return {
        "samples": len(samples),
        "maxTemperatureC": {gpu: max(row[1] for row in rows) for gpu, rows in grouped.items()},
        "maxPowerW": {gpu: max(row[2] for row in rows) for gpu, rows in grouped.items()},
        "maxUtilizationPct": {gpu: max(row[3] for row in rows) for gpu, rows in grouped.items()},
        "maxMemoryMiB": {gpu: max(row[4] for row in rows) for gpu, rows in grouped.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--targets", default="253955,383955")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=192)
    parser.add_argument("--telemetry", action="store_true")
    parser.add_argument("--continue-after-error", action="store_true")
    args = parser.parse_args()
    if args.rounds < 1 or args.max_tokens < 1:
        parser.error("--rounds and --max-tokens must be positive")

    api_key = os.environ.get("STRICT_API_KEY", "")
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    telemetry: list[list[float]] = []
    stop = False

    def watch() -> None:
        while not stop:
            telemetry.extend(read_telemetry())
            time.sleep(1)

    watcher = threading.Thread(target=watch, daemon=True) if args.telemetry else None
    if watcher:
        watcher.start()

    results: list[dict[str, object]] = []
    aborted = False
    warmup_error: str | None = None
    try:
        try:
            request(args.base_url, args.model, "warmup", 1, headers)
        except Exception as exc:  # noqa: BLE001
            warmup_error = f"{type(exc).__name__}: {str(exc)[:240]}"
            aborted = True
            row = {"phase": "warmup", "validity": "ERROR", "error": warmup_error}
            results.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
        if not warmup_error:
            for target_index, target in enumerate(int(value) for value in args.targets.split(",")):
                for round_index in range(args.rounds):
                    row: dict[str, object] = {
                        "targetContextTokens": target,
                        "round": round_index + 1,
                        "maxTokens": args.max_tokens,
                    }
                    try:
                        row.update(request(args.base_url, args.model, make_prompt(target, 31000 + target_index * 100 + round_index), args.max_tokens, headers))
                        row["validity"] = "PASS" if row["complete"] else "PARTIAL"
                    except Exception as exc:  # noqa: BLE001
                        row.update({"validity": "ERROR", "error": f"{type(exc).__name__}: {str(exc)[:240]}"})
                        results.append(row)
                        print(json.dumps(row, ensure_ascii=False), flush=True)
                        if not args.continue_after_error:
                            aborted = True
                            break
                    else:
                        results.append(row)
                        print(json.dumps(row, ensure_ascii=False), flush=True)
                if aborted:
                    break
    finally:
        stop = True
        if watcher:
            watcher.join(timeout=2)

    output = {
        "baseUrl": args.base_url.rstrip("/"),
        "model": args.model,
        "targets": [int(value) for value in args.targets.split(",")],
        "rounds": args.rounds,
        "maxTokens": args.max_tokens,
        "profile": "temperature=0, ignore_eos=true, streaming usage, warmup discarded",
        "abortedAfterError": aborted,
        "warmupError": warmup_error,
        "results": results,
        "telemetry": summarize_telemetry(telemetry) if args.telemetry else None,
    }
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(output, handle, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
