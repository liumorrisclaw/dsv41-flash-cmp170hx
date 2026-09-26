#!/usr/bin/env python3
"""Mode 1 0731 dimension matrix: prefill, decode and bounded long-context fan-out."""
from __future__ import annotations

import argparse
import json
import os
import random
import threading
import time
import urllib.request

WORDS = (
    "system kernel memory buffer thread process socket packet register cache pointer "
    "allocate schedule interrupt virtual physical address translate compile execute "
    "branch predict pipeline vector matrix tensor gradient cluster network storage "
    "device driver module segment offset boundary"
).split()


def prompt(approx_tokens: int, seed: int) -> str:
    rng = random.Random(seed)
    return "[mode1-matrix-%d] " % seed + " ".join(rng.choice(WORDS) for _ in range(approx_tokens))


def post(base: str, model: str, text: str, max_tokens: int, headers: dict[str, str]) -> dict[str, object]:
    payload = {"model": model, "prompt": text, "max_tokens": max_tokens, "temperature": 0, "ignore_eos": True}
    req = urllib.request.Request(f"{base.rstrip('/')}/v1/completions", json.dumps(payload).encode(), {"Content-Type": "application/json", **headers})
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=3600) as response:
            data = json.load(response)
        usage = data.get("usage", {})
        return {"ok": "usage" in data, "usage": usage, "wallSeconds": time.perf_counter() - started}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": f"{type(exc).__name__}: {str(exc)[:240]}", "wallSeconds": time.perf_counter() - started}


def sweep(base: str, model: str, levels: list[int], prompt_tokens: int, max_tokens: int, label: str, headers: dict[str, str]) -> list[dict[str, object]]:
    rows = []
    for level in levels:
        results: list[dict[str, object] | None] = [None] * level
        threads = []
        started = time.perf_counter()
        for index in range(level):
            thread = threading.Thread(target=lambda i=index: results.__setitem__(i, post(base, model, prompt(prompt_tokens, 10000 + level * 100 + i), max_tokens, headers)))
            thread.start()
            threads.append(thread)
        for thread in threads:
            thread.join()
        wall = time.perf_counter() - started
        ok = [row for row in results if row and row.get("ok")]
        prompt_total = sum(int(row.get("usage", {}).get("prompt_tokens", 0)) for row in ok)
        completion_total = sum(int(row.get("usage", {}).get("completion_tokens", 0)) for row in ok)
        rows.append({
            "label": label,
            "concurrency": level,
            "requested": level,
            "ok": len(ok),
            "promptTokens": prompt_total,
            "completionTokens": completion_total,
            "wallSeconds": round(wall, 3),
            "aggregateTokensPerSecond": round(completion_total / wall, 2) if wall else None,
            "aggregatePromptTokensPerSecond": round(prompt_total / wall, 2) if wall else None,
            "errors": [row.get("error") for row in results if row and not row.get("ok")],
            "validity": "PASS" if len(ok) == level else "PARTIAL_OR_ERROR",
        })
        print(json.dumps(rows[-1], ensure_ascii=False), flush=True)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    key = os.environ.get("STRICT_API_KEY", "")
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    post(args.base_url, args.model, "warmup", 1, headers)
    result = {
        "baseUrl": args.base_url.rstrip("/"),
        "model": args.model,
        "startedAt": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "profile": "temperature=0, ignore_eos=true, warmup discarded; no technical/open-prose/code content benchmark",
        "prefillConcurrency": sweep(args.base_url, args.model, [1, 2, 4, 8], 3000, 1, "prefill-3K", headers),
        "decodeConcurrency": sweep(args.base_url, args.model, [1, 4, 8, 16], 60, 300, "decode-60prompt-300output", headers),
        "contextConcurrency": {
            "32K": sweep(args.base_url, args.model, [1, 2, 4], 32000, 64, "context-32K", headers),
            "128K": sweep(args.base_url, args.model, [1, 2, 4], 128000, 64, "context-128K", headers),
        },
    }
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
