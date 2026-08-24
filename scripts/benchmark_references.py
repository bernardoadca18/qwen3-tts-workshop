#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from benchmark_generation import download, json_request, wait_job


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare configured voice references")
    parser.add_argument("--base-url", default="http://localhost:8889")
    parser.add_argument("--voices", nargs="+", required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--generation", default="narration_v1")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--output", type=Path, default=Path("benchmark-results/references"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    results = []
    for voice in args.voices:
        started = time.monotonic()
        job = json_request(
            f"{args.base_url}/v1/narrations",
            payload={
                "text": args.text,
                "voice_profile": voice,
                "generation_profile": args.generation,
                "seed": args.seed,
            },
        )
        result = wait_job(args.base_url, job["id"])
        if result["status"] != "completed":
            raise RuntimeError(f"reference benchmark failed for {voice}")
        path = args.output / f"{voice}.wav"
        download(f"{args.base_url}{result['audio_url']}", path)
        results.append(
            {
                "voice_profile": voice,
                "voice_identity": result["metadata"].get("voice_identity"),
                "seed": result["resolved_seed"],
                "wall_clock_seconds": time.monotonic() - started,
                "duration_seconds": result["result_duration_seconds"],
                "sample_rate": result["result_sample_rate"],
                "sha256": result["result_sha256"],
                "path": str(path),
            }
        )
    manifest = args.output / "manifest.json"
    manifest.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
