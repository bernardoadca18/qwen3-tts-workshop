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
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--languages", nargs="+", default=["Portuguese"])
    parser.add_argument("--output", type=Path, default=Path("benchmark-results/references"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    results = []
    for voice in args.voices:
        for seed in args.seeds or [args.seed]:
            for language in args.languages:
                started = time.monotonic()
                job = json_request(
                    f"{args.base_url}/v1/narrations",
                    payload={
                        "text": args.text,
                        "voice_profile": voice,
                        "generation_profile": args.generation,
                        "seed": seed,
                        "language": language,
                    },
                )
                result = wait_job(args.base_url, job["id"])
                elapsed = time.monotonic() - started
                metadata = result.get("metadata") or {}
                record = {
                    "job_id": result["id"],
                    "status": result["status"],
                    "voice_profile": voice,
                    "voice_identity": metadata.get("voice_identity"),
                    "generation_profile": result["generation_profile"],
                    "generation_identity": metadata.get("generation_identity"),
                    "model_id": result["model_id"],
                    "model_revision": result["model_revision"],
                    "chunking_strategy_version": metadata.get("chunking_strategy_version"),
                    "chunking_identity": metadata.get("chunking_identity"),
                    "chunks": metadata.get("chunks", []),
                    "quality_guard_status": metadata.get(
                        "quality_guard_status",
                        "passed" if result["status"] == "completed" else "not_evaluated",
                    ),
                    "seed": result["resolved_seed"],
                    "wall_clock_seconds": elapsed,
                }
                if result["status"] != "completed":
                    record.update(
                        {
                            "error_category": result.get("error_category"),
                            "error_message": result.get("error_message"),
                            "duration_seconds": None,
                            "sample_rate": None,
                            "sha256": None,
                            "path": None,
                            "words_per_minute": None,
                            "chars_per_second": None,
                        }
                    )
                    results.append(record)
                    continue
                path = args.output / f"{voice}--{language}--seed-{seed}.wav"
                download(f"{args.base_url}{result['audio_url']}", path)
                duration = result["result_duration_seconds"]
                record.update(
                    {
                        "duration_seconds": duration,
                        "sample_rate": result["result_sample_rate"],
                        "sha256": result["result_sha256"],
                        "path": str(path),
                        "words_per_minute": (len(args.text.split()) / duration) * 60,
                        "chars_per_second": len(args.text) / duration,
                    }
                )
                results.append(record)
    manifest = args.output / "manifest.json"
    manifest.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
