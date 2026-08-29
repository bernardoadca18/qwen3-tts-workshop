#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import resource
import time
from pathlib import Path

from benchmark_generation import json_request, wait_job


def main() -> None:
    parser = argparse.ArgumentParser(description="Sequential narration soak test")
    parser.add_argument("--base-url", default="http://localhost:8889")
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--text", default="Este e um teste sequencial de estabilidade da voz.")
    parser.add_argument("--voice", default="bernardo_v1")
    parser.add_argument("--generation", default="narration_v1")
    parser.add_argument("--output", type=Path, default=Path("benchmark-results/soak.json"))
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if not args.run:
        print(
            json.dumps(
                {
                    "mode": "plan",
                    "count": args.count,
                    "voice": args.voice,
                    "generation": args.generation,
                    "output": str(args.output),
                },
                indent=2,
                sort_keys=True,
            )
        )
        print("PLAN ONLY: no Qwen request or output write was performed. Add --run to execute.")
        return
    records = []
    for index in range(args.count):
        started = time.monotonic()
        created = json_request(
            f"{args.base_url}/v1/narrations",
            payload={
                "text": args.text,
                "voice_profile": args.voice,
                "generation_profile": args.generation,
                "seed": index + 1,
            },
        )
        result = wait_job(args.base_url, created["id"])
        elapsed = time.monotonic() - started
        duration = result.get("result_duration_seconds")
        records.append(
            {
                "index": index,
                "job_id": created["id"],
                "status": result["status"],
                "latency_seconds": elapsed,
                "output_duration_seconds": duration,
                "real_time_factor": elapsed / duration if duration else None,
                "process_max_rss_kb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                "error_category": result.get("error_category"),
            }
        )
        print(json.dumps(records[-1]))
    summary = {
        "count": len(records),
        "successes": sum(record["status"] == "completed" for record in records),
        "failures": sum(record["status"] != "completed" for record in records),
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
