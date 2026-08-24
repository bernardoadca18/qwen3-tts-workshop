#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import soundfile as sf

DEFAULT_CORPUS = [
    {"name": "short", "text": "Geralt aprende o sinal Aard."},
    {
        "name": "medium",
        "text": "A jornada continua pelas muralhas antigas. O vento muda, e o perigo se aproxima.",
    },
]


def json_request(url: str, *, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    data = json.dumps(payload).encode() if payload is not None else None
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST" if data is not None else "GET",
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.loads(response.read())


def wait_job(base_url: str, job_id: str) -> dict[str, Any]:
    while True:
        result = json_request(f"{base_url}/v1/narrations/{job_id}")
        if result["status"] in {"completed", "failed"}:
            return result
        time.sleep(1)


def download(url: str, path: Path) -> None:
    with urllib.request.urlopen(url, timeout=120) as response:
        path.write_bytes(response.read())


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproducible Qwen voice benchmark")
    parser.add_argument("--base-url", default="http://localhost:8889")
    parser.add_argument("--corpus", type=Path)
    parser.add_argument("--voice", default="bernardo_v1")
    parser.add_argument("--generation", default="narration_v1")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--output", type=Path, default=Path("benchmark-results/generation"))
    args = parser.parse_args()
    corpus = json.loads(args.corpus.read_text()) if args.corpus else DEFAULT_CORPUS
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = args.output / "manifest.jsonl"
    with manifest.open("w", encoding="utf-8") as output:
        for sample in corpus:
            started = time.monotonic()
            created = json_request(
                f"{args.base_url}/v1/narrations",
                payload={
                    "text": sample["text"],
                    "voice_profile": args.voice,
                    "generation_profile": args.generation,
                    "seed": args.seed,
                },
            )
            result = wait_job(args.base_url, created["id"])
            if result["status"] != "completed":
                raise RuntimeError(f"benchmark job failed: {result['error_category']}")
            wav_path = args.output / f"{sample['name']}.wav"
            download(f"{args.base_url}{result['audio_url']}", wav_path)
            info = sf.info(wav_path)
            elapsed = time.monotonic() - started
            record = {
                "name": sample["name"],
                "text_sha256": hashlib.sha256(sample["text"].encode()).hexdigest(),
                "voice_profile": result["voice_profile"],
                "generation_profile": result["generation_profile"],
                "model_id": result["model_id"],
                "model_revision": result["model_revision"],
                "seed": result["resolved_seed"],
                "wall_clock_seconds": elapsed,
                "output_duration_seconds": info.duration,
                "real_time_factor": elapsed / info.duration,
                "sample_rate": info.samplerate,
                "output_path": str(wav_path),
                "output_sha256": hashlib.sha256(wav_path.read_bytes()).hexdigest(),
                "sample_peak_dbfs": result["sample_peak_dbfs"],
            }
            output.write(json.dumps(record, sort_keys=True) + "\n")
            output.flush()
            print(json.dumps(record, indent=2))


if __name__ == "__main__":
    try:
        main()
    except urllib.error.URLError as exc:
        raise SystemExit(f"voice service request failed: {exc}") from exc
