#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, TextIO, cast

import soundfile as sf

DEFAULT_CORPUS = [
    {"name": "short", "text": "Geralt aprende o sinal Aard."},
    {
        "name": "medium",
        "text": "A jornada continua pelas muralhas antigas. O vento muda, e o perigo se aproxima.",
    },
    {
        "name": "synthetic-long-regression",
        "text": (
            "Em uma análise detalhada, o narrador explica como decisões de projeto, "
            "mudanças de ritmo e escolhas de exploração transformam uma campanha longa. "
            "Primeiro, ele apresenta o contexto histórico e descreve os personagens sem "
            "pressa, mantendo cada ideia em uma frase clara. Depois, compara estratégias, "
            "recursos, distâncias e consequências, sempre conectando a informação seguinte "
            "ao ponto anterior. O objetivo deste texto sintético é representar uma narração "
            "informativa extensa, com pontuação natural em português e parágrafos que possam "
            "ser divididos de modo determinístico. Nenhum trecho exige uma pausa artificial, "
            "e nenhuma frase isolada deve produzir minutos de áudio repetitivo. Ao final, a "
            "conclusão retoma os argumentos principais, destaca o que foi aprendido durante "
            "a jornada e oferece ao público uma síntese objetiva. Esse material não reproduz "
            "um roteiro editorial privado; ele existe apenas para detectar gerações que não "
            "terminam normalmente e se aproximam do limite autoregressivo configurado."
        ),
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
        return cast(dict[str, Any], json.loads(response.read()))


def wait_job(base_url: str, job_id: str) -> dict[str, Any]:
    while True:
        result = json_request(f"{base_url}/v1/narrations/{job_id}")
        if result["status"] in {"completed", "failed"}:
            return result
        time.sleep(1)


def download(url: str, path: Path) -> None:
    with urllib.request.urlopen(url, timeout=120) as response:
        with path.open("wb") as output:
            shutil.copyfileobj(response, output, length=1024 * 1024)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproducible Qwen voice benchmark")
    parser.add_argument("--base-url", default="http://localhost:8889")
    parser.add_argument("--corpus", type=Path)
    parser.add_argument("--voice", default="bernardo_v1")
    parser.add_argument("--generation", default="narration_v1")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--generations", nargs="+")
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--languages", nargs="+", default=["Portuguese"])
    parser.add_argument("--output", type=Path, default=Path("benchmark-results/generation"))
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    corpus = json.loads(args.corpus.read_text()) if args.corpus else DEFAULT_CORPUS
    generations = args.generations or [args.generation]
    seeds = args.seeds or [args.seed]
    plan = {
        "mode": "run" if args.run else "plan",
        "corpus_cases": len(corpus),
        "generations": generations,
        "seeds": seeds,
        "languages": args.languages,
        "planned_generations": len(corpus) * len(generations) * len(seeds) * len(args.languages),
        "output": str(args.output),
    }
    if not args.run:
        print(json.dumps(plan, indent=2, sort_keys=True))
        print("PLAN ONLY: no Qwen request or output write was performed. Add --run to execute.")
        return
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = args.output / "manifest.jsonl"
    with manifest.open("w", encoding="utf-8") as output:
        for sample in corpus:
            for generation in generations:
                for seed in seeds:
                    for language in args.languages:
                        run_benchmark(
                            args=args,
                            sample=sample,
                            generation=generation,
                            seed=seed,
                            language=language,
                            output=output,
                        )


def run_benchmark(
    *,
    args: argparse.Namespace,
    sample: dict[str, str],
    generation: str,
    seed: int,
    language: str,
    output: TextIO,
) -> None:
    started = time.monotonic()
    created = json_request(
        f"{args.base_url}/v1/narrations",
        payload={
            "text": sample["text"],
            "voice_profile": args.voice,
            "generation_profile": generation,
            "seed": seed,
            "language": language,
        },
    )
    result = wait_job(args.base_url, created["id"])
    elapsed = time.monotonic() - started
    metadata = result.get("metadata") or {}
    record = {
        "job_id": result["id"],
        "status": result["status"],
        "name": sample["name"],
        "text_sha256": hashlib.sha256(sample["text"].encode()).hexdigest(),
        "voice_profile": result["voice_profile"],
        "generation_profile": result["generation_profile"],
        "model_id": result["model_id"],
        "model_revision": result["model_revision"],
        "seed": result["resolved_seed"],
        "language": language,
        "wall_clock_seconds": elapsed,
        "voice_identity": metadata.get("voice_identity"),
        "generation_identity": metadata.get("generation_identity"),
        "chunking_strategy_version": metadata.get("chunking_strategy_version"),
        "chunking_identity": metadata.get("chunking_identity"),
        "chunks": metadata.get("chunks", []),
        "quality_guard_status": metadata.get(
            "quality_guard_status",
            "passed" if result["status"] == "completed" else "not_evaluated",
        ),
    }
    if result["status"] != "completed":
        record.update(
            {
                "error_category": result.get("error_category"),
                "error_message": result.get("error_message"),
                "output_duration_seconds": None,
                "real_time_factor": None,
                "sample_rate": None,
                "output_path": None,
                "output_sha256": None,
                "sample_peak_dbfs": None,
                "words_per_minute": None,
                "chars_per_second": None,
            }
        )
        output.write(json.dumps(record, sort_keys=True) + "\n")
        output.flush()
        print(json.dumps(record, indent=2))
        return
    wav_path = args.output / f"{sample['name']}--{generation}--{language}--seed-{seed}.wav"
    download(f"{args.base_url}{result['audio_url']}", wav_path)
    info = sf.info(wav_path)
    record.update(
        {
            "output_duration_seconds": info.duration,
            "real_time_factor": elapsed / info.duration,
            "sample_rate": info.samplerate,
            "output_path": str(wav_path),
            "output_sha256": sha256_file(wav_path),
            "sample_peak_dbfs": result["sample_peak_dbfs"],
            "words_per_minute": (len(sample["text"].split()) / info.duration) * 60,
            "chars_per_second": len(sample["text"]) / info.duration,
        }
    )
    output.write(json.dumps(record, sort_keys=True) + "\n")
    output.flush()
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    try:
        main()
    except urllib.error.URLError as exc:
        raise SystemExit(f"voice service request failed: {exc}") from exc
