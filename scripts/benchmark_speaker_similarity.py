#!/usr/bin/env python3
"""Speaker similarity calibration benchmark for Qwen3-TTS.

This script measures speaker embedding similarity between:
1. Reference audio (production voice profile)
2. Generated audio samples (known GOOD and suspected BAD seeds)

Purpose: Derive speaker similarity threshold from real measurements.

The script uses the running Qwen voice service for BOTH generation and
speaker embedding extraction via the /v1/speaker-embeddings endpoint. It does
NOT load a second model — it reuses the service's already-loaded ECAPA-TDNN
speaker encoder over HTTP.

Usage:
    # Dry-run (plan only)
    python scripts/benchmark_speaker_similarity.py --text "Teste de áudio" --output /data/audio/benchmarks/speaker-similarity

    # Execute benchmark
    python scripts/benchmark_speaker_similarity.py --run --text "Teste de áudio" --output /data/audio/benchmarks/speaker-similarity

    # With suspected bad seeds
    python scripts/benchmark_speaker_similarity.py --run --text "Teste de áudio" --bad-seeds 61 101 --output /data/audio/benchmarks/speaker-similarity

    # With existing WAV files (no generation needed for those categories)
    python scripts/benchmark_speaker_similarity.py --run --text "Teste de áudio" \
        --good-wav /data/good1.wav /data/good2.wav \
        --bad-wav /data/bad1.wav \
        --output /data/audio/benchmarks/speaker-similarity
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import urllib.request
import urllib.error

import numpy as np
import soundfile as sf


@dataclass(frozen=True)
class SpeakerEmbedding:
    """Speaker embedding vector with metadata."""
    vector: np.ndarray
    source_path: str
    sample_rate: int
    extraction_method: str
    dimension: int


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def extract_speaker_embedding_via_api(
    base_url: str,
    audio_path: str,
    *,
    timeout: float = 120.0,
) -> SpeakerEmbedding:
    """Extract speaker embedding from audio using the service /v1/speaker-embeddings API.

    This reuses the already-loaded Qwen model inside the running voice service —
    no second model is loaded.
    """
    import email.generator
    import email.message
    import io

    boundary = "----WebKitFormBoundarySpeakerEmbed"
    with open(audio_path, "rb") as f:
        audio_bytes = f.read()

    # Build multipart/form-data body
    body = io.BytesIO()
    body.write(f"--{boundary}\r\n".encode())
    body.write(f'Content-Disposition: form-data; name="audio"; filename="{Path(audio_path).name}"\r\n'.encode())
    body.write(b"Content-Type: audio/wav\r\n\r\n")
    body.write(audio_bytes)
    body.write(f"\r\n--{boundary}--\r\n".encode())

    req = urllib.request.Request(
        f"{base_url}/v1/speaker-embeddings",
        data=body.getvalue(),
        method="POST",
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        payload = json.loads(resp.read().decode())

    vector = np.asarray(payload["embedding"], dtype=np.float32)
    return SpeakerEmbedding(
        vector=vector,
        source_path=str(audio_path),
        sample_rate=int(payload["sample_rate"]),
        extraction_method=payload.get("extraction_method", "qwen_ecapa_tdnn"),
        dimension=int(payload["dimension"]),
    )


def load_embedding_from_file(path: Path) -> np.ndarray:
    """Load a previously-saved reference embedding JSON."""
    data = json.loads(path.read_text())
    return np.asarray(data["vector"], dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Speaker similarity calibration benchmark")
    parser.add_argument("--base-url", default="http://localhost:8889", help="Qwen voice service URL")
    parser.add_argument("--voice", default="bernardo_xvec_v2", help="Voice profile name")
    parser.add_argument("--generation", default="narration_v2", help="Generation profile name")
    parser.add_argument("--text", required=True, help="Text to synthesize")
    parser.add_argument("--seed", type=int, default=303, help="Primary seed (known GOOD)")
    parser.add_argument("--good-seeds", nargs="+", type=int, default=[303], help="Known GOOD seeds")
    parser.add_argument("--bad-seeds", nargs="+", type=int, default=[], help="Suspected BAD seeds")
    parser.add_argument(
        "--good-wav",
        nargs="+",
        type=Path,
        default=[],
        help="Existing GOOD WAV files to use instead of generating good seeds",
    )
    parser.add_argument(
        "--bad-wav",
        nargs="+",
        type=Path,
        default=[],
        help="Existing BAD WAV files to use instead of generating bad seeds",
    )
    parser.add_argument(
        "--reference-audio",
        type=Path,
        help="Reference audio path (defaults to voice profile reference)",
    )
    parser.add_argument(
        "--reference-embedding",
        type=Path,
        help="Pre-extracted reference embedding JSON (skips reference extraction)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/data/audio/benchmarks/speaker-similarity"),
        help="Output directory",
    )
    parser.add_argument("--run", action="store_true", help="Execute benchmark (default is dry-run)")
    parser.add_argument("--repetitions", type=int, default=1, help="Repetitions per seed")
    args = parser.parse_args()

    has_good = bool(args.good_seeds or args.good_wav)
    has_bad = bool(args.bad_seeds or args.bad_wav)

    # Plan mode
    if not args.run:
        plan = {
            "mode": "plan",
            "voice": args.voice,
            "generation": args.generation,
            "text_length": len(args.text),
            "good_seeds": args.good_seeds,
            "bad_seeds": args.bad_seeds,
            "good_wav": [str(p) for p in args.good_wav],
            "bad_wav": [str(p) for p in args.bad_wav],
            "repetitions": args.repetitions,
            "total_generations": (len(args.good_seeds) + len(args.bad_seeds)) * args.repetitions,
            "total_wav_files": len(args.good_wav) + len(args.bad_wav),
            "output": str(args.output),
            "embedding_source": "service_api",
            "note": "Add --run to execute benchmark. Requires Qwen service running.",
        }
        print(json.dumps(plan, indent=2, sort_keys=True))
        return

    args.output.mkdir(parents=True, exist_ok=True)

    # Resolve reference embedding
    if args.reference_embedding:
        print(f"Loading reference embedding from {args.reference_embedding}...")
        ref_vector = load_embedding_from_file(args.reference_embedding)
        ref_path = args.reference_embedding
        ref_sample_rate = None
        ref_extraction_method = "qwen_ecapa_tdnn"
    else:
        if args.reference_audio:
            ref_path = args.reference_audio
        else:
            ref_path = Path("/data/references/ref_full_processed_raw.wav")
        if not ref_path.exists():
            raise FileNotFoundError(f"Reference audio not found: {ref_path}")
        print(f"Extracting reference embedding from {ref_path} via service API...")
        ref_embedding = extract_speaker_embedding_via_api(args.base_url, str(ref_path))
        ref_vector = ref_embedding.vector
        ref_sample_rate = ref_embedding.sample_rate
        ref_extraction_method = ref_embedding.extraction_method
        print(f"  Dimension: {ref_embedding.dimension}")

    # Persist reference embedding for reuse
    ref_embedding_path = args.output / "reference_embedding.json"
    ref_embedding_path.write_text(json.dumps({
        "vector": ref_vector.tolist(),
        "source_path": str(ref_path),
        "sample_rate": ref_sample_rate,
        "extraction_method": ref_extraction_method,
        "dimension": int(len(ref_vector)),
    }, indent=2))

    # Import wait_job helper
    from benchmark_generation import wait_job

    results = []

    def process_existing_wav(wav_path: Path, category: str) -> None:
        print(f"Processing existing {category} WAV: {wav_path}...")
        if not wav_path.exists():
            print(f"  SKIP: file not found")
            return
        emb = extract_speaker_embedding_via_api(args.base_url, str(wav_path))
        similarity = cosine_similarity(ref_vector, emb.vector)
        print(f"  Similarity: {similarity:.6f}")
        results.append({
            "seed": None,
            "category": category,
            "repetition": 0,
            "status": "completed",
            "audio_path": str(wav_path),
            "similarity": similarity,
            "embedding_dimension": emb.dimension,
            "source": "existing_wav",
        })

    def process_generated_seed(seed: int, category: str, rep: int) -> None:
        print(f"Generating seed={seed} category={category} rep={rep+1}/{args.repetitions}...")
        started = time.monotonic()

        # Create narration job
        req = urllib.request.Request(
            f"{args.base_url}/v1/narrations",
            data=json.dumps({
                "text": args.text,
                "voice_profile": args.voice,
                "generation_profile": args.generation,
                "seed": seed,
            }).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=300) as resp:
            job = json.loads(resp.read().decode())

        result = wait_job(args.base_url, job["id"])
        elapsed = time.monotonic() - started

        if result["status"] != "completed":
            print(f"  FAILED: {result.get('error_message')}")
            results.append({
                "seed": seed,
                "category": category,
                "repetition": rep,
                "status": "failed",
                "error": result.get("error_message"),
                "elapsed_seconds": elapsed,
            })
            return

        audio_path = args.output / f"seed_{seed}_rep_{rep}.wav"
        audio_url = f"{args.base_url}{result['audio_url']}"
        with urllib.request.urlopen(audio_url, timeout=60) as resp:
            audio_path.write_bytes(resp.read())

        print(f"  Extracting embedding via service API...")
        emb = extract_speaker_embedding_via_api(args.base_url, str(audio_path))
        similarity = cosine_similarity(ref_vector, emb.vector)
        print(f"  Similarity: {similarity:.6f}")

        results.append({
            "seed": seed,
            "category": category,
            "repetition": rep,
            "status": "completed",
            "elapsed_seconds": elapsed,
            "audio_path": str(audio_path),
            "similarity": similarity,
            "embedding_dimension": emb.dimension,
            "duration_seconds": result.get("result_duration_seconds"),
            "sha256": result.get("result_sha256"),
            "source": "generated",
        })

    # Existing WAV files
    for wav_path in args.good_wav:
        process_existing_wav(wav_path, "good")
    for wav_path in args.bad_wav:
        process_existing_wav(wav_path, "bad")

    # Generated seeds
    for seed in args.good_seeds:
        for rep in range(args.repetitions):
            process_generated_seed(seed, "good", rep)
    for seed in args.bad_seeds:
        for rep in range(args.repetitions):
            process_generated_seed(seed, "bad", rep)

    # Analyze results
    good_similarities = [r["similarity"] for r in results if r["category"] == "good" and "similarity" in r]
    bad_similarities = [r["similarity"] for r in results if r["category"] == "bad" and "similarity" in r]

    analysis = {
        "schema": "neovids_speaker_similarity_calibration_v1",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "voice": args.voice,
        "generation": args.generation,
        "reference_audio": str(ref_path),
        "reference_embedding_dimension": int(len(ref_vector)),
        "extraction_method": "qwen_ecapa_tdnn",
        "embedding_source": "service_api",
        "text_length": len(args.text),
        "results": results,
        "analysis": {
            "good": {
                "count": len(good_similarities),
                "mean": float(np.mean(good_similarities)) if good_similarities else None,
                "std": float(np.std(good_similarities)) if good_similarities else None,
                "min": float(np.min(good_similarities)) if good_similarities else None,
                "max": float(np.max(good_similarities)) if good_similarities else None,
            },
            "bad": {
                "count": len(bad_similarities),
                "mean": float(np.mean(bad_similarities)) if bad_similarities else None,
                "std": float(np.std(bad_similarities)) if bad_similarities else None,
                "min": float(np.min(bad_similarities)) if bad_similarities else None,
                "max": float(np.max(bad_similarities)) if bad_similarities else None,
            },
        },
    }

    # Compute recommended threshold if we have both good and bad
    if good_similarities and bad_similarities:
        min_good = min(good_similarities)
        max_bad = max(bad_similarities)

        if min_good > max_bad:
            threshold = (min_good + max_bad) / 2
            analysis["analysis"]["threshold_recommendation"] = {
                "value": threshold,
                "method": "midpoint_clear_separation",
                "rationale": "Clear separation between good and bad distributions",
                "min_good": min_good,
                "max_bad": max_bad,
            }
        else:
            threshold = min_good
            analysis["analysis"]["threshold_recommendation"] = {
                "value": threshold,
                "method": "conservative_min_good",
                "rationale": "Overlapping distributions - threshold at minimum good value",
                "min_good": min_good,
                "max_bad": max_bad,
                "warning": "Distributions overlap - manual review recommended",
            }
    elif good_similarities:
        threshold = min(good_similarities)
        analysis["analysis"]["threshold_recommendation"] = {
            "value": threshold,
            "method": "min_good_only",
            "rationale": "No bad samples - threshold at minimum good value",
            "warning": "No bad samples for comparison - collect more data",
        }
    else:
        analysis["analysis"]["threshold_recommendation"] = {
            "value": None,
            "method": "insufficient_data",
            "warning": "No good samples collected",
        }

    manifest_path = args.output / "manifest.json"
    manifest_path.write_text(json.dumps(analysis, indent=2))
    print()
    print(f"Results written to {manifest_path}")
    print()
    print("=== Analysis ===")
    print(json.dumps(analysis["analysis"], indent=2))


if __name__ == "__main__":
    main()
