#!/usr/bin/env python3
"""Speaker similarity calibration benchmark for Qwen3-TTS.

This script measures speaker embedding similarity between:
1. Reference audio (production voice profile)
2. Generated audio samples (known GOOD and suspected BAD seeds)

Purpose: Derive speaker similarity threshold from real measurements.

Usage:
    # Dry-run (plan only)
    python scripts/benchmark_speaker_similarity.py --text "Teste de áudio" --output /data/audio/benchmarks/speaker-similarity

    # Execute benchmark
    python scripts/benchmark_speaker_similarity.py --run --text "Teste de áudio" --output /data/audio/benchmarks/speaker-similarity

    # With suspected bad seeds
    python scripts/benchmark_speaker_similarity.py --run --text "Teste de áudio" --bad-seeds 61 101 --output /data/audio/benchmarks/speaker-similarity
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


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


def extract_speaker_embedding_from_model(
    model: torch.nn.Module,
    audio_path: str,
    target_sr: int = 24000,
) -> SpeakerEmbedding:
    """Extract speaker embedding from audio using Qwen's internal ECAPA-TDNN.

    Args:
        model: Qwen3TTSForConditionalGeneration model instance
        audio_path: Path to audio file (WAV)
        target_sr: Target sample rate (Qwen expects 24kHz)

    Returns:
        SpeakerEmbedding with extracted vector and metadata
    """
    import librosa
    import soundfile as sf

    # Load audio
    audio, sr = sf.read(audio_path)

    # Convert to mono if stereo
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)

    # Resample to 24kHz if needed
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        sr = target_sr

    # Ensure float32
    audio = audio.astype(np.float32)

    # Extract embedding using Qwen's internal speaker encoder
    with torch.inference_mode():
        # Call extract_speaker_embedding(audio, sr)
        # The model expects numpy array and sample rate
        embedding_tensor = model.extract_speaker_embedding(audio, sr)
        embedding = embedding_tensor.cpu().numpy()

    return SpeakerEmbedding(
        vector=embedding,
        source_path=str(audio_path),
        sample_rate=sr,
        extraction_method="qwen_ecapa_tdnn",
        dimension=len(embedding),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Speaker similarity calibration benchmark")
    parser.add_argument("--base-url", default="http://localhost:8889", help="Qwen voice service URL")
    parser.add_argument("--voice", default="bernardo_xvec_v2", help="Voice profile name")
    parser.add_argument("--generation", default="narration_v1", help="Generation profile name")
    parser.add_argument("--text", required=True, help="Text to synthesize")
    parser.add_argument("--seed", type=int, default=303, help="Primary seed (known GOOD)")
    parser.add_argument("--good-seeds", nargs="+", type=int, default=[303], help="Known GOOD seeds")
    parser.add_argument("--bad-seeds", nargs="+", type=int, default=[], help="Suspected BAD seeds")
    parser.add_argument("--reference-audio", type=Path, help="Reference audio path (defaults to voice profile reference)")
    parser.add_argument("--output", type=Path, default=Path("/data/audio/benchmarks/speaker-similarity"), help="Output directory")
    parser.add_argument("--run", action="store_true", help="Execute benchmark (default is dry-run)")
    parser.add_argument("--repetitions", type=int, default=1, help="Repetitions per seed")
    args = parser.parse_args()

    # Plan mode
    if not args.run:
        plan = {
            "mode": "plan",
            "voice": args.voice,
            "generation": args.generation,
            "text_length": len(args.text),
            "good_seeds": args.good_seeds,
            "bad_seeds": args.bad_seeds,
            "repetitions": args.repetitions,
            "total_generations": (len(args.good_seeds) + len(args.bad_seeds)) * args.repetitions,
            "output": str(args.output),
            "note": "Add --run to execute benchmark. Requires Qwen service running.",
        }
        print(json.dumps(plan, indent=2, sort_keys=True))
        return

    # Execute benchmark
    args.output.mkdir(parents=True, exist_ok=True)

    # Import Qwen model
    print("Loading Qwen model...")
    from qwen_tts import Qwen3TTSModel

    # Load model from HuggingFace (uses cached version if available)
    # Production config: MODEL_ID=Qwen/Qwen3-TTS-12Hz-1.7B-Base
    # MODEL_REVISION=fd4b254389122332181a7c3db7f27e918eec64e3
    model = Qwen3TTSModel.from_pretrained(
        "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        revision="fd4b254389122332181a7c3db7f27e918eec64e3",
        device_map="auto",
        dtype=torch.float16,
    )
    print(f"  Speaker encoder sample rate: {model.model.speaker_encoder_sample_rate}")
    print(f"  Expected embedding dimension: 2048")

    # Find reference audio
    if args.reference_audio:
        ref_path = args.reference_audio
    else:
        # Default to production reference
        ref_path = Path("/data/references/ref_full_processed_raw.wav")

    print(f"Extracting reference embedding from {ref_path}...")
    ref_embedding = extract_speaker_embedding_from_model(model.model, str(ref_path))
    print(f"  Dimension: {ref_embedding.dimension}")

    # Store reference embedding
    ref_embedding_path = args.output / "reference_embedding.json"
    ref_embedding_path.write_text(json.dumps({
        "vector": ref_embedding.vector.tolist(),
        "source_path": ref_embedding.source_path,
        "sample_rate": ref_embedding.sample_rate,
        "extraction_method": ref_embedding.extraction_method,
        "dimension": ref_embedding.dimension,
    }, indent=2))

    # Generate test samples and compute similarities
    results = []

    all_seeds = [(s, "good") for s in args.good_seeds] + [(s, "bad") for s in args.bad_seeds]

    for seed, category in all_seeds:
        for rep in range(args.repetitions):
            print(f"Generating seed={seed} category={category} rep={rep+1}/{args.repetitions}...")

            # Generate audio via API
            import httpx
            from benchmark_generation import wait_job

            started = time.monotonic()
            response = httpx.post(
                f"{args.base_url}/v1/narrations",
                json={
                    "text": args.text,
                    "voice_profile": args.voice,
                    "generation_profile": args.generation,
                    "seed": seed,
                },
                timeout=300,
            )
            response.raise_for_status()
            job = response.json()

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
                continue

            # Download generated audio
            audio_path = args.output / f"seed_{seed}_rep_{rep}.wav"
            audio_url = f"{args.base_url}{result['audio_url']}"
            audio_response = httpx.get(audio_url, timeout=60)
            audio_response.raise_for_status()
            audio_path.write_bytes(audio_response.content)

            # Extract speaker embedding
            print(f"  Extracting embedding...")
            gen_embedding = extract_speaker_embedding_from_model(model.model, str(audio_path))

            # Compute similarity
            similarity = cosine_similarity(ref_embedding.vector, gen_embedding.vector)
            print(f"  Similarity: {similarity:.6f}")

            results.append({
                "seed": seed,
                "category": category,
                "repetition": rep,
                "status": "completed",
                "elapsed_seconds": elapsed,
                "audio_path": str(audio_path),
                "similarity": similarity,
                "embedding_dimension": gen_embedding.dimension,
                "duration_seconds": result.get("result_duration_seconds"),
                "sha256": result.get("result_sha256"),
            })

    # Analyze results
    good_similarities = [r["similarity"] for r in results if r["category"] == "good" and "similarity" in r]
    bad_similarities = [r["similarity"] for r in results if r["category"] == "bad" and "similarity" in r]

    analysis = {
        "schema": "neovids_speaker_similarity_calibration_v1",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "voice": args.voice,
        "reference_audio": str(ref_path),
        "reference_embedding_dimension": ref_embedding.dimension,
        "extraction_method": "qwen_ecapa_tdnn",
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
        # Threshold: midpoint between min_good and max_bad
        min_good = min(good_similarities)
        max_bad = max(bad_similarities)

        if min_good > max_bad:
            # Clear separation
            threshold = (min_good + max_bad) / 2
            analysis["analysis"]["threshold_recommendation"] = {
                "value": threshold,
                "method": "midpoint_clear_separation",
                "rationale": "Clear separation between good and bad distributions",
                "min_good": min_good,
                "max_bad": max_bad,
            }
        else:
            # Overlapping distributions - use conservative threshold
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
        # Only good samples - use conservative threshold
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

    # Write results
    manifest_path = args.output / "manifest.json"
    manifest_path.write_text(json.dumps(analysis, indent=2))
    print()
    print(f"Results written to {manifest_path}")
    print()
    print("=== Analysis ===")
    print(json.dumps(analysis["analysis"], indent=2))


if __name__ == "__main__":
    main()
