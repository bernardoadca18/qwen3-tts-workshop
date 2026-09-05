#!/usr/bin/env python3
"""Generate fresh samples for speaker drift blind review.

Creates 8-12 WAV samples with distinct seeds using production baseline:
- voice_profile: bernardo_xvec_v2
- generation_profile: narration_v2
- clone_mode: x_vector_only
- language: Portuguese

Extracts speaker embeddings and computes similarity vs reference.
Organizes into blind review package with neutral filenames.

Does NOT calibrate threshold or enable blocking guard.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Add runtime to path when running in container
sys.path.insert(0, "/app/runtime")

import numpy as np
import soundfile as sf


# Production baseline
VOICE_PROFILE = "bernardo_xvec_v2"
GENERATION_PROFILE = "narration_v2"
LANGUAGE = "Portuguese"

# Seed 303 is the production baseline control
CONTROL_SEED = 303
# Additional seeds (deterministic, not pre-labeled)
ADDITIONAL_SEEDS = [101, 202, 404, 505, 606, 707, 808, 909, 1111, 2222, 3333]

# Representative narration texts (15-30 seconds each)
TEXTS = [
    "O documento apresenta uma análise detalhada do mercado brasileiro de tecnologia. "
    "Os especialistas indicam que o setor cresceu significativamente nos últimos cinco anos, "
    "com destaque para as áreas de inteligência artificial e computação em nuvem. "
    "As projeções indicam que essa tendência deve continuar nos próximos anos.",

    "A nova plataforma de streaming oferece uma experiência personalizada para cada usuário. "
    "O algoritmo analisa os padrões de consumo e sugere conteúdos relevantes. "
    "A interface foi redesenhada para facilitar a navegação e descoberta de novos títulos. "
    "Os assinantes podem criar perfis individuais para toda a família.",
]


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def main():
    parser = argparse.ArgumentParser(description="Generate speaker drift review samples")
    parser.add_argument("--output-dir", type=Path, help="Output directory")
    parser.add_argument("--sample-count", type=int, default=10, help="Number of samples to generate")
    args = parser.parse_args()

    # Import voice service components
    from voice_service.config.profiles import load_voice_profile, load_generation_profile
    from voice_service.providers.qwen import Qwen3TTSModel, QwenConfig

    # Create run ID
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + hex(random.randint(0, 0xFFFFFF))[2:].zfill(6)

    # Create output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = Path("/data/speaker-drift-review") / run_id

    output_dir.mkdir(parents=True, exist_ok=True)

    # Select seeds
    seeds = [CONTROL_SEED] + ADDITIONAL_SEEDS[:args.sample_count - 1]

    print(f"Generating {len(seeds)} samples for blind review")
    print(f"Run ID: {run_id}")
    print(f"Output: {output_dir}")
    print(f"Seeds: {seeds}")
    print()

    # Initialize provider
    print("Initializing Qwen provider...")
    config = QwenConfig()
    provider = Qwen3TTSModel(config)
    provider.initialize()

    # Load profiles
    voice_profile = load_voice_profile(VOICE_PROFILE)
    generation_profile = load_generation_profile(GENERATION_PROFILE)

    # Get reference embedding for the voice profile
    print(f"Loading reference embedding for {VOICE_PROFILE}...")
    reference_embedding = provider.get_reference_embedding(voice_profile)
    print(f"Reference embedding shape: {reference_embedding.shape}")

    manifest_entries = []

    for i, seed in enumerate(seeds, start=1):
        sample_id = f"sample_{i:02d}"
        text = TEXTS[i % len(TEXTS)]

        print(f"\nGenerating {sample_id} (seed={seed})...")

        try:
            # Generate audio
            start_time = time.time()
            audio, sample_rate = provider.generate(
                text=text,
                language=LANGUAGE,
                voice_profile=voice_profile,
                generation_profile=generation_profile,
                seed=seed,
            )
            elapsed = time.time() - start_time

            # Calculate duration
            duration_seconds = len(audio) / sample_rate

            # Save to neutral filename
            wav_path = output_dir / f"{sample_id}.wav"
            sf.write(str(wav_path), audio, sample_rate)

            # Extract speaker embedding from generated audio
            generated_embedding = provider.extract_speaker_embedding(audio, sample_rate)

            # Compute similarity
            similarity = cosine_similarity(generated_embedding, reference_embedding)

            print(f"  Duration: {duration_seconds:.2f}s, Similarity: {similarity:.4f}, Time: {elapsed:.1f}s")

            manifest_entries.append({
                "sample_id": sample_id,
                "seed": seed,
                "duration_seconds": round(duration_seconds, 2),
                "speaker_similarity": round(similarity, 6),
                "voice_profile": VOICE_PROFILE,
                "generation_profile": GENERATION_PROFILE,
                "language": LANGUAGE,
                "text_preview": text[:100] + "..." if len(text) > 100 else text,
            })

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            manifest_entries.append({
                "sample_id": sample_id,
                "seed": seed,
                "error": str(e),
            })

    # Write manifest
    manifest_path = output_dir / "manifest.json"
    manifest = {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "voice_profile": VOICE_PROFILE,
        "generation_profile": GENERATION_PROFILE,
        "language": LANGUAGE,
        "reference_embedding_shape": list(reference_embedding.shape),
        "samples": manifest_entries,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))

    # Create listening worksheet
    worksheet_path = output_dir / "listening.md"
    worksheet_lines = [
        "# Speaker Drift Blind Review",
        "",
        f"Run ID: {run_id}",
        f"Created: {datetime.now(timezone.utc).isoformat()}",
        "",
        "For each sample, classify after listening:",
        "",
        "Labels:",
        "- GOOD: Expected masculine PT-BR voice",
        "- BAD_FEMALE_PTPT: Female and/or PT-PT accent detected",
        "- OTHER_QUALITY_ISSUE: Any other problem (specify in notes)",
        "",
    ]

    for entry in manifest_entries:
        sample_id = entry["sample_id"]
        worksheet_lines.extend([
            f"## {sample_id}.wav",
            "",
            "**Classification:** (GOOD / BAD_FEMALE_PTPT / OTHER_QUALITY_ISSUE)",
            "",
            "**Notes:**",
            "",
            "---",
            "",
        ])

    worksheet_path.write_text("\n".join(worksheet_lines))

    # Copy to host-accessible location
    host_output_dir = Path("/data/speaker-drift-review") / run_id

    print()
    print("=" * 60)
    print("SAMPLE GENERATION COMPLETE")
    print("=" * 60)
    print(f"REVIEW_RUN_ID={run_id}")
    print(f"OUTPUT_DIR={host_output_dir}")
    print(f"SAMPLE_COUNT={len([e for e in manifest_entries if 'error' not in e])}")
    print()
    print(f"MANIFEST_PATH={host_output_dir}/manifest.json")
    print(f"LISTENING_WORKSHEET_PATH={host_output_dir}/listening.md")
    print()
    for entry in manifest_entries:
        if 'error' not in entry:
            print(f"SAMPLE_{entry['sample_id'].upper()}_PATH={host_output_dir}/{entry['sample_id']}.wav")
    print()
    print("SECOND_QWEN_LOADED=NO")
    print("THRESHOLD=NOT_CALIBRATED")
    print("SPEAKER_GUARD_BLOCKING=DISABLED")
    print("=" * 60)


if __name__ == "__main__":
    main()
