#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import shutil
import time
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PT_TEXT = """Quando um jogo realmente funciona, você percebe isso antes mesmo de entender exatamente o motivo. A primeira missão pode parecer simples: atravessar uma área, conversar com alguns personagens e encontrar um objeto escondido. Mas, aos poucos, o cenário começa a contar outra história. Um NPC muda de comportamento, uma porta que antes estava aberta aparece trancada, e um detalhe no mapa sugere que existe um caminho que você ainda não explorou.

É nesse momento que a experiência fica interessante. Você pode seguir a rota principal, gastar seus recursos e tentar resolver tudo do jeito mais rápido, ou pode parar, observar e testar uma ideia diferente. Talvez aquela passagem estreita leve a uma sala secreta. Talvez o inimigo mais forte possa ser evitado. Talvez uma decisão feita dez minutos atrás tenha mudado completamente o que acontece agora.

Em jogos maiores, esse tipo de detalhe também revela a qualidade dos sistemas. A inteligência artificial precisa reagir sem parecer previsível; a física precisa ser consistente; a animação precisa acompanhar o ritmo da ação. E, quando o jogo roda a sessenta FPS com uma boa iluminação, efeitos de ray tracing e uma direção de arte convincente, a tecnologia praticamente desaparece. Você deixa de pensar na GPU, no carregamento ou no código e simplesmente entra naquele mundo.

Só que nem tudo precisa ser perfeito. Um pequeno bug pode ser engraçado, uma missão pode falhar de um jeito inesperado, e até um DLC controverso pode gerar boas histórias. No fim, o que permanece não é apenas a pontuação, o tamanho do mapa ou o número de horas jogadas. O que fica na memória são os momentos em que o jogo surpreende você e faz parecer que aquela história aconteceu de um jeito único."""

EN_TEXT = """When a game really works, you usually feel it before you can explain exactly why. The first mission may look simple: cross an area, talk to a few characters, and find an object hidden somewhere nearby. But little by little, the environment starts telling a different story. An NPC changes its behavior, a door that used to be open is suddenly locked, and a small detail on the map suggests there is another route you have not explored yet.

That is where the experience becomes interesting. You can follow the main objective, spend your resources, and try to solve everything as quickly as possible, or you can stop, look around, and test a different idea. Maybe that narrow passage leads to a secret room. Maybe the strongest enemy can be avoided. Maybe a decision you made ten minutes ago has completely changed what happens next.

In larger games, details like these also reveal the quality of the underlying systems. Artificial intelligence needs to react without feeling predictable. Physics needs to remain consistent. Animation has to match the pace of the action. And when the game runs at sixty FPS with convincing lighting, ray tracing, and strong art direction, the technology almost disappears. You stop thinking about the GPU, loading screens, or the code behind the scene, and you simply enter that world.

Of course, not everything has to be perfect. A small bug can be funny, a mission can fail in an unexpected way, and even a controversial DLC can create a memorable discussion. In the end, what stays with you is not just the score, the size of the map, or the number of hours you played. What you remember are the moments when the game surprises you and makes that particular story feel like it happened only to you."""

CASES = [
    {"id": "pt_01", "reference_id": "xvec-shootout-v1/pt_01.wav", "language": "Portuguese", "text": PT_TEXT},
    {"id": "pt_02", "reference_id": "xvec-shootout-v1/pt_02.wav", "language": "Portuguese", "text": PT_TEXT},
    {"id": "pt_03", "reference_id": "xvec-shootout-v1/pt_03.wav", "language": "Portuguese", "text": PT_TEXT},
    {"id": "pt_04", "reference_id": "xvec-shootout-v1/pt_04.wav", "language": "Portuguese", "text": PT_TEXT},
    {"id": "en_01", "reference_id": "xvec-shootout-v1/en_01.wav", "language": "English", "text": EN_TEXT},
    {"id": "en_02", "reference_id": "xvec-shootout-v1/en_02.wav", "language": "English", "text": EN_TEXT},
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def request_json(
    url: str,
    *,
    payload: dict[str, Any] | None = None,
    idempotency_key: str | None = None,
    timeout: int = 60,
) -> dict[str, Any]:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8") if payload is not None else None
    headers = {"Content-Type": "application/json"}
    if idempotency_key:
        headers["Idempotency-Key"] = idempotency_key
    request = urllib.request.Request(
        url,
        data=data,
        headers=headers,
        method="POST" if payload is not None else "GET",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def wait_job(base_url: str, job_id: str, timeout_seconds: int) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    last_status = None
    while time.monotonic() < deadline:
        result = request_json(f"{base_url}/v1/narrations/{job_id}")
        status = result["status"]
        if status != last_status:
            print(
                f"    status={status} "
                f"chunk={result.get('current_chunk', 0)}/{result.get('total_chunks', 0)}"
            )
            last_status = status
        if status in {"completed", "failed"}:
            return result
        time.sleep(1)
    raise TimeoutError(f"job {job_id} exceeded {timeout_seconds}s")


def download(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=300) as response, path.open("wb") as output:
        shutil.copyfileobj(response, output, length=1024 * 1024)


def build_blind_pack(run_dir: Path, records: list[dict[str, Any]], run_id: str) -> None:
    successful = [r for r in records if r["status"] == "completed"]
    blind_dir = run_dir / "blind"
    private_dir = blind_dir / "private"
    blind_dir.mkdir(parents=True, exist_ok=True)
    private_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(run_id)
    mapping: list[dict[str, Any]] = []
    listening_rows: list[dict[str, Any]] = []

    for language, prefix in (("Portuguese", "PT"), ("English", "EN")):
        group = [r for r in successful if r["language"] == language]
        rng.shuffle(group)
        for index, record in enumerate(group):
            letter = chr(ord("A") + index)
            blind_id = f"{prefix}_{letter}"
            src = Path(record["output_path"])
            dst = blind_dir / f"{blind_id}.wav"
            shutil.copy2(src, dst)
            mapping.append(
                {
                    "blind_id": blind_id,
                    "case_id": record["case_id"],
                    "reference_id": record["reference_id"],
                    "reference_sha256": record["reference_sha256"],
                    "output_sha256": record["output_sha256"],
                }
            )
            listening_rows.append(
                {
                    "blind_id": blind_id,
                    "language": language,
                    "duration_seconds": record["duration_seconds"],
                    "identity_similarity_1_5": "",
                    "naturality_1_5": "",
                    "stability_1_5": "",
                    "pronunciation_1_5": "",
                    "prosody_1_5": "",
                    "notes": "",
                }
            )

    with (private_dir / "mapping_PRIVATE.json").open("w", encoding="utf-8") as handle:
        json.dump(mapping, handle, ensure_ascii=False, indent=2)

    fieldnames = [
        "blind_id",
        "language",
        "duration_seconds",
        "identity_similarity_1_5",
        "naturality_1_5",
        "stability_1_5",
        "pronunciation_1_5",
        "prosody_1_5",
        "notes",
    ]
    with (blind_dir / "listening.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(listening_rows)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="One long-form Qwen x-vector narration per prepared reference sample."
    )
    parser.add_argument("--base-url", default="http://localhost:8889")
    parser.add_argument("--generation", default="narration_v2")
    parser.add_argument("--seed", type=int, default=303)
    parser.add_argument(
        "--references-root",
        type=Path,
        default=Path("data/references"),
        help="Host-side Qwen references directory, used only for preflight/hash verification.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark-results/xvec-reference-shootout"),
    )
    parser.add_argument("--timeout-seconds", type=int, default=3600)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()

    base_url = args.base_url.rstrip("/")
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = args.output / run_id

    refs = []
    missing = []
    for case in CASES:
        path = args.references_root / case["reference_id"]
        if path.is_file():
            refs.append(
                {
                    "case_id": case["id"],
                    "reference_id": case["reference_id"],
                    "path": str(path),
                    "sha256": sha256_file(path),
                    "bytes": path.stat().st_size,
                }
            )
        else:
            missing.append(str(path))

    plan = {
        "mode": "run" if args.run else "plan",
        "base_url": base_url,
        "generation_profile": args.generation,
        "seed": args.seed,
        "planned_generations": len(CASES),
        "pt_generations": sum(1 for c in CASES if c["language"] == "Portuguese"),
        "en_generations": sum(1 for c in CASES if c["language"] == "English"),
        "references": refs,
        "missing_references": missing,
        "pt_word_count": len(PT_TEXT.split()),
        "en_word_count": len(EN_TEXT.split()),
    }
    print(json.dumps(plan, ensure_ascii=False, indent=2))

    if missing:
        print("\nERROR: missing reference files. Stage them before running.")
        return 2

    if not args.run:
        print("\nPLAN ONLY: no Qwen request was made. Add --run to execute.")
        return 0

    try:
        ready = request_json(f"{base_url}/health/ready")
    except Exception as exc:
        print(f"\nERROR: Qwen readiness check failed: {exc}")
        return 3

    print(
        "\nQwen ready: "
        f"model={ready.get('model_id')} revision={ready.get('model_revision')} "
        f"default_generation={ready.get('default_generation_profile')}"
    )

    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "corpus_pt.txt").write_text(PT_TEXT + "\n", encoding="utf-8")
    (run_dir / "corpus_en.txt").write_text(EN_TEXT + "\n", encoding="utf-8")
    (run_dir / "plan.json").write_text(
        json.dumps(plan, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    records: list[dict[str, Any]] = []
    ref_by_id = {item["reference_id"]: item for item in refs}

    for position, case in enumerate(CASES, start=1):
        print(
            f"\n[{position}/{len(CASES)}] {case['id']} "
            f"language={case['language']} ref={case['reference_id']}"
        )
        idempotency_key = f"xvec-shootout-{run_id}-{case['id']}"
        started = time.monotonic()

        try:
            created = request_json(
                f"{base_url}/v1/benchmark/narrations",
                payload={
                    "text": case["text"],
                    "language": case["language"],
                    "reference_id": case["reference_id"],
                    "generation_profile": args.generation,
                    "seed": args.seed,
                },
                idempotency_key=idempotency_key,
            )
            result = wait_job(base_url, created["id"], args.timeout_seconds)
        except Exception as exc:
            records.append(
                {
                    "case_id": case["id"],
                    "reference_id": case["reference_id"],
                    "reference_sha256": ref_by_id[case["reference_id"]]["sha256"],
                    "language": case["language"],
                    "generation_profile": args.generation,
                    "requested_seed": args.seed,
                    "status": "request_failed",
                    "error": str(exc),
                }
            )
            print(f"    FAILED: {exc}")
            continue

        elapsed = time.monotonic() - started
        record: dict[str, Any] = {
            "case_id": case["id"],
            "reference_id": case["reference_id"],
            "reference_sha256": ref_by_id[case["reference_id"]]["sha256"],
            "language": case["language"],
            "generation_profile": args.generation,
            "requested_seed": args.seed,
            "resolved_seed": result.get("resolved_seed"),
            "job_id": result["id"],
            "voice_profile": result.get("voice_profile"),
            "model_id": result.get("model_id"),
            "model_revision": result.get("model_revision"),
            "status": result["status"],
            "duration_seconds": result.get("result_duration_seconds"),
            "sample_rate": result.get("result_sample_rate"),
            "channels": result.get("result_channels"),
            "subtype": result.get("result_subtype"),
            "service_output_sha256": result.get("result_sha256"),
            "sample_peak_dbfs": result.get("sample_peak_dbfs"),
            "total_chunks": result.get("total_chunks"),
            "wall_clock_seconds": round(elapsed, 3),
            "metadata": result.get("metadata") or {},
        }

        if result["status"] != "completed":
            record["error_category"] = result.get("error_category")
            record["error_message"] = result.get("error_message")
            records.append(record)
            print(
                f"    FAILED: {record['error_category']} "
                f"{record['error_message']}"
            )
            continue

        output_path = run_dir / "named" / f"{case['id']}.wav"
        download(f"{base_url}{result['audio_url']}", output_path)
        output_sha = sha256_file(output_path)

        record.update(
            {
                "output_path": str(output_path),
                "output_sha256": output_sha,
                "sha256_matches_service": output_sha == result.get("result_sha256"),
            }
        )
        records.append(record)

        duration = result.get("result_duration_seconds") or 0
        range_note = "OK" if 60 <= duration <= 150 else "CHECK_DURATION"
        print(
            f"    completed duration={duration:.1f}s "
            f"chunks={result.get('total_chunks')} "
            f"duration_range={range_note}"
        )

        (run_dir / "manifest.json").write_text(
            json.dumps(records, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    (run_dir / "manifest.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    build_blind_pack(run_dir, records, run_id)

    success_count = sum(1 for r in records if r["status"] == "completed")
    print("\n============================================================")
    print("SHOOTOUT COMPLETE")
    print("============================================================")
    print(f"Run:       {run_dir}")
    print(f"Success:   {success_count}/{len(CASES)}")
    print(f"Named:     {run_dir / 'named'}")
    print(f"Blind:     {run_dir / 'blind'}")
    print(f"Worksheet: {run_dir / 'blind' / 'listening.csv'}")
    print("Mapping:   blind/private/mapping_PRIVATE.json  (open only after listening)")
    print("Manifest:  manifest.json")

    return 0 if success_count == len(CASES) else 1


if __name__ == "__main__":
    raise SystemExit(main())
