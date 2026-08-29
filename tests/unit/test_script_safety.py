from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parents[2]


def _plan(script: str, output: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / script),
            *arguments,
            "--output",
            str(output),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def test_real_qwen_benchmarks_are_strict_plans_without_run(tmp_path: Path) -> None:
    cases = [
        _plan("benchmark_generation.py", tmp_path / "generation"),
        _plan(
            "benchmark_references.py",
            tmp_path / "references",
            "--voices",
            "bernardo_v1",
            "--text",
            "Plano seguro.",
        ),
        _plan("soak_test.py", tmp_path / "soak.json", "--count", "1"),
    ]
    for result in cases:
        assert result.returncode == 0, result.stderr
        assert "PLAN ONLY" in result.stdout
    assert not (tmp_path / "generation").exists()
    assert not (tmp_path / "references").exists()
    assert not (tmp_path / "soak.json").exists()
