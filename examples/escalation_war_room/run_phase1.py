"""Run Phase 1a → 1b → 1c sequentially and summarize pass/fail."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STEPS = [
    "01_tools_and_io.py",
    "02_documents.py",
    "03_multimodal.py",
]


def run_step(name: str) -> int:
    print("\n" + "=" * 72)
    print(f"RUNNING {name}")
    print("=" * 72)
    proc = subprocess.run([sys.executable, str(ROOT / name)], cwd=str(ROOT))
    return proc.returncode


def main() -> None:
    results: list[tuple[str, int]] = []
    for step in STEPS:
        code = run_step(step)
        results.append((step, code))
        if code != 0:
            print(f"\n[FAIL] {step} exited {code} — stopping ladder")
            break
    print("\n======== PHASE 1 SUMMARY ========")
    for name, code in results:
        print(f"  {'PASS' if code == 0 else 'FAIL'}: {name}")
    sys.exit(0 if all(c == 0 for _, c in results) else 1)


if __name__ == "__main__":
    main()
