"""Run every quality gate: lint, types, tests, regression benchmarks.

    uv run python scripts/check.py           # everything
    uv run python scripts/check.py --fast    # skip the type check

Exits non-zero if any gate fails. The real-data benchmarks run when present
(see scripts/build_benchmark.py); the synthetic one always runs.
"""

from __future__ import annotations

import subprocess
import sys

GATES: list[tuple[str, list[str]]] = [
    ("lint", ["ruff", "check", "src", "tests", "scripts"]),
    ("types", ["mypy", "src"]),
    ("tests + benchmarks", ["pytest", "-q", "-m", "not integration",
                            "--cov=pv_geom", "--cov-fail-under=88"]),
]


def main() -> int:
    fast = "--fast" in sys.argv
    failed: list[str] = []
    for name, cmd in GATES:
        if fast and name == "types":
            continue
        print(f"\n== {name}: {' '.join(cmd)}", flush=True)
        if subprocess.run([sys.executable, "-m", *cmd]).returncode != 0:
            failed.append(name)
    print("\n" + ("all gates passed" if not failed else f"FAILED: {', '.join(failed)}"))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
