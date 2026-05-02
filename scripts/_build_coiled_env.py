"""One-shot Coiled software-env build for pv-geom-2026-05."""

from __future__ import annotations

import time

from pv_geom.coiled_env import SOFTWARE_ENV, ensure_software_env


def main() -> None:
    t0 = time.perf_counter()
    print(f"[build] requesting {SOFTWARE_ENV}")
    name = ensure_software_env()
    dt = time.perf_counter() - t0
    print(f"[build] done: {name} ({dt:.0f} s)")


if __name__ == "__main__":
    main()
