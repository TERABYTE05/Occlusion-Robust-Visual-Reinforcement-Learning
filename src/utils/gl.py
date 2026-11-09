from __future__ import annotations

import os
import sys

GPU_BACKENDS = frozenset({"egl", "glfw"})


def requested_backend() -> str | None:
    value = os.environ.get("MUJOCO_GL", "").strip().lower()
    return value or None


def resolved_backend() -> str:
    try:
        import mujoco

        module = getattr(mujoco.GLContext, "__module__", "") or ""
        leaf = module.rsplit(".", 1)[-1]
        if leaf in {"egl", "osmesa", "glfw"}:
            return leaf
    except Exception:
        pass
    return requested_backend() or "unknown"


def print_backend(stream=sys.stdout) -> str:
    requested = requested_backend()
    active = resolved_backend()
    print(
        f"[gl] MUJOCO_GL={requested or '<unset>'} -> active backend: {active}",
        file=stream,
        flush=True,
    )
    return active


def require_gpu_backend() -> str:
    active = print_backend()
    if requested_backend() is None:
        raise RuntimeError(
            "MUJOCO_GL is not set. MuJoCo will guess a backend and on a headless "
            "machine that guess is osmesa (CPU software rendering), which is "
            "several times slower and would quietly wreck the run schedule. "
            "Set MUJOCO_GL=egl -- scripts/run.sh does this for you."
        )
    if active not in GPU_BACKENDS:
        raise RuntimeError(
            f"Active rendering backend is {active!r}, which is not GPU-accelerated. "
            f"Expected one of {sorted(GPU_BACKENDS)}. Refusing to run: see "
            "CLAUDE.md, Landmine 4. Override only for a deliberate CPU-only "
            "smoke test, by calling print_backend() instead of this function."
        )
    return active
