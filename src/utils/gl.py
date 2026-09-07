"""Rendering-backend detection.

Landmine 4. `MUJOCO_GL=egl` renders on the GPU; `osmesa` is CPU software
rasterisation and is several times slower. Landing on osmesa by accident means
500K-step runs stop fitting in the term, and nothing in the training logs would
say so -- the runs would simply be slow. So every entry point prints the active
backend before doing any work, and `require_gpu_backend` turns a silent
slowdown into a loud failure.

MuJoCo resolves the backend at import time from the ``MUJOCO_GL`` environment
variable; if it is unset it tries glfw, then egl, then osmesa. Because "unset"
is the case that silently lands on osmesa on a headless machine, we treat an
unset variable as an error rather than guessing.
"""

from __future__ import annotations

import os
import sys

#: Backends that rasterise on the GPU. Anything else is software rendering.
GPU_BACKENDS = frozenset({"egl", "glfw"})


def requested_backend() -> str | None:
    """The backend asked for via ``MUJOCO_GL``, or None if it was never set."""
    value = os.environ.get("MUJOCO_GL", "").strip().lower()
    return value or None


def resolved_backend() -> str:
    """The backend MuJoCo actually bound at import time.

    Read from the module that provides ``mujoco.GLContext``, which is selected
    when ``mujoco`` is imported. Falls back to the requested value if that
    introspection ever stops working against a future mujoco release.
    """
    try:
        import mujoco

        module = getattr(mujoco.GLContext, "__module__", "") or ""
        # e.g. "mujoco.egl", "mujoco.osmesa", "mujoco.glfw"
        leaf = module.rsplit(".", 1)[-1]
        if leaf in {"egl", "osmesa", "glfw"}:
            return leaf
    except Exception:  # pragma: no cover - depends on the mujoco build
        pass
    return requested_backend() or "unknown"


def print_backend(stream=sys.stdout) -> str:
    """Print the active rendering backend. Never remove this call (Landmine 4)."""
    requested = requested_backend()
    active = resolved_backend()
    print(
        f"[gl] MUJOCO_GL={requested or '<unset>'} -> active backend: {active}",
        file=stream,
        flush=True,
    )
    return active


def require_gpu_backend() -> str:
    """Print the backend and refuse to continue on a software rasteriser.

    Raises:
        RuntimeError: if ``MUJOCO_GL`` is unset, or resolves to osmesa.
    """
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
