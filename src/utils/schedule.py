"""Exploration-noise schedules.

DrQ-v2 anneals the exploration standard deviation over training rather than
holding it fixed, and the paper's Algorithm 1 carries the same ``sigma``. The
spec is stored in the config as a string -- ``"linear(1.0,0.1,100000)"`` -- so
a run stays reproducible from its config file plus a seed, with no
hyperparameter on the command line (CLAUDE.md, Conventions).

An unrecognised spec raises rather than falling back to a default. A silent
fallback would change the exploration profile of a run without changing
anything recorded about it, which is the class of failure this project keeps
finding the hard way.
"""

from __future__ import annotations

import re

_LINEAR = re.compile(
    r"^linear\(\s*([\d.eE+-]+)\s*,\s*([\d.eE+-]+)\s*,\s*([\d.eE+-]+)\s*\)$"
)


def linear_schedule(spec: str, step: int) -> float:
    """The value of ``spec`` at ``step``.

    ``"linear(a,b,n)"`` interpolates from ``a`` to ``b`` over ``n`` steps and
    holds at ``b`` thereafter. A bare number is a constant schedule.
    """
    text = str(spec).strip()

    match = _LINEAR.match(text)
    if match is not None:
        start, end, horizon = (float(group) for group in match.groups())
        if horizon <= 0:
            return end
        fraction = min(1.0, max(0.0, step / horizon))
        return start + fraction * (end - start)

    try:
        return float(text)
    except ValueError as exc:
        raise ValueError(
            f"unrecognised schedule {spec!r}; expected 'linear(a,b,n)' or a number"
        ) from exc
