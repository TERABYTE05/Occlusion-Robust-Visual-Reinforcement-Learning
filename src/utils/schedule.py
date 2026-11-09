from __future__ import annotations

import re

_LINEAR = re.compile(
    r"^linear\(\s*([\d.eE+-]+)\s*,\s*([\d.eE+-]+)\s*,\s*([\d.eE+-]+)\s*\)$"
)


def linear_schedule(spec: str, step: int) -> float:
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
