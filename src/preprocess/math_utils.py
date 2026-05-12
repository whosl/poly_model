from __future__ import annotations

import math


def normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def clip(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))
