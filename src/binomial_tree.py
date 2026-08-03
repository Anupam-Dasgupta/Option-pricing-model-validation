from math import exp, sqrt
from typing import Literal

import numpy as np

OptionType = Literal["call", "put"]


def binomial_price(
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    volatility: float,
    steps: int = 200,
    option_type: OptionType = "call",
    dividend_yield: float = 0.0,
) -> float:
    """Price a European option with a Cox-Ross-Rubinstein tree."""
    if spot <= 0 or strike <= 0:
        raise ValueError("spot and strike must be positive")
    if maturity <= 0 or volatility <= 0:
        raise ValueError("maturity and volatility must be positive")
    if steps < 1:
        raise ValueError("steps must be at least 1")
    if option_type not in ("call", "put"):
        raise ValueError("option_type must be 'call' or 'put'")

    dt = maturity / steps
    up = exp(volatility * sqrt(dt))
    down = 1.0 / up
    growth = exp((rate - dividend_yield) * dt)
    probability = (growth - down) / (up - down)

    if not 0.0 <= probability <= 1.0:
        raise ValueError(
            "invalid risk-neutral probability; increase steps or check inputs"
        )

    moves_up = np.arange(steps + 1)
    terminal_spot = spot * up**moves_up * down ** (steps - moves_up)

    if option_type == "call":
        values = np.maximum(terminal_spot - strike, 0.0)
    else:
        values = np.maximum(strike - terminal_spot, 0.0)

    discount = exp(-rate * dt)
    for _ in range(steps):
        values = discount * (
            probability * values[1:]
            + (1.0 - probability) * values[:-1]
        )

    return float(values[0])
