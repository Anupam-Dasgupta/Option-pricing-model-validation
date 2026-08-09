from math import exp, sqrt
from typing import Literal

import numpy as np

OptionType = Literal["call", "put"]


def monte_carlo_price(
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    volatility: float,
    paths: int = 100_000,
    option_type: OptionType = "call",
    dividend_yield: float = 0.0,
    seed: int | None = 42,
    antithetic: bool = True,
) -> tuple[float, float]:
    """
    Price a European option by Monte Carlo.

    Returns (price, standard_error).
    """
    if spot <= 0 or strike <= 0:
        raise ValueError("spot and strike must be positive")
    if maturity <= 0 or volatility <= 0:
        raise ValueError("maturity and volatility must be positive")
    if paths < 2:
        raise ValueError("paths must be at least 2")
    if option_type not in ("call", "put"):
        raise ValueError("option_type must be 'call' or 'put'")

    rng = np.random.default_rng(seed)

    if antithetic:
        if paths < 4 or paths % 2 != 0:
            raise ValueError(
                "antithetic sampling requires an even number of paths >= 4"
            )
    
        half = paths // 2
        z_half = rng.standard_normal(half)
        z = np.concatenate((z_half, -z_half))
    else:
        z = rng.standard_normal(paths)

    drift = (
        rate - dividend_yield - 0.5 * volatility**2
    ) * maturity
    diffusion = volatility * sqrt(maturity) * z
    terminal_spot = spot * np.exp(drift + diffusion)

    if option_type == "call":
        payoff = np.maximum(terminal_spot - strike, 0.0)
    else:
        payoff = np.maximum(strike - terminal_spot, 0.0)

    discounted_payoff = exp(-rate * maturity) * payoff
    if antithetic:
        pair_payoffs = 0.5 * (
            discounted_payoff[:half]
            + discounted_payoff[half:]
        )
    
        price = float(pair_payoffs.mean())
        standard_error = float(
            pair_payoffs.std(ddof=1) / sqrt(half)
        )
    else:
        price = float(discounted_payoff.mean())
        standard_error = float(
            discounted_payoff.std(ddof=1) / sqrt(paths)
        )
    return price, standard_error
