from math import exp
from typing import Literal

try:
    from .black_scholes import black_scholes_price
    from .binomial_tree import binomial_price
    from .monte_carlo import monte_carlo_price
except ImportError:
    from black_scholes import black_scholes_price
    from binomial_tree import binomial_price
    from monte_carlo import monte_carlo_price

OptionType = Literal["call", "put"]


def no_arbitrage_bounds(
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    option_type: OptionType = "call",
    dividend_yield: float = 0.0,
) -> tuple[float, float]:
    """Return lower and upper no-arbitrage bounds."""
    spot_pv = spot * exp(-dividend_yield * maturity)
    strike_pv = strike * exp(-rate * maturity)

    if option_type == "call":
        return max(0.0, spot_pv - strike_pv), spot_pv
    if option_type == "put":
        return max(0.0, strike_pv - spot_pv), strike_pv

    raise ValueError("option_type must be 'call' or 'put'")


def put_call_parity_gap(
    call_price: float,
    put_price: float,
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    dividend_yield: float = 0.0,
) -> float:
    """Return the signed put-call parity error."""
    left = call_price - put_price
    right = (
        spot * exp(-dividend_yield * maturity)
        - strike * exp(-rate * maturity)
    )
    return left - right


def finite_difference_greeks(
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    volatility: float,
    option_type: OptionType = "call",
    dividend_yield: float = 0.0,
    spot_step: float | None = None,
    volatility_step: float = 1e-4,
    rate_step: float = 1e-4,
    time_step: float = 1e-5,
) -> dict[str, float]:
    """Estimate Black-Scholes Greeks with central finite differences."""
    ds = spot_step or max(1e-4, spot * 1e-4)
    if maturity <= time_step:
        raise ValueError("maturity must be larger than time_step")

    def price(
        s: float = spot,
        t: float = maturity,
        r: float = rate,
        sigma: float = volatility,
    ) -> float:
        return black_scholes_price(
            s,
            strike,
            t,
            r,
            sigma,
            option_type,
            dividend_yield,
        )

    base = price()
    up_spot = price(s=spot + ds)
    down_spot = price(s=spot - ds)

    delta = (up_spot - down_spot) / (2.0 * ds)
    gamma = (up_spot - 2.0 * base + down_spot) / ds**2
    vega = (
        price(sigma=volatility + volatility_step)
        - price(sigma=volatility - volatility_step)
    ) / (2.0 * volatility_step)
    rho = (
        price(r=rate + rate_step)
        - price(r=rate - rate_step)
    ) / (2.0 * rate_step)

    # Market theta is the negative derivative with respect to time to maturity.
    theta = -(
        price(t=maturity + time_step)
        - price(t=maturity - time_step)
    ) / (2.0 * time_step)

    return {
        "delta": delta,
        "gamma": gamma,
        "vega": vega,
        "theta": theta,
        "rho": rho,
    }


def benchmark_models(
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    volatility: float,
    option_type: OptionType = "call",
    dividend_yield: float = 0.0,
    tree_steps: int = 500,
    paths: int = 100_000,
    seed: int | None = 42,
) -> dict[str, float]:
    """Run all three pricing models on the same option."""
    bs = black_scholes_price(
        spot,
        strike,
        maturity,
        rate,
        volatility,
        option_type,
        dividend_yield,
    )
    tree = binomial_price(
        spot,
        strike,
        maturity,
        rate,
        volatility,
        tree_steps,
        option_type,
        dividend_yield,
    )
    mc, mc_error = monte_carlo_price(
        spot,
        strike,
        maturity,
        rate,
        volatility,
        paths,
        option_type,
        dividend_yield,
        seed,
    )

    return {
        "black_scholes": bs,
        "binomial": tree,
        "monte_carlo": mc,
        "monte_carlo_standard_error": mc_error,
        "binomial_error": tree - bs,
        "monte_carlo_error": mc - bs,
    }
