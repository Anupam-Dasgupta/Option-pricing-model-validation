from math import erf, exp, log, pi, sqrt
from typing import Literal

OptionType = Literal["call", "put"]


def _check_inputs(
    spot: float,
    strike: float,
    maturity: float,
    volatility: float,
    option_type: OptionType,
) -> None:
    if spot <= 0 or strike <= 0:
        raise ValueError("spot and strike must be positive")
    if maturity <= 0:
        raise ValueError("maturity must be positive")
    if volatility <= 0:
        raise ValueError("volatility must be positive")
    if option_type not in ("call", "put"):
        raise ValueError("option_type must be 'call' or 'put'")


def _normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def _normal_pdf(x: float) -> float:
    return exp(-0.5 * x * x) / sqrt(2.0 * pi)


def _d1_d2(
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    volatility: float,
    dividend_yield: float,
) -> tuple[float, float]:
    root_t = sqrt(maturity)
    d1 = (
        log(spot / strike)
        + (rate - dividend_yield + 0.5 * volatility**2) * maturity
    ) / (volatility * root_t)
    return d1, d1 - volatility * root_t


def black_scholes_price(
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    volatility: float,
    option_type: OptionType = "call",
    dividend_yield: float = 0.0,
) -> float:
    """Price a European call or put under Black-Scholes."""
    _check_inputs(spot, strike, maturity, volatility, option_type)

    d1, d2 = _d1_d2(
        spot, strike, maturity, rate, volatility, dividend_yield
    )
    spot_pv = spot * exp(-dividend_yield * maturity)
    strike_pv = strike * exp(-rate * maturity)

    if option_type == "call":
        return spot_pv * _normal_cdf(d1) - strike_pv * _normal_cdf(d2)

    return strike_pv * _normal_cdf(-d2) - spot_pv * _normal_cdf(-d1)


def black_scholes_greeks(
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    volatility: float,
    option_type: OptionType = "call",
    dividend_yield: float = 0.0,
) -> dict[str, float]:
    """
    Return annualised Greeks.

    Vega and rho are for a 1.00 change in volatility or rate.
    Divide them by 100 for a one-percentage-point change.
    """
    _check_inputs(spot, strike, maturity, volatility, option_type)

    d1, d2 = _d1_d2(
        spot, strike, maturity, rate, volatility, dividend_yield
    )
    root_t = sqrt(maturity)
    spot_discount = exp(-dividend_yield * maturity)
    strike_discount = exp(-rate * maturity)
    density = _normal_pdf(d1)

    gamma = spot_discount * density / (spot * volatility * root_t)
    vega = spot * spot_discount * density * root_t

    if option_type == "call":
        delta = spot_discount * _normal_cdf(d1)
        theta = (
            -spot * spot_discount * density * volatility / (2.0 * root_t)
            - rate * strike * strike_discount * _normal_cdf(d2)
            + dividend_yield * spot * spot_discount * _normal_cdf(d1)
        )
        rho = strike * maturity * strike_discount * _normal_cdf(d2)
    else:
        delta = spot_discount * (_normal_cdf(d1) - 1.0)
        theta = (
            -spot * spot_discount * density * volatility / (2.0 * root_t)
            + rate * strike * strike_discount * _normal_cdf(-d2)
            - dividend_yield * spot * spot_discount * _normal_cdf(-d1)
        )
        rho = -strike * maturity * strike_discount * _normal_cdf(-d2)

    return {
        "delta": delta,
        "gamma": gamma,
        "vega": vega,
        "theta": theta,
        "rho": rho,
    }
