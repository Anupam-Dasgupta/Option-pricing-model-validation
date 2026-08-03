from math import exp

import pytest

from src.black_scholes import black_scholes_greeks, black_scholes_price
from src.validation import (
    finite_difference_greeks,
    no_arbitrage_bounds,
    put_call_parity_gap,
)


BASE = {
    "spot": 100.0,
    "strike": 100.0,
    "maturity": 1.0,
    "rate": 0.05,
    "volatility": 0.20,
    "dividend_yield": 0.0,
}


def test_put_call_parity():
    call = black_scholes_price(**BASE, option_type="call")
    put = black_scholes_price(**BASE, option_type="put")

    gap = put_call_parity_gap(
        call_price=call,
        put_price=put,
        spot=BASE["spot"],
        strike=BASE["strike"],
        maturity=BASE["maturity"],
        rate=BASE["rate"],
        dividend_yield=BASE["dividend_yield"],
    )

    assert gap == pytest.approx(0.0, abs=1e-10)


def test_prices_respect_no_arbitrage_bounds():
    for option_type in ("call", "put"):
        price = black_scholes_price(**BASE, option_type=option_type)
        lower, upper = no_arbitrage_bounds(
            spot=BASE["spot"],
            strike=BASE["strike"],
            maturity=BASE["maturity"],
            rate=BASE["rate"],
            dividend_yield=BASE["dividend_yield"],
            option_type=option_type,
        )

        assert lower <= price <= upper


def test_analytical_and_numerical_greeks_match():
    analytical = black_scholes_greeks(**BASE, option_type="call")
    numerical = finite_difference_greeks(**BASE, option_type="call")

    tolerances = {
        "delta": 1e-5,
        "gamma": 1e-5,
        "vega": 1e-4,
        "theta": 1e-4,
        "rho": 1e-4,
    }

    for greek, tolerance in tolerances.items():
        assert numerical[greek] == pytest.approx(
            analytical[greek],
            abs=tolerance,
        )


def test_call_price_increases_with_spot_and_volatility():
    base_price = black_scholes_price(**BASE, option_type="call")
    higher_spot = black_scholes_price(
        **{**BASE, "spot": 110.0},
        option_type="call",
    )
    higher_volatility = black_scholes_price(
        **{**BASE, "volatility": 0.30},
        option_type="call",
    )

    assert higher_spot > base_price
    assert higher_volatility > base_price
