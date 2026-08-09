import pytest

from src.black_scholes import black_scholes_price
from src.binomial_tree import binomial_price
from src.monte_carlo import monte_carlo_price


BASE = {
    "spot": 100.0,
    "strike": 100.0,
    "maturity": 1.0,
    "rate": 0.05,
    "volatility": 0.20,
    "dividend_yield": 0.0,
}


def test_black_scholes_known_values():
    call = black_scholes_price(**BASE, option_type="call")
    put = black_scholes_price(**BASE, option_type="put")

    assert call == pytest.approx(10.4506, abs=1e-4)
    assert put == pytest.approx(5.5735, abs=1e-4)


def test_binomial_converges_to_black_scholes():
    expected = black_scholes_price(**BASE, option_type="call")
    actual = binomial_price(**BASE, option_type="call", steps=1000)

    assert actual == pytest.approx(expected, abs=0.01)


def test_monte_carlo_is_within_four_standard_errors():
    expected = black_scholes_price(**BASE, option_type="call")
    actual, standard_error = monte_carlo_price(
        **BASE,
        option_type="call",
        paths=100_000,
        seed=7,
    )

    assert abs(actual - expected) <= 4 * standard_error


def test_invalid_inputs_raise_clear_errors():
    with pytest.raises(ValueError):
        black_scholes_price(**{**BASE, "spot": 0}, option_type="call")

    with pytest.raises(ValueError):
        binomial_price(**BASE, option_type="call", steps=0)

    with pytest.raises(ValueError):
        monte_carlo_price(**BASE, option_type="call", paths=1)

def test_antithetic_requires_even_paths():
    with pytest.raises(ValueError):
        monte_carlo_price(
            **BASE,
            option_type="call",
            paths=99_999,
            seed=42,
            antithetic=True,
        )
