import pytest

from kassist.agent.calculator import CalculatorError, evaluate, format_number


@pytest.mark.parametrize("expr,expected", [
    ("3072*4*10000000/1e9", 122.88),
    ("2^10", 1024),
    ("1,536 * 4", 6144),
    ("round(122.88/4, 2)", 30.72),
    ("max(1, 2) + sqrt(16)", 6),
    ("-(3 + 4) * 2", -14),
])
def test_valid_expressions(expr, expected):
    assert evaluate(expr) == pytest.approx(expected)


@pytest.mark.parametrize("expr", [
    "__import__('os').system('echo hi')",
    "open('x')",
    "(1).__class__",
    "9**9**9",
    "1/0",
    "",
    "1+" * 150,
    "x + 1",
    "[1, 2]",
    "True + 1",
])
def test_rejects_unsafe_or_invalid(expr):
    with pytest.raises(CalculatorError):
        evaluate(expr)


def test_format_number():
    assert format_number(122880000000.0) == "122,880,000,000"
    assert format_number(30.72) == "30.72"
