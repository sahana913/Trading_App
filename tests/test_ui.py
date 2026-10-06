"""Tests for the small formatting helpers in src/ui.py."""

from src.ui import arrow, money, pct


def test_money_puts_the_sign_before_the_rupee():
    assert money(1234.5) == "₹1,234.50"
    assert money(-1234.5) == "−₹1,234.50"
    assert money(0) == "₹0.00"
    assert money(None) == "–"


def test_pct_and_arrow():
    assert pct(0.0123) == "1.23%" and pct(None) == "–"
    assert (arrow(2), arrow(-2), arrow(0)) == ("▲", "▼", "•")
