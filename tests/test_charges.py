"""Tests for src/trading/charges.py, checked against hand calculations."""

import pytest

from src.trading.charges import calculate_charges


def test_cnc_buy():
    # turnover 1000: STT 1.00 + exchange 0.0297 + SEBI 0.001 + stamp 0.15
    # + GST 18% of (0 + 0.0297 + 0.001) = 0.0055  -> 1.1862 -> 1.19
    c = calculate_charges("CNC", "BUY", 10, 100.0)
    assert c["brokerage"] == 0
    assert c["stt"] == 1.0
    assert c["stamp_duty"] == 0.15
    assert c["total"] == 1.19


def test_cnc_sell_has_no_stamp_duty():
    # STT 1.00 + exchange 0.0297 + SEBI 0.001 + GST 0.0055 -> 1.0362 -> 1.04
    c = calculate_charges("CNC", "SELL", 10, 100.0)
    assert c["stamp_duty"] == 0
    assert c["total"] == 1.04


def test_mis_brokerage_is_capped_at_20():
    # turnover 1,00,000: 0.03% would be 30, capped at 20. No STT on an MIS buy.
    # 20 + stamp 3 + exchange 2.97 + SEBI 0.10 + GST 18% of 23.07 (4.1526) -> 30.22
    c = calculate_charges("MIS", "BUY", 100, 1000.0)
    assert c["brokerage"] == 20.0
    assert c["stt"] == 0
    assert c["total"] == 30.22


def test_mis_small_sell():
    # turnover 1000: brokerage 0.30, STT 0.25 (sell side), exchange 0.0297,
    # SEBI 0.001, GST 18% of 0.3307 = 0.0595 -> 0.6402 -> 0.64
    c = calculate_charges("MIS", "SELL", 10, 100.0)
    assert c["brokerage"] == pytest.approx(0.30)
    assert c["stt"] == 0.25
    assert c["total"] == 0.64
