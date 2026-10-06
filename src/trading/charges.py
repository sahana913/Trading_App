"""
charges.py - Brokerage and government charges for one fill.

Rates are a simplified version of a typical Indian discount broker for NSE
equity (they change over time; update RATES if you want current numbers).
"Turnover" = quantity x price of the fill.
"""

RATES = {
    "brokerage_mis_pct": 0.0003,    # 0.03% of turnover for intraday...
    "brokerage_mis_cap": 20.0,      # ...but never more than ₹20 per fill
    "brokerage_cnc": 0.0,           # delivery is free
    "stt_cnc_pct": 0.001,           # Securities Transaction Tax: 0.1% on buy AND sell
    "stt_mis_sell_pct": 0.00025,    # intraday: 0.025% on the sell side only
    "exchange_pct": 0.0000297,      # NSE transaction charge
    "sebi_per_crore": 10.0,         # SEBI fee: ₹10 per ₹1 crore of turnover
    "stamp_cnc_buy_pct": 0.00015,   # stamp duty is charged on BUY only
    "stamp_mis_buy_pct": 0.00003,
    "gst_pct": 0.18,                # 18% GST on brokerage + exchange + SEBI fees
}


def calculate_charges(product: str, action: str, quantity: int, price: float) -> dict:
    """Return each charge and the "total", in rupees, for one fill."""
    turnover = quantity * price
    is_buy = action == "BUY"

    if product == "MIS":
        brokerage = min(RATES["brokerage_mis_cap"], turnover * RATES["brokerage_mis_pct"])
        stt = 0.0 if is_buy else turnover * RATES["stt_mis_sell_pct"]
        stamp = turnover * RATES["stamp_mis_buy_pct"] if is_buy else 0.0
    else:  # CNC
        brokerage = RATES["brokerage_cnc"]
        stt = turnover * RATES["stt_cnc_pct"]
        stamp = turnover * RATES["stamp_cnc_buy_pct"] if is_buy else 0.0

    exchange = turnover * RATES["exchange_pct"]
    sebi = turnover * RATES["sebi_per_crore"] / 1e7  # 1 crore = 1,00,00,000
    gst = (brokerage + exchange + sebi) * RATES["gst_pct"]

    parts = {
        "brokerage": brokerage,
        "stt": stt,
        "exchange": exchange,
        "sebi": sebi,
        "stamp_duty": stamp,
        "gst": gst,
    }
    # Contract notes show paise, so round to 2 decimals
    result = {name: round(value, 2) for name, value in parts.items()}
    result["total"] = round(sum(parts.values()), 2)
    return result
