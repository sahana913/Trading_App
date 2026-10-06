"""
The paper-trading engine. Import everything from here:

    from src.trading import placeorder, orderbook, funds, match_orders, ...

Every function takes a SQLAlchemy session first. Functions that change data
commit their own transaction, so each call is all-or-nothing.
"""

from src.trading.accounts import create_user
from src.trading.books import funds, holdings, orderbook, positionbook, tradebook
from src.trading.market import history, quotes
from src.trading.matching import match_orders, square_off_mis
from src.trading.orders import cancelorder, modifyorder, placeorder

__all__ = [
    "create_user",
    "placeorder", "modifyorder", "cancelorder",
    "orderbook", "tradebook", "positionbook", "holdings", "funds",
    "quotes", "history",
    "match_orders", "square_off_mis",
]
