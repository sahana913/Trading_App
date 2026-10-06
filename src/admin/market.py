"""market.py - Admin control of the simulated market (logged wrappers around
src/trading/simulator.py)."""

from datetime import datetime

from sqlalchemy.orm import Session

from src.admin.audit import log_action, require_admin
from src.trading import simulator


def _logged(session: Session, admin_id: int, action: str, result: dict, details: dict) -> dict:
    """Write the audit row only if the action worked."""
    if result["status"] == "success":
        log_action(session, admin_id, action, details=details)
        session.commit()
    return result


def start_market(session: Session, admin_id: int, when: datetime, intraday: bool = False) -> dict:
    require_admin(session, admin_id)
    return _logged(session, admin_id, "market_start", simulator.start(session, when, intraday=intraday),
                   {"date": when.isoformat(), "intraday": intraday})


def set_running(session: Session, admin_id: int, running: bool) -> dict:
    """Start (True) or pause (False) automatic advancing."""
    require_admin(session, admin_id)
    return _logged(session, admin_id, "market_resume" if running else "market_pause",
                   simulator.update_settings(session, is_running=running), {})


def change_settings(session: Session, admin_id: int, **changes) -> dict:
    """mode / speed_seconds / volatility. Logs old -> new values."""
    require_admin(session, admin_id)
    before = simulator.settings(session)
    result = simulator.update_settings(session, **changes)
    details = {k: {"from": before[k], "to": v} for k, v in changes.items() if before.get(k) != v}
    if not details:
        return result  # nothing actually changed: no log row
    return _logged(session, admin_id, "market_settings", result, details)


def step_once(session: Session, admin_id: int, days: int = 1) -> dict:
    require_admin(session, admin_id)
    result = simulator.run(session, days)
    if result["data"]["steps"] == 0:
        return {"status": "error", "message": simulator.step(session)["message"]}
    return _logged(session, admin_id, "market_step", result,
                   {"days": result["data"]["steps"], "to": result["data"]["current_time"].isoformat()})


def reset_market(session: Session, admin_id: int) -> dict:
    """Wipe ALL users' trading and the clock (see simulator.reset)."""
    require_admin(session, admin_id)
    return _logged(session, admin_id, "market_reset", simulator.reset(session), {})
