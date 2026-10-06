"""audit.py - Who may act as admin, and the record of everything they did."""

from datetime import datetime

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session, aliased

from src.db.models import AdminLog, User


def require_admin(session: Session, admin_id: int) -> User:
    """The acting user, if they are an ACTIVE admin. Otherwise PermissionError."""
    user = session.get(User, admin_id)
    if user is None or user.role != "admin" or not user.is_active:
        raise PermissionError("Only active admin accounts can do this")
    return user


def log_action(session: Session, admin_id: int, action: str, target_user_id: int | None = None,
               details: dict | None = None) -> AdminLog:
    """Add an audit row. The caller commits it together with the change itself,
    so a change and its log entry are saved together or not at all."""
    row = AdminLog(admin_user_id=admin_id, action=action, target_user_id=target_user_id,
                   details=details or {}, created_at=datetime.now())
    session.add(row)
    return row


def audit_frame(session: Session, action: str | None = None, limit: int | None = None) -> pd.DataFrame:
    """The audit log, newest first, with usernames instead of ids."""
    admin, target = aliased(User), aliased(User)
    query = (select(AdminLog.created_at, admin.username, AdminLog.action, target.username, AdminLog.details)
             .join(admin, AdminLog.admin_user_id == admin.id)
             .outerjoin(target, AdminLog.target_user_id == target.id)
             .order_by(AdminLog.created_at.desc(), AdminLog.id.desc()))
    if action:
        query = query.where(AdminLog.action == action)
    if limit:
        query = query.limit(limit)
    return pd.DataFrame(session.execute(query).all(),
                        columns=["time", "admin", "action", "target_user", "details"])
