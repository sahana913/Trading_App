"""
Admin actions, as plain functions the admin app calls (and tests can call).

Two rules every function that CHANGES something follows:
  1. require_admin() first: the caller must be an active admin, otherwise
     PermissionError. So even code that skips the UI can't act as an admin.
  2. log_action(): every change leaves a row in admin_log (the audit trail).

    audit.py        require_admin, log_action, the audit log table
    users.py        search, view, enable/disable, top up, reset accounts
    stats.py        overview numbers, activity over time, leaderboard
    market.py       start / pause / mode / speed / volatility / step / reset
    instruments.py  add / edit / remove instruments
    mlops.py        model registry, retrain, activate a version
"""
