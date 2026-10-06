"""
security.py - Password hashing with bcrypt. No Streamlit here, so the trading
engine and seed script can use it without loading the UI.

We never store a password itself, only a bcrypt hash such as
    "$2b$12$<22-char salt><31-char hash>"
"2b" is the bcrypt version, "12" the cost (2^12 rounds: slow on purpose, so
guessing passwords takes an attacker a very long time), and the random salt
means two users with the same password still get different hashes.

Older hashes made by this project with PBKDF2 ("pbkdf2_sha256$...") are still
accepted; auth.py upgrades them to bcrypt on the next successful login.
"""

import hashlib
import hmac

import bcrypt

BCRYPT_ROUNDS = 12   # cost factor; each +1 doubles the work
MAX_PASSWORD_BYTES = 72  # bcrypt only looks at the first 72 bytes, so we refuse longer


def hash_password(password: str) -> str:
    """Return a storable bcrypt hash for `password`."""
    if not password:
        raise ValueError("Password must not be empty")
    raw = password.encode("utf-8")
    if len(raw) > MAX_PASSWORD_BYTES:
        raise ValueError(f"Password must be at most {MAX_PASSWORD_BYTES} bytes")
    return bcrypt.hashpw(raw, bcrypt.gensalt(rounds=BCRYPT_ROUNDS)).decode("ascii")


def verify_password(password: str, stored: str) -> bool:
    """True if `password` matches the stored hash (bcrypt or legacy PBKDF2)."""
    raw = password.encode("utf-8")
    if stored.startswith("$2"):
        if len(raw) > MAX_PASSWORD_BYTES:
            return False  # could never have been stored
        # checkpw re-hashes with the salt inside `stored` and compares safely
        return bcrypt.checkpw(raw, stored.encode("ascii"))
    if stored.startswith("pbkdf2_sha256$"):
        return _verify_legacy_pbkdf2(raw, stored)
    return False  # unknown format


def needs_rehash(stored: str) -> bool:
    """True if the hash is not bcrypt with the current cost, so it should be replaced."""
    if not stored.startswith("$2"):
        return True
    cost = int(stored.split("$")[2])  # "$2b$12$..." -> "12"
    return cost != BCRYPT_ROUNDS


def _verify_legacy_pbkdf2(raw: bytes, stored: str) -> bool:
    """Check a hash in the old "pbkdf2_sha256$<iterations>$<salt>$<hash>" format."""
    try:
        _, iterations, salt_hex, digest_hex = stored.split("$")
    except ValueError:
        return False
    digest = hashlib.pbkdf2_hmac("sha256", raw, bytes.fromhex(salt_hex), int(iterations))
    # compare_digest takes the same time wherever the first difference is,
    # so timing can't reveal how close a guess was
    return hmac.compare_digest(digest.hex(), digest_hex)
