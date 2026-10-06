"""
security.py - Password hashing using only Python's standard library.

We never store a password itself, only a salted PBKDF2 hash:
    "pbkdf2_sha256$<iterations>$<salt hex>$<hash hex>"
The random salt means two users with the same password get different hashes,
and the many iterations make guessing passwords slow for an attacker.
"""

import hashlib
import hmac
import secrets

ALGORITHM = "pbkdf2_sha256"
ITERATIONS = 600_000  # OWASP's recommended minimum for PBKDF2-SHA256


def hash_password(password: str) -> str:
    """Return a storable hash string for `password`."""
    if not password:
        raise ValueError("Password must not be empty")
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, ITERATIONS)
    return f"{ALGORITHM}${ITERATIONS}${salt.hex()}${digest.hex()}"


def verify_password(password: str, stored: str) -> bool:
    """True if `password` matches the stored hash string."""
    try:
        algorithm, iterations, salt_hex, digest_hex = stored.split("$")
    except ValueError:
        return False  # not a hash we created
    if algorithm != ALGORITHM:
        return False
    digest = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), bytes.fromhex(salt_hex), int(iterations)
    )
    # compare_digest takes the same time whether the first byte or the last
    # byte differs, so timing cannot leak how close a guess was
    return hmac.compare_digest(digest.hex(), digest_hex)
