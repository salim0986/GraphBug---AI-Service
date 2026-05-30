"""
M12 — Service-to-service HMAC authentication helpers.

Kept in a separate module so unit tests can import these functions without
triggering the module-level database connection in api.py.
"""

from __future__ import annotations

import hashlib
import hmac
import os

# Endpoints protected by the HMAC check.
HMAC_PROTECTED_PREFIXES: tuple[str, ...] = ("/review", "/ingest")


def verify_service_hmac(body: bytes, signature: str, secret: str) -> bool:
    """
    Verify X-Service-Signature: sha256=<hex> using constant-time comparison.

    Returns True when:
    - *secret* is empty — auth is disabled (dev mode / backward compat).
    - *signature* matches the HMAC-SHA256 of *body* with *secret*.

    Returns False in all other cases.
    """
    if not secret:
        return True
    if not signature.startswith("sha256="):
        return False
    expected = "sha256=" + hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, signature)
