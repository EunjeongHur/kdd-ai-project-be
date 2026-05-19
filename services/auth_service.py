"""Authentication dependencies for FastAPI.

Matches docs/conventions.md §4 for Bearer JWT and Admin Token verification.

Auth fallbacks (X-User-Id header, DEFAULT_TEST_USER_ID, unsigned JWT decode,
literal-default ADMIN_TOKEN) are guarded behind `ENV` so they only activate
in development/test environments. Production deploys must set ENV=production
(or omit it entirely — production is the default for safety) and provide
real SUPABASE_JWT_SECRET / ADMIN_TOKEN values.
"""
import os
import jwt
from typing import Optional
from fastapi import Header, HTTPException, Request, Security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

security = HTTPBearer(auto_error=False)

# Default test UUID (must exist in profiles table for foreign key constraint)
# Can be customized in local dev via TEST_USER_ID environment variable.
DEFAULT_TEST_USER_ID = os.getenv("TEST_USER_ID") or "69416410-ddda-431d-816b-e5a64d1a1e7e"

# Placeholder shipped in .env templates — treat as "unset" for auth purposes
# so a copy-paste deploy doesn't accidentally grant admin access.
_ADMIN_TOKEN_PLACEHOLDER = "replace-me-with-a-strong-random-string"


def _is_dev_auth_enabled() -> bool:
    """Dynamically determine if development auth shortcuts are allowed."""
    env = os.getenv("ENV", "production").lower()
    return env in {"development", "dev", "test", "local"}


def get_current_user(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Security(security),
    x_user_id: Optional[str] = Header(None, alias="X-User-Id"),
) -> str:
    """Resolve the current user_id (UUID string).

    Order of resolution:
    1. Authorization Bearer JWT (Supabase auth token) -> extract `sub`
    2. X-User-Id header (dev only)
    3. DEFAULT_TEST_USER_ID (dev only)

    In production (DEV_AUTH_ENABLED=False), step 1 is the only path; missing
    or invalid credentials raise 401.
    """
    dev_auth_enabled = _is_dev_auth_enabled()

    # 1. Bearer Token
    if credentials and credentials.scheme.lower() == "bearer" and credentials.credentials:
        token = credentials.credentials
        try:
            secret = os.getenv("SUPABASE_JWT_SECRET")
            if secret:
                payload = jwt.decode(
                    token,
                    secret,
                    algorithms=["HS256"],
                    audience="authenticated",
                )
            elif dev_auth_enabled:
                # Local dev convenience: accept tokens without verifying signature
                # when no secret is configured. NEVER used in production.
                payload = jwt.decode(token, options={"verify_signature": False})
            else:
                raise HTTPException(
                    status_code=500,
                    detail={
                        "error": {
                            "code": "SERVER_MISCONFIGURED",
                            "message": "SUPABASE_JWT_SECRET is not set on the server.",
                        }
                    },
                )

            user_id = payload.get("sub")
            if user_id:
                return str(user_id)
        except HTTPException:
            raise
        except Exception:
            raise HTTPException(
                status_code=401,
                detail={"error": {"code": "INVALID_TOKEN", "message": "Invalid authentication token."}},
            )

    # 2. X-User-Id Header (dev only)
    if x_user_id and dev_auth_enabled:
        return x_user_id

    # 3. Default dev user (dev only)
    if dev_auth_enabled:
        default_test_user_id = os.getenv("TEST_USER_ID") or "69416410-ddda-431d-816b-e5a64d1a1e7e"
        return default_test_user_id

    raise HTTPException(
        status_code=401,
        detail={"error": {"code": "UNAUTHORIZED", "message": "Authentication required."}},
    )


def get_admin_token(x_admin_token: Optional[str] = Header(None, alias="X-Admin-Token")) -> str:
    """Verify admin access for /admin/* endpoints."""
    expected_token = os.getenv("ADMIN_TOKEN")
    dev_auth_enabled = _is_dev_auth_enabled()

    # Treat unset OR the literal placeholder as "no token configured" — fail
    # closed in production, soft-allow in dev so local testing isn't blocked.
    if not expected_token or expected_token == _ADMIN_TOKEN_PLACEHOLDER:
        if dev_auth_enabled:
            return x_admin_token or ""
        raise HTTPException(
            status_code=500,
            detail={
                "error": {
                    "code": "SERVER_MISCONFIGURED",
                    "message": "ADMIN_TOKEN is not configured on the server.",
                }
            },
        )

    if not x_admin_token or x_admin_token != expected_token:
        raise HTTPException(
            status_code=401,
            detail={"error": {"code": "INVALID_ADMIN_TOKEN", "message": "Admin token is missing or invalid."}},
        )

    return x_admin_token
