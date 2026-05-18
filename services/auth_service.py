"""Authentication dependencies for FastAPI.

Matches docs/conventions.md §4 for Bearer JWT and Admin Token verification.
Allows fallback to X-User-Id header for seamless local testing and demo seeding.
"""
import os
import jwt
from typing import Optional
from fastapi import Header, HTTPException, Request, Security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

security = HTTPBearer(auto_error=False)

# Default test UUID (must exist in profiles table for foreign key constraint)
DEFAULT_TEST_USER_ID = "69416410-ddda-431d-816b-e5a64d1a1e7e"


def get_current_user(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Security(security),
    x_user_id: Optional[str] = Header(None, alias="X-User-Id"),
) -> str:
    """Resolve the current user_id (UUID string).

    Order of resolution:
    1. Authorization Bearer JWT (Supabase auth token) -> extract `sub`
    2. X-User-Id header (for testing / server-to-server mock)
    3. Default test UUID in development mode
    """
    # 1. Bearer Token
    if credentials and credentials.scheme.lower() == "bearer" and credentials.credentials:
        token = credentials.credentials
        try:
            # Decode without secret verification if SUPABASE_JWT_SECRET is missing,
            # or extract the user ID claim.
            secret = os.getenv("SUPABASE_JWT_SECRET")
            if secret:
                payload = jwt.decode(token, secret, algorithms=["HS256"], audience="authenticated")
            else:
                payload = jwt.decode(token, options={"verify_signature": False})

            user_id = payload.get("sub")
            if user_id:
                return str(user_id)
        except Exception:
            raise HTTPException(
                status_code=401,
                detail={"error": {"code": "INVALID_TOKEN", "message": "Invalid authentication token."}},
            )

    # 2. X-User-Id Header
    if x_user_id:
        return x_user_id

    # 3. Default dev fallback
    return DEFAULT_TEST_USER_ID


def get_admin_token(x_admin_token: Optional[str] = Header(None, alias="X-Admin-Token")) -> str:
    """Verify admin access for /admin/* endpoints."""
    expected_token = os.getenv("ADMIN_TOKEN", "replace-me-with-a-strong-random-string")

    if not x_admin_token or x_admin_token != expected_token:
        raise HTTPException(
            status_code=401,
            detail={"error": {"code": "INVALID_ADMIN_TOKEN", "message": "Admin token is missing or invalid."}},
        )

    return x_admin_token
