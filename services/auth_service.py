"""Authentication dependencies for FastAPI."""
import os
import jwt
from jwt import PyJWKClient
from typing import Optional
from fastapi import Header, HTTPException, Request, Security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

security = HTTPBearer(auto_error=False)

DEFAULT_TEST_USER_ID = "69416410-ddda-431d-816b-e5a64d1a1e7e"
_ADMIN_TOKEN_PLACEHOLDER = "replace-me-with-a-strong-random-string"

ENV = os.getenv("ENV", "production").lower()
DEV_AUTH_ENABLED = ENV in {"development", "dev", "test", "local"}

_supabase_url = os.getenv("SUPABASE_URL", "")
_jwks_client = PyJWKClient(f"{_supabase_url}/auth/v1/.well-known/jwks.json") if _supabase_url else None


def get_current_user(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Security(security),
    x_user_id: Optional[str] = Header(None, alias="X-User-Id"),
) -> str:
    if credentials and credentials.scheme.lower() == "bearer" and credentials.credentials:
        token = credentials.credentials
        try:
            if _jwks_client:
                signing_key = _jwks_client.get_signing_key_from_jwt(token)
                payload = jwt.decode(
                    token,
                    signing_key.key,
                    algorithms=["ES256", "HS256"],
                    audience="authenticated",
                )
            elif DEV_AUTH_ENABLED:
                payload = jwt.decode(token, options={"verify_signature": False})
            else:
                raise HTTPException(
                    status_code=500,
                    detail={"error": {"code": "SERVER_MISCONFIGURED", "message": "SUPABASE_URL is not set."}},
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

    if x_user_id and DEV_AUTH_ENABLED:
        return x_user_id
    if DEV_AUTH_ENABLED:
        return DEFAULT_TEST_USER_ID

    raise HTTPException(
        status_code=401,
        detail={"error": {"code": "UNAUTHORIZED", "message": "Authentication required."}},
    )


def get_admin_token(x_admin_token: Optional[str] = Header(None, alias="X-Admin-Token")) -> str:
    expected_token = os.getenv("ADMIN_TOKEN")

    if not expected_token or expected_token == _ADMIN_TOKEN_PLACEHOLDER:
        if DEV_AUTH_ENABLED:
            return x_admin_token or ""
        raise HTTPException(
            status_code=500,
            detail={"error": {"code": "SERVER_MISCONFIGURED", "message": "ADMIN_TOKEN is not configured."}},
        )

    if not x_admin_token or x_admin_token != expected_token:
        raise HTTPException(
            status_code=401,
            detail={"error": {"code": "INVALID_ADMIN_TOKEN", "message": "Admin token is missing or invalid."}},
        )

    return x_admin_token