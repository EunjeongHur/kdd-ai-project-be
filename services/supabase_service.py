"""Supabase DB client service.

Initializes the Supabase client using environment variables.
"""
import os
from pathlib import Path
from dotenv import load_dotenv
from supabase import Client, create_client


def get_supabase() -> Client:
    """Initialize the Supabase client dynamically to guarantee fresh env vars."""
    # Ensure .env is explicitly loaded from project root
    _env_path = Path(__file__).resolve().parent.parent / ".env"
    load_dotenv(dotenv_path=_env_path, override=True)

    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_KEY")

    if not url or not key:
        raise RuntimeError(
            f"SUPABASE_URL and SUPABASE_KEY (or SERVICE_ROLE_KEY) must be set. Loaded URL: {url}, Key: {key}"
        )

    return create_client(url, key)
