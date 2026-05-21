import os
import sys
import jwt
import base64
from dotenv import load_dotenv

load_dotenv()

secret = os.getenv("SUPABASE_JWT_SECRET")

if not secret:
    print("Error: SUPABASE_JWT_SECRET is not set in .env")
    sys.exit(1)

# Supabase secrets are base64 encoded. We must decode them first.
padded_secret = secret + "=" * (-len(secret) % 4)
secret_bytes = base64.b64decode(padded_secret)

# Use provided UUID or default to the test user UUID
user_id = sys.argv[1] if len(sys.argv) > 1 else "69416410-ddda-431d-816b-e5a64d1a1e7e"

payload = {
    "sub": user_id,
    "aud": "authenticated",
    "role": "authenticated"
}

token = jwt.encode(payload, secret_bytes, algorithm="HS256")
print(f"Generated JWT for user_id={user_id}:")
print(f"Bearer {token}")

