import os
import sys
import requests
from datetime import date, timedelta
from dotenv import load_dotenv

# Load .env variables (ACLED_EMAIL, ACLED_PASSWORD)
load_dotenv()

ACLED_OAUTH_URL = "https://acleddata.com/oauth/token"
ACLED_READ_URL = "https://acleddata.com/api/acled/read"


def get_env(name: str) -> str:
    val = os.getenv(name)
    if not val:
        raise RuntimeError(f"{name} not set (check your .env)")
    return val


def get_access_token(email: str, password: str) -> str:
    payload = {
        "grant_type": "password",
        "client_id": "acled",
        "username": email,
        "password": password,
    }

    r = requests.post(ACLED_OAUTH_URL, data=payload, timeout=20)

    if r.status_code != 200:
        raise RuntimeError(f"OAuth failed {r.status_code}: {r.text[:300]}")

    data = r.json()
    token = data.get("access_token")
    if not token:
        raise RuntimeError(f"OAuth response missing access_token: {data}")

    return token


def smoke_test_acled():
    email = get_env("ACLED_EMAIL")
    password = get_env("ACLED_PASSWORD")

    print("Requesting ACLED access token...")
    token = get_access_token(email, password)
    print("Access token received")

    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/json",
    }

    # Small recent window; widen if you get 0 events
    today = date.today()
    start = (today - timedelta(days=400)).isoformat()

    params = {
        "_format": "json",
        "event_type": "Protests",
        "event_date": f"{start}|{today.isoformat()}",
        "limit": 5,
    }

    print("Querying ACLED protest events...")
    r = requests.get(ACLED_READ_URL, headers=headers, params=params, timeout=30)

    if r.status_code != 200:
        raise RuntimeError(f"ACLED read failed {r.status_code}: {r.text[:300]}")

    data = r.json()
    events = data.get("data")
    if events is None:
        raise RuntimeError(f"Unexpected response (missing 'data'): {data}")

    print(f"Received {len(events)} events")

    if not events:
        print("Warning: no events returned in this window. Try widening the date range.")
        print("ACLED smoke test PASSED (connectivity/auth OK)")
        return

    required_fields = [
        "event_id_cnty",
        "event_date",
        "country",
        "event_type",
        "sub_event_type",
        "actor1",
        "latitude",
        "longitude",
    ]

    sample = events[0]
    missing = [f for f in required_fields if f not in sample]
    if missing:
        raise RuntimeError(f"Missing expected fields in sample event: {missing}")

    print("\nSample event:")
    for k in required_fields:
        print(f"  {k}: {sample.get(k)}")

    print("\nACLED smoke test PASSED")


if __name__ == "__main__":
    try:
        smoke_test_acled()
    except Exception as e:
        print(f"\nACLED smoke test FAILED: {e}")
        sys.exit(1)
