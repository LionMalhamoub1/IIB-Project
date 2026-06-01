from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional, Dict

import requests


TOKEN_URL = "https://acleddata.com/oauth/token"


class ACLEDAuthError(RuntimeError):
    pass


class ACLEDAuth:

    def __init__(
        self,
        email: str,
        password: str,
        token_path: str = "data/secrets/acled_tokens.json",
        timeout_s: int = 60,
    ):
        if not email or not password:
            raise ValueError("email and password are required for OAuth token acquisition.")
        self.email = email
        self.password = password
        self.timeout_s = timeout_s
        self.token_path = Path(token_path)
        self.token_path.parent.mkdir(parents=True, exist_ok=True)

    def _load_tokens(self) -> Optional[Dict]:
        if self.token_path.exists():
            with open(self.token_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return None

    def _save_tokens(self, tokens: Dict) -> None:
        with open(self.token_path, "w", encoding="utf-8") as f:
            json.dump(tokens, f, indent=2)

    @staticmethod
    def _expired(tokens: Dict, buffer_s: int = 300) -> bool:
        # Treat the token as expired 5 minutes early to avoid using it
        # right at the boundary and hitting a 401 mid-request
        issued_at = float(tokens.get("issued_at", 0))
        expires_in = float(tokens.get("expires_in", 0))
        return time.time() > (issued_at + expires_in - buffer_s)

    def _password_grant(self) -> Dict:
        r = requests.post(
            TOKEN_URL,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            data={
                "username": self.email,
                "password": self.password,
                "grant_type": "password",
                "client_id": "acled",
            },
            timeout=self.timeout_s,
        )
        if r.status_code != 200:
            raise ACLEDAuthError(f"Password grant failed: {r.status_code} {r.text}")
        tokens = r.json()
        tokens["issued_at"] = time.time()
        self._save_tokens(tokens)
        return tokens

    def _refresh_grant(self, refresh_token: str) -> Dict:
        r = requests.post(
            TOKEN_URL,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            data={
                "refresh_token": refresh_token,
                "grant_type": "refresh_token",
                "client_id": "acled",
            },
            timeout=self.timeout_s,
        )
        if r.status_code != 200:
            raise ACLEDAuthError(f"Refresh grant failed: {r.status_code} {r.text}")
        tokens = r.json()
        tokens["issued_at"] = time.time()
        self._save_tokens(tokens)
        return tokens

    def get_access_token(self) -> str:
        tokens = self._load_tokens()

        if tokens is None:
            tokens = self._password_grant()
        elif self._expired(tokens):
            try:
                tokens = self._refresh_grant(tokens["refresh_token"])
            except Exception:
                tokens = self._password_grant()

        return tokens["access_token"]
