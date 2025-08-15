import hashlib
import json
import os
import time
import webbrowser
from base64 import urlsafe_b64encode
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional
from uuid import uuid4

import requests

CACHE_DIR = Path.home() / ".qwen"
OAUTH_CREDS_FILE = CACHE_DIR / "oauth_creds.json"

CLIENT_ID = "f0304373b74a44d2b584a3fb70ca9e56"
SCOPE = "openid profile email model.completion"
DEVICE_CODE_URL = "https://chat.qwen.ai/api/v1/oauth2/device/code"
TOKEN_URL = "https://chat.qwen.ai/api/v1/oauth2/token"
API_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"


@dataclass
class PKCE:
    code_verifier: str
    code_challenge: str


@dataclass
class DeviceAuth:
    device_code: str
    user_code: str
    verification_uri: str
    verification_uri_complete: str
    expires_in: int


@dataclass
class OAuthCreds:
    access_token: str
    refresh_token: str
    token_type: str
    resource_url: str
    expiry_date: int


def generate_pkce_pair() -> PKCE:
    code_verifier = urlsafe_b64encode(os.urandom(32)).rstrip(b"=").decode("utf-8")
    code_challenge = (
        urlsafe_b64encode(hashlib.sha256(code_verifier.encode("utf-8")).digest())
        .rstrip(b"=")
        .decode("utf-8")
    )
    return PKCE(code_verifier=code_verifier, code_challenge=code_challenge)


class QwenClient:
    def __init__(self, model: str = "qwen3-coder-plus"):
        self._model = model
        self._oauth_creds = self._load_oauth_creds()

    def _load_oauth_creds(self) -> Optional[OAuthCreds]:
        if not OAUTH_CREDS_FILE.exists():
            return None
        with open(OAUTH_CREDS_FILE, "r") as f:
            creds_json = json.load(f)
        return OAuthCreds(**creds_json)

    def _save_oauth_creds(self, creds: OAuthCreds):
        CACHE_DIR.mkdir(exist_ok=True)
        with open(OAUTH_CREDS_FILE, "w") as f:
            json.dump(creds.__dict__, f)
        self._oauth_creds = creds

    def _is_token_expired(self) -> bool:
        if not self._oauth_creds:
            return True
        return self._oauth_creds.expiry_date < time.time() - 30

    def _perform_device_auth(self) -> DeviceAuth:
        pkce = generate_pkce_pair()
        self._pkce = pkce
        payload = {
            "client_id": CLIENT_ID,
            "scope": SCOPE,
            "code_challenge": pkce.code_challenge,
            "code_challenge_method": "S256",
        }
        headers = {
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/json",
            "x-request-id": str(uuid4()),
        }
        response = requests.post(DEVICE_CODE_URL, data=payload, headers=headers)
        response.raise_for_status()
        return DeviceAuth(**response.json())

    def _poll_for_token(self, device_auth: DeviceAuth) -> OAuthCreds:
        payload = {
            "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
            "client_id": CLIENT_ID,
            "device_code": device_auth.device_code,
            "code_verifier": self._pkce.code_verifier,
        }
        headers = {
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/json",
        }

        start_time = time.time()
        poll_interval = 2
        while time.time() - start_time < device_auth.expires_in:
            time.sleep(poll_interval)
            response = requests.post(TOKEN_URL, data=payload, headers=headers)
            data = response.json()

            if response.status_code == 200:
                expiry_date = int(time.time()) + data["expires_in"]
                creds = OAuthCreds(
                    access_token=data["access_token"],
                    refresh_token=data["refresh_token"],
                    token_type=data["token_type"],
                    resource_url=data.get(
                        "resource_url", API_BASE_URL.replace("/v1", "")
                    ),
                    expiry_date=expiry_date,
                )
                self._save_oauth_creds(creds)
                return creds

            elif data.get("error") == "authorization_pending":
                continue
            elif data.get("error") == "slow_down":
                poll_interval = min(poll_interval * 1.5, 10)
                continue
            else:
                raise Exception(f"OAuth error: {data.get('error_description')}")

        raise Exception("OAuth device flow timed out.")

    def _refresh_token(self):
        if not self._oauth_creds:
            raise Exception("No credentials to refresh.")

        payload = {
            "grant_type": "refresh_token",
            "refresh_token": self._oauth_creds.refresh_token,
            "client_id": CLIENT_ID,
        }
        headers = {
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/json",
        }
        response = requests.post(TOKEN_URL, data=payload, headers=headers)
        if response.status_code != 200:
            self._oauth_creds = None
            OAUTH_CREDS_FILE.unlink(missing_ok=True)
            raise Exception("Failed to refresh token. Please re-authenticate.")

        data = response.json()
        expiry_date = int(time.time()) + data["expires_in"]
        creds = OAuthCreds(
            access_token=data["access_token"],
            refresh_token=data.get(
                "refresh_token", self._oauth_creds.refresh_token
            ),
            token_type=data["token_type"],
            resource_url=data.get(
                "resource_url", self._oauth_creds.resource_url
            ),
            expiry_date=expiry_date,
        )
        self._save_oauth_creds(creds)

    def _ensure_valid_token(self):
        if self._is_token_expired():
            if self._oauth_creds and self._oauth_creds.refresh_token:
                try:
                    self._refresh_token()
                except Exception:
                    self._authenticate()
            else:
                self._authenticate()

    def _authenticate(self):
        device_auth = self._perform_device_auth()
        print(
            "Please visit the following URL to authorize the application:"
            f"\n{device_auth.verification_uri_complete}"
        )
        try:
            webbrowser.open(device_auth.verification_uri_complete)
        except webbrowser.Error:
            print("Could not open browser automatically.")

        self._poll_for_token(device_auth)

    def make_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        self._ensure_valid_token()
        
        resource_url = self._oauth_creds.resource_url
        if not resource_url.startswith("https://"):
            resource_url = f"https://{resource_url}"
        api_url = f"{resource_url}/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self._oauth_creds.access_token}",
            "Content-Type": "application/json",
            "User-Agent": f"QwenCodeProxy/1.0.0",
        }
        
        payload = {
            "model": self._model,
            **request_data,
        }

        stream = payload.get("stream", False)
        
        response = requests.post(api_url, json=payload, headers=headers, stream=stream)

        if response.status_code in [400, 401, 403]:
            try:
                self._refresh_token()
                headers["Authorization"] = f"Bearer {self._oauth_creds.access_token}"
                response = requests.post(api_url, json=payload, headers=headers, stream=stream)
            except Exception:
                self._authenticate()
                headers["Authorization"] = f"Bearer {self._oauth_creds.access_token}"
                response = requests.post(api_url, json=payload, headers=headers, stream=stream)

        response.raise_for_status()

        if stream:
            def stream_generator():
                for chunk in response.iter_content(chunk_size=8192):
                    yield chunk.decode("utf-8")
            return stream_generator()
        else:
            return response.json()