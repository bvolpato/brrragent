import base64
import json
import stat

import pytest

from brrragent import auth_cli


def _jwt(payload):
    encoded = base64.urlsafe_b64encode(json.dumps(payload).encode()).decode()
    return f"header.{encoded.rstrip('=')}.signature"


def test_login_writes_isolated_direct_and_cli_credentials(
    monkeypatch, tmp_path, capsys
):
    direct_path = tmp_path / "direct.json"
    codex_home = tmp_path / "codex-home"
    tokens = {
        "id_token": _jwt({"chatgpt_account_id": "acct_test"}),
        "access_token": "access_test",
        "refresh_token": "refresh_test",
        "expires_in": 3600,
    }
    monkeypatch.setattr(
        auth_cli,
        "_request_device_code",
        lambda: {"device_auth_id": "device", "user_code": "ABCD-EFGH"},
    )
    monkeypatch.setattr(
        auth_cli,
        "_poll_device_authorization",
        lambda device, timeout: {
            "authorization_code": "authorization",
            "code_verifier": "verifier",
        },
    )
    monkeypatch.setattr(auth_cli, "_exchange_tokens", lambda authorization: tokens)

    result = auth_cli.main(
        [
            "login",
            "--auth-file",
            str(direct_path),
            "--codex-home",
            str(codex_home),
        ]
    )

    assert result == 0
    direct = json.loads(direct_path.read_text(encoding="utf-8"))
    cli = json.loads((codex_home / "auth.json").read_text(encoding="utf-8"))
    assert direct["openai"] == {
        "type": "oauth",
        "access": "access_test",
        "refresh": "refresh_test",
        "expires": direct["openai"]["expires"],
        "accountId": "acct_test",
    }
    assert cli["auth_mode"] == "chatgpt"
    assert cli["tokens"] == {
        "id_token": tokens["id_token"],
        "access_token": "access_test",
        "refresh_token": "refresh_test",
        "account_id": "acct_test",
    }
    assert stat.S_IMODE(direct_path.stat().st_mode) == 0o600
    assert stat.S_IMODE((codex_home / "auth.json").stat().st_mode) == 0o600
    output = capsys.readouterr().out
    assert "ABCD-EFGH" in output
    assert "access_test" not in output
    assert "refresh_test" not in output


def test_login_rejects_overlapping_auth_paths(tmp_path):
    codex_home = tmp_path / "codex-home"

    with pytest.raises(SystemExit, match="2"):
        auth_cli.main(
            [
                "login",
                "--auth-file",
                str(codex_home / "auth.json"),
                "--codex-home",
                str(codex_home),
            ]
        )


def test_login_writes_named_profile(monkeypatch, tmp_path):
    monkeypatch.setenv("BRRRAGENT_PROFILE_ROOT", str(tmp_path / "profiles"))
    tokens = {
        "id_token": _jwt({"chatgpt_account_id": "acct_profile"}),
        "access_token": "access_profile",
        "refresh_token": "refresh_profile",
    }
    monkeypatch.setattr(
        auth_cli,
        "_request_device_code",
        lambda: {"device_auth_id": "device", "user_code": "PROFILE-CODE"},
    )
    monkeypatch.setattr(
        auth_cli,
        "_poll_device_authorization",
        lambda device, timeout: {
            "authorization_code": "authorization",
            "code_verifier": "verifier",
        },
    )
    monkeypatch.setattr(auth_cli, "_exchange_tokens", lambda authorization: tokens)

    assert auth_cli.main(["login", "--profile", "content-worker"]) == 0

    profile_root = tmp_path / "profiles/content-worker"
    direct = json.loads((profile_root / "codex-auth.json").read_text())
    cli = json.loads((profile_root / "codex-home/auth.json").read_text())
    assert direct["openai"]["refresh"] == "refresh_profile"
    assert cli["tokens"]["refresh_token"] == "refresh_profile"


def test_login_rejects_invalid_profile_name():
    with pytest.raises(SystemExit, match="2"):
        auth_cli.main(["login", "--profile", "../shared"])
