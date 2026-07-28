import base64
import io
import json
import stat

import pytest

from brrragent import auth_cli


def _jwt(payload):
    encoded = base64.urlsafe_b64encode(json.dumps(payload).encode()).decode()
    return f"header.{encoded.rstrip('=')}.signature"


class _JsonResponse:
    def __init__(self, value):
        self.body = json.dumps(value).encode()

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def read(self):
        return self.body


def _http_error(code, detail=b""):
    return auth_cli.error.HTTPError(
        auth_cli.DEVICE_TOKEN_URL,
        code,
        "request failed",
        hdrs=None,
        fp=io.BytesIO(detail),
    )


def test_device_polling_retries_pending_authorization_then_succeeds(monkeypatch):
    responses = iter(
        [
            _http_error(403),
            _JsonResponse(
                {
                    "authorization_code": "authorization",
                    "code_verifier": "verifier",
                }
            ),
        ]
    )
    requests = []

    def urlopen(req, timeout):
        requests.append((json.loads(req.data), timeout))
        response = next(responses)
        if isinstance(response, Exception):
            raise response
        return response

    sleeps = []
    monkeypatch.setattr(auth_cli.request, "urlopen", urlopen)
    monkeypatch.setattr(auth_cli.time, "monotonic", lambda: 0)
    monkeypatch.setattr(auth_cli.time, "sleep", sleeps.append)

    result = auth_cli._poll_device_authorization(
        {"device_auth_id": "device", "user_code": "USER-CODE", "interval": 2},
        timeout=30,
    )

    assert result == {
        "authorization_code": "authorization",
        "code_verifier": "verifier",
    }
    assert requests == [
        ({"device_auth_id": "device", "user_code": "USER-CODE"}, 60),
        ({"device_auth_id": "device", "user_code": "USER-CODE"}, 60),
    ]
    assert sleeps == [5]


def test_device_polling_times_out_after_pending_authorization(monkeypatch):
    monotonic_values = iter([10, 10, 16])
    sleeps = []

    def pending(*_args, **_kwargs):
        raise _http_error(404)

    monkeypatch.setattr(auth_cli.request, "urlopen", pending)
    monkeypatch.setattr(auth_cli.time, "monotonic", lambda: next(monotonic_values))
    monkeypatch.setattr(auth_cli.time, "sleep", sleeps.append)

    with pytest.raises(TimeoutError, match="timed out after 5s"):
        auth_cli._poll_device_authorization(
            {"device_auth_id": "device", "user_code": "USER-CODE", "interval": 1},
            timeout=5,
        )

    assert sleeps == [4]


def test_device_polling_reports_non_pending_http_error(monkeypatch):
    def server_error(*_args, **_kwargs):
        raise _http_error(500, b"provider unavailable")

    monkeypatch.setattr(auth_cli.request, "urlopen", server_error)

    with pytest.raises(RuntimeError, match=r"failed: HTTP 500; provider unavailable"):
        auth_cli._poll_device_authorization(
            {"device_auth_id": "device", "user_code": "USER-CODE"}, timeout=5
        )


@pytest.mark.parametrize(
    "tokens",
    [
        {
            "access_token": "access",
            "refresh_token": "refresh",
        },
        {
            "id_token": _jwt({"chatgpt_account_id": "account"}),
            "refresh_token": "refresh",
        },
        {
            "id_token": _jwt({"chatgpt_account_id": "account"}),
            "access_token": "access",
        },
        {
            "id_token": _jwt({}),
            "access_token": "access",
            "refresh_token": "refresh",
        },
    ],
    ids=["id-token", "access-token", "refresh-token", "account-id"],
)
def test_auth_documents_reject_incomplete_credentials(tokens):
    with pytest.raises(
        RuntimeError, match="Token exchange returned incomplete Codex credentials"
    ):
        auth_cli._auth_documents(tokens)


def test_two_file_write_cleans_first_stage_if_second_staging_fails(
    monkeypatch, tmp_path
):
    direct_path = tmp_path / "direct" / "auth.json"
    cli_path = tmp_path / "cli" / "auth.json"
    direct_path.parent.mkdir()
    cli_path.parent.mkdir()
    direct_path.write_text('{"version": "old-direct"}', encoding="utf-8")
    cli_path.write_text('{"version": "old-cli"}', encoding="utf-8")
    stage_auth = auth_cli._stage_auth
    calls = 0

    def fail_second_stage(path, value):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("disk full")
        return stage_auth(path, value)

    monkeypatch.setattr(auth_cli, "_stage_auth", fail_second_stage)

    with pytest.raises(OSError, match="disk full"):
        auth_cli._write_auth_documents(
            direct_path,
            cli_path,
            {"version": "new-direct"},
            {"version": "new-cli"},
        )

    assert json.loads(direct_path.read_text(encoding="utf-8")) == {
        "version": "old-direct"
    }
    assert json.loads(cli_path.read_text(encoding="utf-8")) == {"version": "old-cli"}
    assert list(tmp_path.rglob("*.pending")) == []


def test_two_file_write_rolls_back_if_second_replace_fails(monkeypatch, tmp_path):
    direct_path = tmp_path / "direct" / "auth.json"
    cli_path = tmp_path / "cli" / "auth.json"
    direct_path.parent.mkdir()
    cli_path.parent.mkdir()
    direct_path.write_text('{"version": "old-direct"}', encoding="utf-8")
    cli_path.write_text('{"version": "old-cli"}', encoding="utf-8")
    replace = auth_cli.os.replace
    calls = 0

    def fail_second_replace(source, target):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("read-only filesystem")
        replace(source, target)

    monkeypatch.setattr(auth_cli.os, "replace", fail_second_replace)

    with pytest.raises(OSError, match="read-only filesystem"):
        auth_cli._write_auth_documents(
            direct_path,
            cli_path,
            {"version": "new-direct"},
            {"version": "new-cli"},
        )

    assert json.loads(direct_path.read_text(encoding="utf-8")) == {
        "version": "old-direct"
    }
    assert json.loads(cli_path.read_text(encoding="utf-8")) == {"version": "old-cli"}
    assert list(tmp_path.rglob("*.pending")) == []
    assert list(tmp_path.rglob("*.backup")) == []


def test_two_file_write_removes_new_file_if_second_replace_fails(monkeypatch, tmp_path):
    direct_path = tmp_path / "direct" / "auth.json"
    cli_path = tmp_path / "cli" / "auth.json"
    replace = auth_cli.os.replace
    calls = 0

    def fail_second_replace(source, target):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("read-only filesystem")
        replace(source, target)

    monkeypatch.setattr(auth_cli.os, "replace", fail_second_replace)

    with pytest.raises(OSError, match="read-only filesystem"):
        auth_cli._write_auth_documents(
            direct_path,
            cli_path,
            {"version": "new-direct"},
            {"version": "new-cli"},
        )

    assert not direct_path.exists()
    assert not cli_path.exists()
    assert list(tmp_path.rglob("*.pending")) == []


def test_two_file_write_preserves_backup_if_rollback_fails(monkeypatch, tmp_path):
    direct_path = tmp_path / "direct" / "auth.json"
    cli_path = tmp_path / "cli" / "auth.json"
    direct_path.parent.mkdir()
    cli_path.parent.mkdir()
    direct_path.write_text('{"version": "old-direct"}', encoding="utf-8")
    cli_path.write_text('{"version": "old-cli"}', encoding="utf-8")
    replace = auth_cli.os.replace
    calls = 0

    def fail_replace_and_rollback(source, target):
        nonlocal calls
        calls += 1
        if calls >= 2:
            raise OSError(f"replace failure {calls}")
        replace(source, target)

    monkeypatch.setattr(auth_cli.os, "replace", fail_replace_and_rollback)

    with pytest.raises(OSError, match="replace failure 2") as error:
        auth_cli._write_auth_documents(
            direct_path,
            cli_path,
            {"version": "new-direct"},
            {"version": "new-cli"},
        )

    backups = list(direct_path.parent.glob(".auth.json.*.backup"))
    assert json.loads(direct_path.read_text(encoding="utf-8")) == {
        "version": "new-direct"
    }
    assert json.loads(cli_path.read_text(encoding="utf-8")) == {"version": "old-cli"}
    assert len(backups) == 1
    assert json.loads(backups[0].read_text(encoding="utf-8")) == {
        "version": "old-direct"
    }
    assert stat.S_IMODE(backups[0].stat().st_mode) == 0o600
    assert str(backups[0]) in "\n".join(error.value.__notes__)


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
        "idToken": tokens["id_token"],
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
