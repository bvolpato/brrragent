import socket

import pytest


@pytest.fixture(autouse=True)
def block_external_network(monkeypatch):
    def fail_network(*_args, **_kwargs):
        pytest.fail("Tests must not access external networks")

    monkeypatch.setattr(socket, "create_connection", fail_network)
    monkeypatch.setattr(socket.socket, "connect", fail_network)
    monkeypatch.setattr(socket.socket, "connect_ex", fail_network)
