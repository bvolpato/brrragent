import socket
from ipaddress import ip_address

import pytest


@pytest.fixture(autouse=True)
def block_external_network(monkeypatch):
    original_create_connection = socket.create_connection
    original_connect = socket.socket.connect
    original_connect_ex = socket.socket.connect_ex

    def fail_network(*_args, **_kwargs):
        pytest.fail("Tests must not access external networks")

    def is_loopback(address):
        try:
            return ip_address(address[0]).is_loopback
        except (TypeError, ValueError):
            return False

    def create_connection(address, *args, **kwargs):
        if not is_loopback(address):
            return fail_network()
        return original_create_connection(address, *args, **kwargs)

    def connect(sock, address):
        if not is_loopback(address):
            return fail_network()
        return original_connect(sock, address)

    def connect_ex(sock, address):
        if not is_loopback(address):
            return fail_network()
        return original_connect_ex(sock, address)

    monkeypatch.setattr(socket, "create_connection", create_connection)
    monkeypatch.setattr(socket.socket, "connect", connect)
    monkeypatch.setattr(socket.socket, "connect_ex", connect_ex)
