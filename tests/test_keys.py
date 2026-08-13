import logging

import pytest

from brrragent.keys import KeyPool, RateLimitExhausted, pick_env_key, pick_key


def test_pick_key_trims_empty_values(monkeypatch):
    monkeypatch.setattr("brrragent.keys.random.choice", lambda values: values[0])

    assert pick_key(" first, ,second ") == "first"


def test_pick_env_key_requires_value(monkeypatch):
    monkeypatch.delenv("BRRRAGENT_TEST_KEY", raising=False)

    with pytest.raises(ValueError, match="BRRRAGENT_TEST_KEY is not set or empty"):
        pick_env_key("BRRRAGENT_TEST_KEY")


def test_key_pool_evicts_rate_limited_key_and_resets_after_exhaustion():
    pool = KeyPool(["key-value-one", "key-value-two"], name="test")

    pool.report_rate_limit("key-value-one")

    assert pool.active_count == 1
    assert pool.total_count == 2
    assert pool.acquire() == "key-value-two"

    with pytest.raises(RateLimitExhausted) as err:
        pool.report_rate_limit("key-value-two")

    assert "All 2 test keys exhausted" in str(err.value)
    assert err.value.evicted_key == "ue-two"
    assert pool.active_count == 2


def test_key_pool_logs_do_not_expose_key_fragments(caplog):
    key = "key-value-secret"
    pool = KeyPool([key], name="test")

    with caplog.at_level(logging.DEBUG), pytest.raises(RateLimitExhausted):
        pool.acquire()
        pool.report_rate_limit(key)

    assert key not in caplog.text
    assert key[-6:] not in caplog.text
