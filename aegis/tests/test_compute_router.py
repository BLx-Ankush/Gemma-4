from unittest.mock import patch

import requests

from aegis.core.compute_router import (
    check_connectivity,
    check_ram,
    get_status,
    refresh,
)


def test_check_ram():
    ram_gb = check_ram()

    assert isinstance(ram_gb, float)
    assert ram_gb > 0
    assert ram_gb < 1024


def test_check_connectivity_returns_bool():
    is_online = check_connectivity()

    assert isinstance(is_online, bool)


def test_offline_mode_when_no_network():
    with patch(
        "aegis.core.compute_router.requests.head",
        side_effect=requests.exceptions.ConnectionError,
    ):
        is_online = check_connectivity()

    assert is_online is False


def test_refresh_returns_dict():
    result = refresh()

    assert isinstance(result, dict)
    assert "mode" in result
    assert result["mode"] in ("online", "offline")
    assert "ram_gb" in result
    assert isinstance(result["ram_gb"], float)
    assert "is_online" in result
    assert isinstance(result["is_online"], bool)


def test_get_status_returns_dict():
    result = get_status()

    assert isinstance(result, dict)
    assert "mode" in result
    assert result["mode"] in ("online", "offline")
    assert "ram_gb" in result
    assert isinstance(result["ram_gb"], float)
    assert "is_online" in result
    assert isinstance(result["is_online"], bool)