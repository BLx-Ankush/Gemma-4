"""
AEGIS Compute Router
Detects connectivity and device RAM. Routes inference to online (cloud) or
offline (local llama-server). Refresh calls are rate-limited to avoid hammering
the network on every request in mobile/Termux environments.
"""

import os
import time

import requests


PING_URL = os.environ.get("AEGIS_CONNECTIVITY_URL", "http://clients3.google.com/generate_204")
PING_TIMEOUT_SEC = float(os.environ.get("AEGIS_PING_TIMEOUT", "2"))
REFRESH_INTERVAL_SEC = float(os.environ.get("AEGIS_REFRESH_INTERVAL", "30"))


_last_refresh_at: float = 0.0
_device_ram_gb: float = 0.0
_is_online: bool = False
_mode: str = "offline"


def _measure_ram() -> float:
    try:
        import psutil
        return round(psutil.virtual_memory().total / (1024 ** 3), 1)
    except Exception:
        try:
            with open("/proc/meminfo", "r", encoding="utf-8") as file_handle:
                for line in file_handle:
                    if line.startswith("MemTotal:"):
                        kb = int(line.split()[1])
                        return round(kb / (1024 ** 2), 1)
        except Exception:
            pass
        return 2.0


def _check_connectivity() -> bool:
    try:
        response = requests.head(
            PING_URL,
            timeout=PING_TIMEOUT_SEC,
            allow_redirects=True,
        )
        return response.status_code in (200, 204, 301, 302)
    except Exception:
        return False


def _do_refresh() -> None:
    global _device_ram_gb, _is_online, _mode, _last_refresh_at
    _device_ram_gb = _measure_ram()
    _is_online = _check_connectivity()
    _mode = "online" if _is_online else "offline"
    _last_refresh_at = time.time()


def check_connectivity() -> bool:
    return _check_connectivity()


def check_ram() -> float:
    return _measure_ram()


_do_refresh()


DEVICE_RAM_GB: float = _device_ram_gb
IS_ONLINE: bool = _is_online
MODE: str = _mode


def refresh() -> dict:
    global DEVICE_RAM_GB, IS_ONLINE, MODE

    now = time.time()
    if now - _last_refresh_at >= REFRESH_INTERVAL_SEC:
        _do_refresh()
        DEVICE_RAM_GB = _device_ram_gb
        IS_ONLINE = _is_online
        MODE = _mode

    return {
        "mode": _mode,
        "ram_gb": _device_ram_gb,
        "is_online": _is_online,
        "last_checked": _last_refresh_at,
        "refresh_interval_sec": REFRESH_INTERVAL_SEC,
    }


def get_status() -> dict:
    return {
        "mode": _mode,
        "ram_gb": _device_ram_gb,
        "is_online": _is_online,
        "last_checked": _last_refresh_at,
    }


def force_refresh() -> dict:
    _do_refresh()
    global DEVICE_RAM_GB, IS_ONLINE, MODE
    DEVICE_RAM_GB = _device_ram_gb
    IS_ONLINE = _is_online
    MODE = _mode
    return get_status()


if __name__ == "__main__":
    print("AEGIS Compute Router")
    status = get_status()
    print(f"  Mode   : {status['mode']}")
    print(f"  RAM    : {status['ram_gb']} GB")
    print(f"  Online : {status['is_online']}")
    refreshed = refresh()
    print(f"  After refresh: {refreshed}")