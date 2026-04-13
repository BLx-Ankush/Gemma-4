"""
AEGIS Compute Router
Detects connectivity and device RAM. Routes inference to online (cloud) or
offline (local llama-server). Refresh calls are rate-limited to avoid hammering
the network on every request — critical for mobile/Termux environments.
"""

import os
import time

import requests

# ── Configuration ──────────────────────────────────────────────────────────────
PING_URL = os.environ.get("AEGIS_CONNECTIVITY_URL", "http://clients3.google.com/generate_204")
PING_TIMEOUT_SEC = float(os.environ.get("AEGIS_PING_TIMEOUT", "2"))
# How many seconds must pass before refresh() will actually re-check the network.
REFRESH_INTERVAL_SEC = float(os.environ.get("AEGIS_REFRESH_INTERVAL", "30"))

# ── Internal state ─────────────────────────────────────────────────────────────
_last_refresh_at: float = 0.0
_device_ram_gb: float = 0.0
_is_online: bool = False
_mode: str = "offline"


def _measure_ram() -> float:
    """Return total device RAM in GB. Falls back gracefully if psutil unavailable."""
    try:
        import psutil
        return round(psutil.virtual_memory().total / (1024 ** 3), 1)
    except Exception:
        # Termux without psutil: try /proc/meminfo (Linux/Android)
        try:
            with open("/proc/meminfo", "r") as fh:
                for line in fh:
                    if line.startswith("MemTotal:"):
                        kb = int(line.split()[1])
                        return round(kb / (1024 ** 2), 1)
        except Exception:
            pass
        return 2.0


def _check_connectivity() -> bool:
    """HEAD-request to Google captive-portal endpoint. Returns True if reachable."""
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
    """Internal: unconditionally refresh all state."""
    global _device_ram_gb, _is_online, _mode, _last_refresh_at
    _device_ram_gb = _measure_ram()
    _is_online = _check_connectivity()
    _mode = "online" if _is_online else "offline"
    _last_refresh_at = time.time()


# Initialise on import
_do_refresh()

# Public aliases (kept in sync by refresh())
DEVICE_RAM_GB: float = _device_ram_gb
IS_ONLINE: bool = _is_online
MODE: str = _mode


def refresh() -> dict:
    """
    Re-checks connectivity and RAM. Rate-limited to once per REFRESH_INTERVAL_SEC
    so that frequent callers (MIRROR audit, status polling) don't each fire a
    network request. For an immediate forced re-check use force_refresh().
    """
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
    """
    Returns the current cached state without triggering a network check.
    Use this inside tight inference loops to avoid latency spikes.
    """
    return {
        "mode": _mode,
        "ram_gb": _device_ram_gb,
        "is_online": _is_online,
        "last_checked": _last_refresh_at,
    }


def force_refresh() -> dict:
    """Bypass rate-limit and force an immediate connectivity re-check."""
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
