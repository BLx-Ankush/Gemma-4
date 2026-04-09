import psutil
import requests
import os


PING_URL = os.environ.get("AEGIS_CONNECTIVITY_URL", "http://clients3.google.com/generate_204")
PING_TIMEOUT_SEC = 2


def check_connectivity():
    try:
        response = requests.head(PING_URL, timeout=PING_TIMEOUT_SEC, allow_redirects=True)
        return response.status_code in (200, 204, 301, 302)
    except Exception:
        return False


def check_ram():
    total_bytes = psutil.virtual_memory().total
    total_gb = total_bytes / (1024 ** 3)
    return round(total_gb, 1)


DEVICE_RAM_GB = check_ram()
IS_ONLINE = check_connectivity()
MODE = "online" if IS_ONLINE else "offline"


def refresh():
    global DEVICE_RAM_GB, IS_ONLINE, MODE

    DEVICE_RAM_GB = check_ram()
    IS_ONLINE = check_connectivity()
    MODE = "online" if IS_ONLINE else "offline"

    return {
        "mode": MODE,
        "ram_gb": DEVICE_RAM_GB,
        "is_online": IS_ONLINE,
    }


def get_status():
    return {
        "mode": MODE,
        "ram_gb": DEVICE_RAM_GB,
        "is_online": IS_ONLINE,
    }


if __name__ == "__main__":
    print("AEGIS Compute Router")
    print(f"Mode: {MODE}")
    print(f"RAM: {DEVICE_RAM_GB} GB")
    print(f"Online: {IS_ONLINE}")
    result = refresh()
    print(f"After refresh: {result}")