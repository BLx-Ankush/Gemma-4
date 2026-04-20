from typing import Any, Dict, List, Optional, Tuple, cast
from dataclasses import dataclass
from . import compute_router
import os
import time
import base64
import json
import requests


MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
MODEL_PATH = os.path.join(MODEL_DIR, "gemma4-e2b.gguf")
MMPROJ_PATH = os.path.join(MODEL_DIR, "mmproj-gemma4-e2b.gguf")

MAX_TOKENS = int(os.environ.get("AEGIS_MAX_TOKENS", "256"))
TEMPERATURE = 0.3

_last_backend = "none"


@dataclass
class InferResult:
    response: str
    model_used: str
    mode: str
    latency_ms: int
    has_vision: bool


_llama_health_cache: Dict[str, Any] = {
    "reachable": False,
    "checked_at": 0.0,
}

_cloud_failure_state: Dict[str, Any] = {
    "consecutive_failures": 0,
    "cooldown_until": 0.0,
    "last_error": "",
}


def _get_mime_type(image_path: str) -> str:
    extension = os.path.splitext(image_path)[1].lower()
    if extension in (".jpg", ".jpeg"):
        return "image/jpeg"
    if extension == ".png":
        return "image/png"
    if extension == ".webp":
        return "image/webp"
    return "image/jpeg"


def encode_image_to_base64(image_path: str) -> str:
    with open(image_path, "rb") as fh:
        data = fh.read()
    b64 = base64.b64encode(data).decode("utf-8")
    return f"data:{_get_mime_type(image_path)};base64,{b64}"


def _encode_image_for_payload(image_path: str) -> Tuple[str, str]:
    with open(image_path, "rb") as fh:
        image_bytes = fh.read()
    return _get_mime_type(image_path), base64.b64encode(image_bytes).decode("utf-8")


def _build_text_prompt(system_prompt: str, user_message: str) -> str:
    return (
        "<start_of_turn>user\n"
        + system_prompt
        + "\n\n"
        + user_message
        + "<end_of_turn>\n"
        + "<start_of_turn>model\n"
    )


def _extract_chat_text(payload: Dict[str, Any]) -> str:
    choices = payload.get("choices", [])
    if not choices:
        raise RuntimeError("No choices in chat-completion response")
    first_choice = choices[0] if isinstance(choices[0], dict) else {}
    message = first_choice.get("message", {}) if isinstance(first_choice, dict) else {}
    content = message.get("content", "") if isinstance(message, dict) else ""
    if isinstance(content, str) and content.strip():
        return content.strip()
    if isinstance(content, list):
        joined: List[str] = []
        for item in content:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                joined.append(item["text"])
        if joined:
            return "\n".join(joined).strip()
    raise RuntimeError("Unable to parse chat-completion text")


def _extract_completion_text(payload: Dict[str, Any]) -> str:
    if isinstance(payload.get("content"), str) and payload["content"].strip():
        return str(payload["content"]).strip()
    if isinstance(payload.get("completion"), str) and payload["completion"].strip():
        return str(payload["completion"]).strip()
    choices = payload.get("choices", [])
    if choices and isinstance(choices[0], dict):
        candidate = choices[0]
        if isinstance(candidate.get("text"), str) and candidate["text"].strip():
            return str(candidate["text"]).strip()
    raise RuntimeError("Unable to parse completion text")


def _get_local_config() -> Dict[str, Any]:
    timeout_raw = os.environ.get("AEGIS_LOCAL_TIMEOUT_SEC", "35")
    try:
        timeout = float(timeout_raw)
    except ValueError:
        timeout = 35.0

    base_url = os.environ.get("AEGIS_LOCAL_LLM_URL", "http://127.0.0.1:8080").rstrip("/")
    return {
        "base_url": base_url,
        "health_path": os.environ.get("AEGIS_LOCAL_HEALTH_PATH", "/health"),
        "chat_path": os.environ.get("AEGIS_LOCAL_CHAT_PATH", "/v1/chat/completions"),
        "completion_path": os.environ.get("AEGIS_LOCAL_COMPLETION_PATH", "/completion"),
        "model_label": os.environ.get("AEGIS_LOCAL_MODEL_LABEL", "gemma4-e2b-turboquant"),
        "timeout": max(5.0, min(timeout, 180.0)),
    }


def _get_cloud_config() -> Dict[str, Any]:
    timeout_raw = os.environ.get("AEGIS_CLOUD_TIMEOUT_SEC", "30")
    try:
        timeout = float(timeout_raw)
    except ValueError:
        timeout = 30.0

    return {
        "api_url": os.environ.get(
            "AEGIS_CLOUD_API_URL",
            "https://generativelanguage.googleapis.com/v1beta/models/gemma-4-31b-it:generateContent",
        ).strip(),
        "api_key": os.environ.get("AEGIS_CLOUD_API_KEY", "").strip(),
        "model_label": os.environ.get("AEGIS_CLOUD_MODEL_LABEL", "gemma-4-31b-cloud").strip(),
        "timeout": max(1.0, timeout),
    }


def _cloud_is_configured() -> bool:
    config = _get_cloud_config()
    return bool(config["api_url"] and config["api_key"])


def _is_env_enabled(name: str, default: bool) -> bool:
    raw = os.environ.get(name, "1" if default else "0").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _safe_float_env(name: str, default: float) -> float:
    raw = os.environ.get(name, str(default)).strip()
    try:
        value = float(raw)
    except ValueError:
        value = default
    return max(0.0, value)


def _safe_int_env(name: str, default: int, minimum: int = 0) -> int:
    raw = os.environ.get(name, str(default)).strip()
    try:
        value = int(raw)
    except ValueError:
        value = default
    return max(minimum, value)


def _clamp_token_budget(max_tokens: Optional[int]) -> int:
    requested = int(max_tokens) if max_tokens else MAX_TOKENS
    hard_cap = max(128, min(_safe_int_env("AEGIS_MAX_TOKENS_HARD_CAP", 384), 768))
    return max(64, min(requested, hard_cap))


def _get_cloud_failure_policy() -> Dict[str, Any]:
    cooldown_sec = _safe_float_env("AEGIS_CLOUD_FAILURE_COOLDOWN_SEC", 25.0)
    threshold = _safe_int_env("AEGIS_CLOUD_FAILURE_THRESHOLD", 2, minimum=1)
    enabled = _is_env_enabled("AEGIS_CLOUD_COOLDOWN_ENABLED", True) and cooldown_sec > 0
    return {
        "enabled": enabled,
        "cooldown_sec": cooldown_sec,
        "threshold": threshold,
    }


def _get_cloud_cooldown_remaining_sec(now: Optional[float] = None) -> float:
    current = now if now is not None else time.time()
    remaining = float(_cloud_failure_state.get("cooldown_until", 0.0)) - current
    return max(0.0, remaining)


def _register_cloud_success() -> None:
    _cloud_failure_state["consecutive_failures"] = 0
    _cloud_failure_state["cooldown_until"] = 0.0
    _cloud_failure_state["last_error"] = ""


def _register_cloud_failure(exc: Exception) -> Dict[str, Any]:
    policy = _get_cloud_failure_policy()
    _cloud_failure_state["last_error"] = str(exc)
    _cloud_failure_state["consecutive_failures"] = int(_cloud_failure_state["consecutive_failures"]) + 1

    entered_cooldown = False
    if policy["enabled"] and _cloud_failure_state["consecutive_failures"] >= int(policy["threshold"]):
        _cloud_failure_state["cooldown_until"] = time.time() + float(policy["cooldown_sec"])
        _cloud_failure_state["consecutive_failures"] = 0
        entered_cooldown = True

    return {
        "entered_cooldown": entered_cooldown,
        "cooldown_sec": float(policy["cooldown_sec"]),
        "threshold": int(policy["threshold"]),
        "remaining_failures": int(_cloud_failure_state["consecutive_failures"]),
    }


def _get_llama_health() -> bool:
    now = time.time()
    if now - float(_llama_health_cache["checked_at"]) < 60:
        return bool(_llama_health_cache["reachable"])
    config = _get_local_config()
    result = False
    try:
        response = requests.get(
            f"{config['base_url']}{config['health_path']}",
            timeout=2.0,
        )
        result = response.status_code < 500
    except Exception:
        result = False
    _llama_health_cache["reachable"] = result
    _llama_health_cache["checked_at"] = now
    return result


def load_model() -> Dict[str, Any]:
    config = _get_local_config()
    health_url = f"{config['base_url']}{config['health_path']}"
    reachable = False
    status_code = None
    try:
        response = requests.get(health_url, timeout=3.0)
        status_code = response.status_code
        reachable = response.status_code < 500
    except Exception:
        reachable = False
    return {
        "reachable": reachable,
        "status_code": status_code,
        "health_url": health_url,
        "managed_by": "llama_server",
    }


def _infer_local(
    system_prompt: str,
    user_message: str,
    image_path: Optional[str] = None,
    max_tokens: Optional[int] = None,
) -> str:
    global _last_backend
    config = _get_local_config()
    timeout = config["timeout"]

    token_budget = _clamp_token_budget(max_tokens)

    chat_payload: Dict[str, Any] = {
        "model": config["model_label"],
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        "temperature": TEMPERATURE,
        "max_tokens": token_budget,
        "stream": False,
    }

    if image_path and os.path.exists(image_path):
        chat_payload["messages"][1] = {
            "role": "user",
            "content": [
                {"type": "text", "text": user_message},
                {"type": "image_url", "image_url": {"url": encode_image_to_base64(image_path)}},
            ],
        }

    chat_url = f"{config['base_url']}{config['chat_path']}"
    chat_error = None
    try:
        chat_response = requests.post(
            chat_url,
            headers={"Content-Type": "application/json"},
            data=json.dumps(chat_payload),
            timeout=timeout,
        )
        chat_response.raise_for_status()
        payload = cast(Dict[str, Any], chat_response.json())
        text = _extract_chat_text(payload)
        _last_backend = "local_llama_server"
        return text
    except requests.exceptions.Timeout as exc:
        raise RuntimeError(f"Local chat endpoint timed out after {timeout}s") from exc
    except Exception as exc:
        chat_error = exc

    # Fallback to /completion endpoint
    completion_message = user_message
    if image_path and os.path.exists(image_path):
        completion_message += (
            "\n[Note: Image provided. Describe what you see and give appropriate first-aid advice.]"
        )

    completion_payload = {
        "prompt": _build_text_prompt(system_prompt, completion_message),
        "temperature": TEMPERATURE,
        "n_predict": token_budget,
        "stop": ["<end_of_turn>"],
        "stream": False,
    }

    completion_url = f"{config['base_url']}{config['completion_path']}"
    try:
        completion_response = requests.post(
            completion_url,
            headers={"Content-Type": "application/json"},
            data=json.dumps(completion_payload),
            timeout=timeout,
        )
        completion_response.raise_for_status()
        payload = cast(Dict[str, Any], completion_response.json())
        text = _extract_completion_text(payload)
        _last_backend = "local_llama_server"
        return text
    except Exception as completion_error:
        raise RuntimeError(
            f"Local llama-server inference failed "
            f"(chat_error={chat_error}; completion_error={completion_error})"
        )


def _extract_cloud_text(payload: Dict[str, Any]) -> str:
    candidates = payload.get("candidates", [])
    if candidates:
        first = candidates[0] if isinstance(candidates[0], dict) else {}
        content = first.get("content", {}) if isinstance(first, dict) else {}
        parts = content.get("parts", []) if isinstance(content, dict) else []
        text_parts = []
        for part in parts:
            if isinstance(part, dict) and isinstance(part.get("text"), str):
                text_parts.append(part["text"])
        if text_parts:
            return "\n".join(text_parts).strip()
    raise RuntimeError("Cloud API response did not contain readable text.")


def _infer_cloud(
    system_prompt: str,
    user_message: str,
    image_path: Optional[str] = None,
    max_tokens: Optional[int] = None,
) -> str:
    token_budget = _clamp_token_budget(max_tokens)

    global _last_backend
    config = _get_cloud_config()
    if not (config["api_url"] and config["api_key"]):
        raise RuntimeError("Cloud inference is not configured")

    url = config["api_url"]
    if "key=" not in url:
        separator = "&" if "?" in url else "?"
        url = f"{url}{separator}key={config['api_key']}"

    message_parts: List[Dict[str, Any]] = [{"text": user_message}]

    if image_path and os.path.exists(image_path):
        mime_type, image_b64 = _encode_image_for_payload(image_path)
        message_parts.append(
            {"inline_data": {"mime_type": mime_type, "data": image_b64}}
        )

    # Prefer structured system instruction so the model is less likely to echo prompt scaffolding.
    request_payload = {
        "systemInstruction": {"parts": [{"text": system_prompt}]},
        "contents": [{"role": "user", "parts": message_parts}],
        "generationConfig": {
            "temperature": TEMPERATURE,
            "maxOutputTokens": token_budget,
        },
    }

    response = requests.post(
        url,
        headers={"Content-Type": "application/json"},
        data=json.dumps(request_payload),
        timeout=config["timeout"],
    )

    if response.status_code >= 400:
        # Compatibility fallback for backends that do not support systemInstruction.
        legacy_parts: List[Dict[str, Any]] = [
            {"text": f"{system_prompt}\n\n{user_message}"}
        ]
        if image_path and os.path.exists(image_path):
            mime_type, image_b64 = _encode_image_for_payload(image_path)
            legacy_parts.append(
                {"inline_data": {"mime_type": mime_type, "data": image_b64}}
            )
        legacy_payload = {
            "contents": [{"role": "user", "parts": legacy_parts}],
            "generationConfig": {
                "temperature": TEMPERATURE,
                "maxOutputTokens": token_budget,
            },
        }
        response = requests.post(
            url,
            headers={"Content-Type": "application/json"},
            data=json.dumps(legacy_payload),
            timeout=config["timeout"],
        )

    response.raise_for_status()
    payload = cast(Dict[str, Any], response.json())
    result_text = _extract_cloud_text(payload)
    _last_backend = "cloud"
    return result_text


def infer(
    system_prompt: str,
    user_message: str,
    image_path: Optional[str] = None,
    max_tokens: Optional[int] = None,
    force_cloud: bool = False,
) -> InferResult:
    start_time = time.time()
    effective_tokens = _clamp_token_budget(max_tokens)
    status = compute_router.get_status()
    has_vision = bool(image_path and os.path.exists(image_path or ""))
    online_cloud_mode = status["mode"] == "online" and _cloud_is_configured()
    allow_local_fallback = _is_env_enabled("AEGIS_ONLINE_FALLBACK_TO_LOCAL", True)
    cooldown_remaining_sec = _get_cloud_cooldown_remaining_sec()
    cloud_cooldown_active = cooldown_remaining_sec > 0

    if force_cloud or online_cloud_mode:
        if cloud_cooldown_active and not (force_cloud and not allow_local_fallback):
            if online_cloud_mode and not allow_local_fallback:
                raise RuntimeError(
                    "Cloud inference is in temporary cooldown after repeated failures and "
                    "online local fallback is disabled. "
                    f"Retry in about {int(round(cooldown_remaining_sec))}s or set "
                    "AEGIS_ONLINE_FALLBACK_TO_LOCAL=1."
                )
            print(
                "[gemma_core] Cloud temporarily in cooldown "
                f"({int(round(cooldown_remaining_sec))}s remaining); using local fallback."
            )
        else:
            try:
                response_text = _infer_cloud(
                    system_prompt, user_message, image_path,
                    max_tokens=effective_tokens,
                )
                _register_cloud_success()
                return InferResult(
                    response=response_text.strip(),
                    model_used=_get_cloud_config()["model_label"],
                    mode="online",
                    latency_ms=int((time.time() - start_time) * 1000),
                    has_vision=has_vision,
                )
            except Exception as exc:
                failure_state = _register_cloud_failure(exc)
                if force_cloud and not allow_local_fallback:
                    raise RuntimeError(f"Cloud inference failed in force_cloud mode: {exc}") from exc

                if online_cloud_mode and not allow_local_fallback:
                    raise RuntimeError(
                        "Cloud inference failed and online local fallback is disabled. "
                        f"Set AEGIS_ONLINE_FALLBACK_TO_LOCAL=1 to allow local fallback. Root cause: {exc}"
                    ) from exc

                if failure_state["entered_cooldown"]:
                    print(
                        "[gemma_core] Cloud failed repeatedly; entering cooldown for "
                        f"{int(round(float(failure_state['cooldown_sec'])))}s. "
                        "Using local fallback."
                    )
                else:
                    threshold = int(failure_state["threshold"])
                    remaining = max(0, threshold - int(failure_state["remaining_failures"]))
                    print(
                        f"[gemma_core] Cloud failed ({exc}), falling back to local "
                        f"({remaining} more failure(s) before cooldown)."
                    )

    if status["mode"] == "online" and not _cloud_is_configured():
        print("[gemma_core] Online but cloud not configured — using local llama-server.")

    response_text = _infer_local(
        system_prompt, user_message, image_path,
        max_tokens=effective_tokens,
    )
    return InferResult(
        response=response_text.strip(),
        model_used=_get_local_config()["model_label"],
        mode="offline",
        latency_ms=int((time.time() - start_time) * 1000),
        has_vision=has_vision,
    )


def get_model_info() -> dict:
    status = compute_router.get_status()
    cloud_config = _get_cloud_config()
    local_config = _get_local_config()
    cloud_configured = bool(cloud_config["api_url"] and cloud_config["api_key"])

    if _last_backend != "none":
        active_backend = _last_backend
    elif status["mode"] == "online" and cloud_configured:
        active_backend = "cloud"
    else:
        active_backend = "local_llama_server"

    return {
        "mode": status["mode"],
        "active_backend": active_backend,
        "active_mode": "cloud_31b" if status["mode"] == "online" and cloud_configured else "local_e2b",
        "local_server_url": local_config["base_url"],
        "local_model": local_config["model_label"],
        "cloud_configured": cloud_configured,
        "cloud_model": cloud_config["model_label"],
        "model_path": MODEL_PATH,
        "mmproj_path": MMPROJ_PATH,
        "has_vision": os.path.exists(MMPROJ_PATH),
        "loaded": _get_llama_health(),
        "max_tokens": MAX_TOKENS,
        "cloud_cooldown_remaining_sec": int(round(_get_cloud_cooldown_remaining_sec())),
        "cloud_last_error": _cloud_failure_state.get("last_error", ""),
    }
