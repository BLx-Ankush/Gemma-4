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
MAX_TOKENS = 512
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
    with open(image_path, "rb") as file_handle:
        data = file_handle.read()
    b64_string = base64.b64encode(data).decode("utf-8")
    return f"data:{_get_mime_type(image_path)};base64,{b64_string}"


def _encode_image_for_payload(image_path: str) -> Tuple[str, str]:
    with open(image_path, "rb") as file_handle:
        image_bytes = file_handle.read()
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
    timeout_raw = os.environ.get("AEGIS_LOCAL_TIMEOUT_SEC", "25")
    try:
        timeout = float(timeout_raw)
    except ValueError:
        timeout = 25.0

    base_url = os.environ.get("AEGIS_LOCAL_LLM_URL", "http://127.0.0.1:8080").rstrip("/")
    return {
        "base_url": base_url,
        "health_path": os.environ.get("AEGIS_LOCAL_HEALTH_PATH", "/health"),
        "chat_path": os.environ.get("AEGIS_LOCAL_CHAT_PATH", "/v1/chat/completions"),
        "completion_path": os.environ.get("AEGIS_LOCAL_COMPLETION_PATH", "/completion"),
        "model_label": os.environ.get("AEGIS_LOCAL_MODEL_LABEL", "gemma4-e2b-turboquant"),
        "timeout": max(1.0, timeout),
    }


def _get_cloud_config() -> Dict[str, Any]:
    timeout_raw = os.environ.get("AEGIS_CLOUD_TIMEOUT_SEC", "12")
    try:
        timeout = float(timeout_raw)
    except ValueError:
        timeout = 12.0

    return {
        "api_url": os.environ.get(
            "AEGIS_CLOUD_API_URL",
            "https://generativelanguage.googleapis.com/v1beta/models/gemma-3-27b-it:generateContent",
        ).strip(),
        "api_key": os.environ.get("AEGIS_CLOUD_API_KEY", "").strip(),
        "model_label": os.environ.get("AEGIS_CLOUD_MODEL_LABEL", "gemma-27b-cloud").strip(),
        "timeout": max(1.0, timeout),
    }


def _cloud_is_configured() -> bool:
    config = _get_cloud_config()
    return bool(config["api_url"] and config["api_key"])


def _get_llama_health() -> bool:
    """Check local llama-server health with a 60-second cache."""
    now = time.time()
    if now - float(_llama_health_cache["checked_at"]) < 60:
        return bool(_llama_health_cache["reachable"])

    config = _get_local_config()
    result = False
    try:
        response = requests.get(
            f"{config['base_url']}{config['health_path']}",
            timeout=min(2.0, config["timeout"]),
        )
        result = response.status_code < 500
    except Exception:
        result = False

    _llama_health_cache["reachable"] = result
    _llama_health_cache["checked_at"] = now
    return result


def load_model() -> Dict[str, Any]:
    """Compatibility method: verifies local llama-server reachability.

    Local model loading is handled by the llama-server process itself.
    """
    config = _get_local_config()
    health_url = f"{config['base_url']}{config['health_path']}"

    reachable = False
    status_code = None
    try:
        response = requests.get(health_url, timeout=min(3.0, config["timeout"]))
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


def _infer_local(system_prompt: str, user_message: str, image_path: Optional[str] = None) -> str:
    global _last_backend

    config = _get_local_config()
    timeout = config["timeout"]

    chat_payload: Dict[str, Any] = {
        "model": config["model_label"],
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
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
    except Exception as exc:
        chat_error = exc

    completion_message = user_message
    if image_path and os.path.exists(image_path):
        completion_message += (
            "\n[Note: Image provided. If this endpoint does not support multimodal chat, "
            "use the user description and recommend safe triage steps.]"
        )

    completion_payload = {
        "prompt": _build_text_prompt(system_prompt, completion_message),
        "temperature": TEMPERATURE,
        "n_predict": MAX_TOKENS,
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
            "Local llama-server inference failed "
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


def _infer_cloud(system_prompt: str, user_message: str, image_path: Optional[str] = None) -> str:
    global _last_backend

    config = _get_cloud_config()
    if not (config["api_url"] and config["api_key"]):
        raise RuntimeError("Cloud inference is not configured")

    url = config["api_url"]
    if "key=" not in url:
        separator = "&" if "?" in url else "?"
        url = f"{url}{separator}key={config['api_key']}"

    message_parts: List[Dict[str, Any]] = [
        {
            "text": (
                "System instructions:\n"
                f"{system_prompt}\n\n"
                "User message:\n"
                f"{user_message}"
            )
        }
    ]

    if image_path and os.path.exists(image_path):
        mime_type, image_b64 = _encode_image_for_payload(image_path)
        message_parts.append(
            {
                "inline_data": {
                    "mime_type": mime_type,
                    "data": image_b64,
                }
            }
        )

    request_payload = {
        "contents": [
            {
                "role": "user",
                "parts": message_parts,
            }
        ],
        "generationConfig": {
            "temperature": TEMPERATURE,
            "maxOutputTokens": MAX_TOKENS,
        },
    }

    response = requests.post(
        url,
        headers={"Content-Type": "application/json"},
        data=json.dumps(request_payload),
        timeout=config["timeout"],
    )
    response.raise_for_status()

    payload = cast(Dict[str, Any], response.json())
    result_text = _extract_cloud_text(payload)
    _last_backend = "cloud"
    return result_text


def infer(system_prompt: str, user_message: str, image_path: Optional[str] = None) -> InferResult:
    start_time = time.time()
    status = compute_router.get_status()
    has_vision = bool(image_path and os.path.exists(image_path or ""))

    if status["mode"] == "online" and _cloud_is_configured():
        try:
            response_text = _infer_cloud(system_prompt, user_message, image_path)
            return InferResult(
                response=response_text.strip(),
                model_used=_get_cloud_config()["model_label"],
                mode="online",
                latency_ms=int((time.time() - start_time) * 1000),
                has_vision=has_vision,
            )
        except Exception as exc:
            print(f"Warning: cloud inference failed; falling back to local llama-server. Error: {exc}")

    if status["mode"] == "online" and not _cloud_is_configured():
        print("Warning: online mode detected but cloud API is not configured. Using local llama-server.")

    response_text = _infer_local(system_prompt, user_message, image_path)
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
        "active_mode": "cloud_27b" if status["mode"] == "online" and cloud_configured else "local_e2b",
        "local_server_url": local_config["base_url"],
        "local_chat_path": local_config["chat_path"],
        "local_completion_path": local_config["completion_path"],
        "local_model": local_config["model_label"],
        "cloud_configured": cloud_configured,
        "cloud_model": cloud_config["model_label"],
        "model_path": MODEL_PATH,
        "mmproj_path": MMPROJ_PATH,
        "has_vision": os.path.exists(MMPROJ_PATH),
        "loaded": _get_llama_health(),
        "max_tokens": MAX_TOKENS,
    }


if __name__ == "__main__":
    print("=" * 50)
    print("AEGIS Gemma Core - HTTP Engine Test")
    print("=" * 50)

    print("\n--- Local server health ---")
    print(load_model())

    print("\n--- Test query ---")
    try:
        result = infer(
            system_prompt="You are a helpful medical assistant. Respond in 2-3 sentences.",
            user_message="What should I do if I burn my hand on a hot pan?",
        )
        print(f"Response: {result.response}")
        print(f"Latency: {result.latency_ms}ms")
    except Exception as exc:
        print(f"Inference failed: {exc}")

    print(f"\nModel info: {get_model_info()}")
