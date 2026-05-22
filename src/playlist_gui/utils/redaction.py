"""
Secret redaction helpers for GUI and worker logging/output.
"""
from __future__ import annotations

import re
from typing import Any, Dict

_KEY_PATTERN = re.compile(r"(api[_-]?key|token|secret|password|authorization|bearer)", re.IGNORECASE)
_KEY_VALUE = re.compile(r"(?P<prefix>(api[_-]?key|token|secret|password|authorization|bearer)[\s]*[:=][\s]*)(?P<secret>[^\s,;]+)", re.IGNORECASE)
_AUTH_BEARER = re.compile(r"(Authorization[:\s]+Bearer\s+)([^\s]+)", re.IGNORECASE)
_URL_TOKEN = re.compile(r"([?&](?:api[_-]?key|token|secret|password)=)([^&]+)", re.IGNORECASE)
_CLI_FLAG = re.compile(r"(--?(?:api[_-]?key|token|secret|password)[\s:=]+)([^\s,'\";]+)", re.IGNORECASE)
_QUOTED_KEY_VALUE = re.compile(r"(?:['\"])?(api[_-]?key|token|secret|password)(?:['\"])?\s*[:=]\s*['\"]?([^\s,'\";]+)", re.IGNORECASE)


def redact_text(value: Any) -> str:
    if value is None:
        return ""

    text = str(value)
    text = _KEY_VALUE.sub(lambda m: f"{m.group('prefix')}***REDACTED***", text)
    text = re.sub(r"(password\s*[:=]\s*)(\S+)", r"\1***REDACTED***", text, flags=re.IGNORECASE)
    text = _AUTH_BEARER.sub(r"\1***REDACTED***", text)
    text = re.sub(r"(Bearer\s+)(\S+)", r"\1***REDACTED***", text, flags=re.IGNORECASE)
    text = re.sub(r"(Authorization:\s*)(.+)", r"\1***REDACTED***", text, flags=re.IGNORECASE)
    text = _URL_TOKEN.sub(r"\1***REDACTED***", text)
    text = _CLI_FLAG.sub(lambda m: f"{m.group(1)}***REDACTED***", text)
    text = _QUOTED_KEY_VALUE.sub(lambda m: f"{m.group(1)}=***REDACTED***", text)
    return text


def redact_mapping(data: Dict[str, Any]) -> Dict[str, Any]:
    redacted: Dict[str, Any] = {}
    for k, v in data.items():
        if _KEY_PATTERN.search(str(k)):
            redacted[k] = "***REDACTED***"
        elif isinstance(v, dict):
            redacted[k] = redact_mapping(v)
        else:
            redacted[k] = v
    return redacted
