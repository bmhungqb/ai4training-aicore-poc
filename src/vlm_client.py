"""Minimal OpenRouter chat-completions client with vision (image) inputs.

The API key comes from the OPENROUTER_API_KEY environment variable (or is
passed explicitly). When no key is available the client reports
`available == False`.
"""
from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request

API_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "qwen/qwen3.7-plus"

_JSON_BLOCK = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


class VlmError(RuntimeError):
    pass


class ResponseParseError(VlmError):
    """The HTTP response body itself wasn't valid JSON (truncated stream,
    proxy error page, etc.) — distinct from an API-level error or empty
    content, both of which are real replies and shouldn't be retried blindly."""


def parse_json_reply(reply: str) -> dict:
    """Extract the JSON object from a model reply (raw JSON or fenced block)."""
    reply = reply.strip()
    m = _JSON_BLOCK.search(reply)
    if m:
        reply = m.group(1)
    else:
        # tolerate leading/trailing prose around a single top-level object
        i, j = reply.find("{"), reply.rfind("}")
        if i >= 0 and j > i:
            reply = reply[i:j + 1]
    try:
        return json.loads(reply)
    except json.JSONDecodeError as e:
        raise VlmError(f"VLM reply is not valid JSON: {e}\n---\n{reply[:500]}") from e


class OpenRouterClient:
    def __init__(self, model: str = DEFAULT_MODEL, api_key: str | None = None,
                 timeout: float = 180.0, max_retries: int = 3,
                 temperature: float = 0.0):
        if api_key is None and "OPENROUTER_API_KEY" not in os.environ:
            from src.utils.env import load_dotenv
            load_dotenv()  # also honor ./.env when used programmatically
        self.model = model
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
        self.timeout = timeout
        self.max_retries = max_retries
        self.temperature = temperature
        self.n_requests = 0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_latency_s = 0.0
        self.total_cost_usd = 0.0
        self.last_usage: dict = {}
        self.last_latency_s = 0.0
        self.last_cost_usd: float | None = None

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    def _do_request(self, body: bytes) -> str:
        """One HTTP attempt: POST, parse the response, update usage/cost
        bookkeeping, and return the assistant message text (or raise)."""
        req = urllib.request.Request(
            API_URL, data=body, method="POST",
            headers={"Authorization": f"Bearer {self.api_key}",
                     "Content-Type": "application/json"})
        t0 = time.monotonic()
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            raw = resp.read()
        latency = time.monotonic() - t0
        try:
            data = json.loads(raw.decode())
        except json.JSONDecodeError as e:
            raise ResponseParseError(
                f"OpenRouter HTTP response is not valid JSON ({len(raw)} bytes): "
                f"{raw[:500]!r}") from e
        self.n_requests += 1
        self.last_latency_s = latency
        self.total_latency_s += latency
        if "error" in data:
            raise VlmError(f"OpenRouter error: {data['error']}")
        usage = data.get("usage") or {}
        self.last_usage = usage
        self.total_prompt_tokens += int(usage.get("prompt_tokens", 0) or 0)
        self.total_completion_tokens += int(usage.get("completion_tokens", 0) or 0)
        cost = usage.get("cost")
        self.last_cost_usd = float(cost) if cost is not None else None
        if cost is not None:
            self.total_cost_usd += float(cost)
        choice = data["choices"][0]
        content = choice["message"]["content"]
        if not isinstance(content, str) or not content.strip():
            # billed but empty (e.g. all output spent on hidden reasoning tokens,
            # or the model refused) -- still counts toward cost/usage above, so
            # raise (not crash) with enough detail for the caller to see why
            raise VlmError(
                f"empty content (finish_reason={choice.get('finish_reason')!r}, "
                f"native_finish_reason={choice.get('native_finish_reason')!r}, "
                f"usage={usage}); full choice: {json.dumps(choice, ensure_ascii=False)[:1000]}")
        return content

    def chat(self, messages: list[dict]) -> str:
        """POST a chat completion (with retries); returns the assistant
        message text."""
        if not self.available:
            raise VlmError("No OpenRouter API key (set OPENROUTER_API_KEY)")
        body = json.dumps({
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "usage": {"include": True},  # ask OpenRouter to report actual USD cost in usage.cost
        }).encode()
        last_err: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                return self._do_request(body)
            except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError,
                    KeyError, json.JSONDecodeError, ResponseParseError) as e:
                last_err = e
                # 4xx (except 429) won't heal on retry
                if isinstance(e, urllib.error.HTTPError) and e.code not in (429, 500, 502, 503):
                    break
                time.sleep(2.0 * (attempt + 1))
        raise VlmError(f"OpenRouter request failed after {self.max_retries} attempts: {last_err}")

    def chat_json(self, messages: list[dict]) -> dict:
        return parse_json_reply(self.chat(messages))
