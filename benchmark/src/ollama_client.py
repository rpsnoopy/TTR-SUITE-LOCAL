"""
TTR-SUITE Benchmark Suite — Ollama REST API client

Wraps the Ollama HTTP API.  All network calls are synchronous (requests).

Qwen3 thinking mode
-------------------
When `thinking=True` the system prompt is prefixed with ``/think\\n``.
The model returns its chain-of-thought inside ``<think>…</think>`` tags.
We strip those tags from the visible response and count their tokens
separately as ``thinking_tokens``.
"""

import re
import subprocess
import time
from typing import NamedTuple

import requests

from src.logger import setup_logger

log = setup_logger(__name__)

_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)


class ChatResult(NamedTuple):
    response_text:    str
    time_ms:          int
    tokens_generated: int
    tok_s:            float
    thinking_tokens:  int   # 0 when thinking mode is off


class OllamaClient:
    """Thin wrapper around the Ollama REST API."""

    def __init__(self, base_url: str = "http://localhost:11434", timeout: int = 120):
        self.base_url = base_url.rstrip("/")
        self.timeout  = timeout

    # ── Public API ─────────────────────────────────────────────────────────────

    def chat(
        self,
        model: str,
        messages: list[dict],
        *,
        thinking: bool = False,
        num_predict: int = 1024,
    ) -> ChatResult:
        """
        Send a chat completion request and return a :class:`ChatResult`.

        Parameters
        ----------
        model    : Ollama model tag (e.g. ``qwen3:32b-q4_K_M``)
        messages : OpenAI-style list of ``{"role": ..., "content": ...}``
        thinking   : if True, prepend ``/think\\n`` to activate Qwen3 thinking mode
        num_predict: max tokens to generate (caps runaway thinking chains)
        """
        if thinking:
            messages = _inject_think_directive(messages)

        payload = {
            "model":    model,
            "messages": messages,
            "stream":   False,
            "options":  {"num_predict": num_predict},
        }

        t0 = time.monotonic()
        try:
            resp = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=self.timeout,
            )
            resp.raise_for_status()
        except requests.RequestException as exc:
            raise RuntimeError(f"Ollama chat failed for model={model}: {exc}") from exc

        elapsed_ms = int((time.monotonic() - t0) * 1000)
        data = resp.json()

        raw_text        = data.get("message", {}).get("content", "")
        tokens_generated = data.get("eval_count", 0)
        eval_duration_ns = data.get("eval_duration", 0)

        tok_s = (
            tokens_generated / (eval_duration_ns / 1e9)
            if eval_duration_ns > 0
            else 0.0
        )

        # Extract and strip <think>…</think> blocks
        thinking_text   = " ".join(_THINK_RE.findall(raw_text))
        thinking_tokens = _approx_tokens(thinking_text) if thinking_text else 0
        clean_response  = _THINK_RE.sub("", raw_text).strip()

        return ChatResult(
            response_text=clean_response,
            time_ms=elapsed_ms,
            tokens_generated=tokens_generated,
            tok_s=round(tok_s, 2),
            thinking_tokens=thinking_tokens,
        )

    def pull_model(self, model: str) -> None:
        """Run ``ollama pull <model>`` in a subprocess (blocking)."""
        log.info("Pulling model: %s", model)
        result = subprocess.run(
            ["ollama", "pull", model],
            capture_output=False,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"ollama pull failed for {model} (rc={result.returncode})")
        log.info("Model ready: %s", model)

    def stop_model(self, model: str) -> None:
        """
        Release VRAM by sending keep_alive=0 to the generate endpoint.
        Errors are logged but not raised (best-effort cleanup).
        """
        try:
            requests.post(
                f"{self.base_url}/api/generate",
                json={"model": model, "keep_alive": "0"},
                timeout=10,
            )
            log.info("Model unloaded from VRAM: %s", model)
        except Exception as exc:  # noqa: BLE001
            log.warning("Could not unload model %s: %s", model, exc)

    def is_model_loaded(self, model: str) -> bool:
        """Return True if *model* tag appears in /api/tags."""
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=10)
            resp.raise_for_status()
            tags = [m["name"] for m in resp.json().get("models", [])]
            return any(model in t for t in tags)
        except Exception:  # noqa: BLE001
            return False

    def health_check(self) -> bool:
        """Return True if Ollama is reachable."""
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return resp.status_code == 200
        except Exception:  # noqa: BLE001
            return False


# ── Helpers ────────────────────────────────────────────────────────────────────

def _inject_think_directive(messages: list[dict]) -> list[dict]:
    """Prepend ``/think\\n`` to the system message, adding one if absent."""
    msgs = [m.copy() for m in messages]
    for m in msgs:
        if m.get("role") == "system":
            m["content"] = "/think\n" + m["content"]
            return msgs
    # No system message found — insert one at the front
    msgs.insert(0, {"role": "system", "content": "/think\n"})
    return msgs


def _approx_tokens(text: str) -> int:
    """Rough token count: ~4 chars per token (GPT-style heuristic)."""
    return max(0, len(text) // 4)
