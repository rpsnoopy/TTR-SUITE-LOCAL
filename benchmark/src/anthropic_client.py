"""
TTR-SUITE Benchmark Suite — Anthropic API client

Stessa interfaccia pubblica di OllamaClient (metodi chat / pull_model /
stop_model / is_model_loaded / health_check) così il runner può usare
i due client in modo intercambiabile.

Richiede: pip install anthropic
Richiede: variabile d'ambiente ANTHROPIC_API_KEY
"""

from __future__ import annotations

import os
import time

from src.logger import setup_logger
from src.ollama_client import ChatResult  # riusa la stessa NamedTuple

log = setup_logger(__name__)


class AnthropicClient:
    """Wrapper attorno all'SDK Anthropic con la stessa interfaccia di OllamaClient."""

    def __init__(self, model_id: str, timeout: int = 120):
        """
        Parameters
        ----------
        model_id : ID modello Anthropic (es. ``claude-sonnet-4-6``)
        timeout  : timeout in secondi per ogni chiamata API
        """
        self.model_id = model_id
        self.timeout  = timeout
        self._client  = None   # lazy init

    # ── Public API (stessa firma di OllamaClient) ──────────────────────────────

    def chat(
        self,
        model: str,           # ignorato — usa self.model_id
        messages: list[dict],
        *,
        thinking: bool = False,   # ignorato — extended thinking non abilitato
        num_predict: int = 1024,  # ignorato — l'API Anthropic usa max_tokens fisso
    ) -> ChatResult:
        """
        Invia i messaggi all'API Anthropic e restituisce un ChatResult.

        Il parametro ``model`` è accettato per compatibilità con OllamaClient
        ma viene ignorato; si usa ``self.model_id`` configurato nel costruttore.
        """
        client = self._get_client()

        # Separa system message dagli altri
        system_text = ""
        user_messages = []
        for m in messages:
            if m.get("role") == "system":
                system_text = m.get("content", "")
            else:
                user_messages.append({"role": m["role"], "content": m["content"]})

        # Assicura che ci sia almeno un messaggio utente
        if not user_messages:
            user_messages = [{"role": "user", "content": system_text}]
            system_text = ""

        t0 = time.monotonic()
        try:
            kwargs: dict = {
                "model":      self.model_id,
                "max_tokens": 2048,
                "messages":   user_messages,
            }
            if system_text:
                kwargs["system"] = system_text

            response = client.messages.create(**kwargs)

        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"Anthropic API call failed for {self.model_id}: {exc}"
            ) from exc

        elapsed_ms       = int((time.monotonic() - t0) * 1000)
        response_text    = response.content[0].text if response.content else ""
        tokens_generated = response.usage.output_tokens
        input_tokens     = response.usage.input_tokens

        tok_s = (
            round(tokens_generated / (elapsed_ms / 1000), 2)
            if elapsed_ms > 0
            else 0.0
        )

        log.debug(
            "Anthropic [%s]: in=%d out=%d tok/s=%.1f ms=%d",
            self.model_id, input_tokens, tokens_generated, tok_s, elapsed_ms,
        )

        return ChatResult(
            response_text=response_text,
            time_ms=elapsed_ms,
            tokens_generated=tokens_generated,
            tok_s=tok_s,
            thinking_tokens=0,   # extended thinking non abilitato
        )

    def pull_model(self, model: str) -> None:
        """No-op: i modelli Anthropic non richiedono download locale."""
        log.debug("AnthropicClient.pull_model() — no-op per %s", model)

    def stop_model(self, model: str) -> None:
        """No-op: nessuna VRAM locale da liberare."""
        log.debug("AnthropicClient.stop_model() — no-op per %s", model)

    def is_model_loaded(self, model: str) -> bool:
        """Restituisce True se l'API key è configurata (modelli sempre disponibili)."""
        return bool(os.environ.get("ANTHROPIC_API_KEY", "").strip())

    def health_check(self) -> bool:
        """
        Verifica che l'SDK sia installato e che ANTHROPIC_API_KEY sia impostata.
        Non fa una chiamata di rete reale.
        """
        try:
            import anthropic  # noqa: F401
        except ImportError:
            log.warning("Package 'anthropic' non installato. Esegui: pip install anthropic")
            return False

        if not os.environ.get("ANTHROPIC_API_KEY", "").strip():
            log.warning("ANTHROPIC_API_KEY non impostata.")
            return False

        return True

    # ── Internals ──────────────────────────────────────────────────────────────

    def _get_client(self):
        """Lazy init del client Anthropic."""
        if self._client is None:
            try:
                import anthropic
            except ImportError as exc:
                raise RuntimeError(
                    "Package 'anthropic' non trovato. Esegui: pip install anthropic"
                ) from exc

            api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
            if not api_key:
                raise RuntimeError(
                    "ANTHROPIC_API_KEY non impostata. "
                    "Imposta la variabile d'ambiente prima di eseguire il benchmark."
                )
            self._client = anthropic.Anthropic(api_key=api_key)
        return self._client
