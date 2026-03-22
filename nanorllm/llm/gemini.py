import json
import os
from typing import Any
from urllib import error, request

from dotenv import load_dotenv



class GeminiLLM:
    API_TEMPLATE = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        "{model}:generateContent?key={api_key}"
    )

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gemini-3.1-flash-lite-preview",
        temperature: float = 0.2,
        max_output_tokens: int = 1024,
        top_p: float | None = None,
        timeout: int = 60,
    ):
        if load_dotenv is not None:
            load_dotenv(override=False)

        self.api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Missing Gemini API key. Set GEMINI_API_KEY (or GOOGLE_API_KEY), "
                "or pass api_key when creating GeminiLLM."
            )

        self.model = model
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.top_p = top_p
        self.timeout = timeout

    def generate(self, messages: list[dict[str, Any]]) -> str:
        url = self.API_TEMPLATE.format(model=self.model, api_key=self.api_key)
        payload = self._build_payload(messages)

        req = request.Request(
            url=url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with request.urlopen(req, timeout=self.timeout) as resp:
                raw = resp.read().decode("utf-8")
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Gemini API HTTP {exc.code}: {detail}") from exc
        except error.URLError as exc:
            raise RuntimeError(f"Gemini API request failed: {exc.reason}") from exc

        data = json.loads(raw)
        if "error" in data:
            raise RuntimeError(f"Gemini API error: {data['error']}")

        text = self._extract_text(data)
        if not text:
            raise RuntimeError(f"Gemini API returned no text: {data}")
        return text

    def _build_payload(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        system_texts: list[str] = []
        contents: list[dict[str, Any]] = []

        for message in messages or []:
            role = str(message.get("role", "user")).lower()
            text = self._to_text(message.get("content"))
            if not text:
                continue

            if role == "system":
                system_texts.append(text)
                continue

            gemini_role = "model" if role == "assistant" else "user"
            contents.append({"role": gemini_role, "parts": [{"text": text}]})

        if not contents:
            contents = [{"role": "user", "parts": [{"text": ""}]}]

        payload: dict[str, Any] = {"contents": contents}
        if system_texts:
            payload["systemInstruction"] = {"parts": [{"text": "\n\n".join(system_texts)}]}

        generation_config: dict[str, Any] = {
            "temperature": self.temperature,
            "maxOutputTokens": self.max_output_tokens,
        }
        if self.top_p is not None:
            generation_config["topP"] = self.top_p
        payload["generationConfig"] = generation_config
        return payload

    @staticmethod
    def _to_text(content: Any) -> str:
        if content is None:
            return ""
        if isinstance(content, str):
            return content

        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                    continue
                if not isinstance(item, dict):
                    continue

                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
                    continue
                if item.get("type") == "text" and isinstance(item.get("text"), str):
                    parts.append(item["text"])
            return "\n".join(part for part in parts if part)

        if isinstance(content, dict):
            text = content.get("text")
            if isinstance(text, str):
                return text
            return json.dumps(content, ensure_ascii=False)

        return str(content)

    @staticmethod
    def _extract_text(response_data: dict[str, Any]) -> str:
        candidates = response_data.get("candidates") or []
        for candidate in candidates:
            content = candidate.get("content") or {}
            parts = content.get("parts") or []
            texts = [
                part.get("text", "")
                for part in parts
                if isinstance(part, dict) and isinstance(part.get("text"), str)
            ]
            if texts:
                return "".join(texts).strip()
        return ""
