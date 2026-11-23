import os
from dataclasses import dataclass, field

from rag.session import Message, ProviderError, ProviderTimeout


class ConfigurationError(ValueError):
    pass


@dataclass(frozen=True)
class ProviderSettings:
    provider: str
    model: str
    key: str = field(repr=False)


def load_settings(environ=None, selected=None):
    env = os.environ if environ is None else environ
    provider = (selected or env.get("RAG_PROVIDER", "gemini")).strip().lower()
    if provider not in {"gemini", "openai"}:
        raise ConfigurationError("RAG_PROVIDER deve ser gemini ou openai.")
    prefix = provider.upper()
    key_name = f"{prefix}_API_KEY"
    key = env.get(key_name, "").strip()
    if not key:
        raise ConfigurationError(f"Configure {key_name}.")
    default = "gemini-3.1-flash-lite" if provider == "gemini" else "gpt-4o-mini"
    model = env.get(f"{prefix}_MODEL", default).strip()
    if not model:
        raise ConfigurationError(f"Configure {prefix}_MODEL com um modelo válido.")
    return ProviderSettings(provider, model, key)


@dataclass
class OpenAIProvider:
    client: object
    model: str

    def generate(self, messages: tuple[Message, ...]) -> str:
        from openai import APITimeoutError

        try:
            result = self.client.chat.completions.create(
                model=self.model, temperature=0, store=False,
                messages=[{"role": message.role, "content": message.content} for message in messages],
            )
            return result.choices[0].message.content or ""
        except (APITimeoutError, TimeoutError):
            raise ProviderTimeout("Tempo de resposta excedido com OpenAI.") from None
        except Exception:
            raise ProviderError("Falha na geração com OpenAI.") from None


@dataclass
class GeminiProvider:
    client: object
    model: str

    def generate(self, messages: tuple[Message, ...]) -> str:
        from httpx import TimeoutException
        from httpx2 import TimeoutException as TimeoutExceptionV2

        contents = [
            {"role": "model" if message.role == "assistant" else "user",
             "parts": [{"text": message.content}]}
            for message in messages if message.role != "system"
        ]
        system = "\n\n".join(message.content for message in messages if message.role == "system")
        try:
            result = self.client.models.generate_content(
                model=self.model, contents=contents, config={"system_instruction": system}
            )
            return result.text or ""
        except (TimeoutException, TimeoutExceptionV2, TimeoutError):
            raise ProviderTimeout("Tempo de resposta excedido com Gemini.") from None
        except Exception:
            raise ProviderError("Falha na geração com Gemini.") from None


def make_provider(settings: ProviderSettings):
    if settings.provider == "gemini":
        from google import genai

        client = genai.Client(
            api_key=settings.key, vertexai=False, http_options={"timeout": 30_000}
        )
        return GeminiProvider(client, settings.model)

    from openai import OpenAI

    client = OpenAI(api_key=settings.key, timeout=30, max_retries=0)
    return OpenAIProvider(client, settings.model)
