import json
import unittest

from rag.providers import GeminiProvider, OpenAIProvider
from rag.session import Message

MESSAGES = (
    Message("system", "Use as evidências."),
    Message("user", "Primeira pergunta"),
    Message("assistant", "Primeira resposta"),
    Message("user", "Segundo turno"),
)


class SDKTransportTests(unittest.TestCase):
    def test_openai_sdk_serialization(self):
        import httpx
        from openai import OpenAI

        requests = []

        def respond(request):
            requests.append(json.loads(request.content))
            return httpx.Response(200, json={"choices": [{"message": {"content": "Resposta local"}}]})

        with httpx.Client(transport=httpx.MockTransport(respond)) as http:
            with OpenAI(api_key="fixture", http_client=http, max_retries=0) as client:
                self.assertEqual(OpenAIProvider(client, "gpt-4o-mini").generate(MESSAGES), "Resposta local")

        self.assertEqual(len(requests), 1)
        self.assertEqual(requests[0]["temperature"], 0)
        self.assertEqual([message["role"] for message in requests[0]["messages"]],
                         ["system", "user", "assistant", "user"])

    def test_gemini_sdk_serialization(self):
        import httpx
        from google import genai

        requests = []

        def respond(request):
            requests.append(json.loads(request.content))
            return httpx.Response(200, json={"candidates": [{"content": {"parts": [{"text": "Resposta local"}]}}]})

        options = {"client_args": {"transport": httpx.MockTransport(respond)}}
        with genai.Client(api_key="fixture", vertexai=False, http_options=options) as client:
            provider = GeminiProvider(client, "gemini-3.1-flash-lite")
            self.assertEqual(provider.generate(MESSAGES), "Resposta local")

        self.assertEqual(len(requests), 1)
        self.assertEqual(requests[0]["systemInstruction"]["parts"][0]["text"], "Use as evidências.")
        self.assertEqual([message["role"] for message in requests[0]["contents"]], ["user", "model", "user"])
