"""
LLM Handler for MedGPT
OpenRouter (primary) with streaming + graceful fallback
"""

import os
from typing import Dict, Generator, Optional
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


class LLMHandler:
    """Handles LLM calls via OpenRouter with streaming and fallback."""
    def __init__(self):
        self.backend = "fallback"
        self.model_name = "Context Extractor (fallback)"
        self.client: Optional[OpenAI] = None
        self.error = None
        # Llama 3.1 70B Instruct on OpenRouter
        self.model_id = "meta-llama/llama-3.1-70b-instruct"
        self._initialize()

    def _initialize(self):
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            self.error = "OPENROUTER_API_KEY not found in .env"
            print(f"⚠️ {self.error}")
            return

        try:
            # OpenRouter recommends these headers when possible.
            self.client = OpenAI(
                api_key=api_key,
                base_url="https://openrouter.ai/api/v1",
                default_headers={
                    "HTTP-Referer": os.getenv("APP_PUBLIC_URL", "http://localhost"),
                    "X-Title": os.getenv("APP_TITLE", "MedGPT"),
                },
            )

            # Light ping to confirm credentials
            _ = self.client.chat.completions.create(
                model=self.model_id,
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=5
            )
            self.backend = "openrouter"
            self.model_name = "Llama 3.1 70B via OpenRouter"
            self.error = None
            print("✅ OpenRouter initialized successfully")

        except Exception as e:
            self.error = f"OpenRouter initialization failed: {e}"
            print(f"❌ {self.error}")

    def _build_messages(self, query: str, context: str):
        system_prompt = (
            "You are MedGPT, a professional, evidence-based medical assistant. "
            "Use ONLY the provided context. If the answer is unknown, say so. "
            "Be concise and clear, suitable for clinicians and students."
        )
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Context:\n{context[:8000]}\n\nQuestion: {query}"},
        ]

    def generate_answer(
        self,
        query: str,
        context: str,
        max_tokens: int = 1024,
        temperature: float = 0.3,
        top_p: float = 0.9,
    ) -> str:
        """Non-streaming completion (used if streaming not desired)."""
        if self.backend == "fallback":
            return self._fallback_answer(query, context)

        try:
            messages = self._build_messages(query, context)
            response = self.client.chat.completions.create(
                model=self.model_id,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
            )
            return (response.choices[0].message.content or "").strip() or "Unable to generate response."
        except Exception as e:
            print(f"⚠️ OpenRouter error: {e}")
            return self._fallback_answer(query, context)

    def stream_answer(
        self,
        query: str,
        context: str,
        max_tokens: int = 1024,
        temperature: float = 0.3,
        top_p: float = 0.9,
    ) -> Generator[str, None, None]:
        """Streaming generator yielding tokens progressively."""
        if self.backend == "fallback":
            yield self._fallback_answer(query, context)
            return

        try:
            messages = self._build_messages(query, context)
            stream = self.client.chat.completions.create(
                model=self.model_id,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stream=True,
            )
            for chunk in stream:
                delta = chunk.choices[0].delta
                if delta and getattr(delta, "content", None):
                    yield delta.content
        except Exception as e:
            print(f"⚠️ Streaming error: {e}")
            yield self._fallback_answer(query, context)

    def _fallback_answer(self, query: str, context: str) -> str:
        """Very simple heuristic extraction from context if API is unavailable."""
        sentences = [
            (s.strip() + ".")
            for line in context.split("\n")
            if len(line.strip()) > 50
            for s in line.split(". ")
            if len(s.strip()) > 50
        ]
        q = set(query.lower().split())
        scored = [
            (sum(1 for w in q if len(w) >= 4 and w in s.lower()), s) for s in sentences
        ]
        scored = [(score, s) for score, s in scored if score > 0]
        scored.sort(reverse=True, key=lambda x: x[0])
        top = [s for _, s in scored[:4]] or sentences[:3]
        answer = " ".join(top) if top else (context[:500] or "No context available.")

        notice = (
            "\n\n---\n"
            "⚠️ **Fallback Mode Active**  \n"
            "Add/repair `OPENROUTER_API_KEY` in `.env` to enable AI responses.\n"
            f"Error: {self.error or 'API key not configured'}"
        )
        return answer + notice

    def get_status(self) -> Dict[str, str]:
        return {
            "backend": self.backend,
            "model": self.model_name,
            "ready": self.backend != "fallback",
            "error": self.error,
        }
