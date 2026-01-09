"""
Professional LLM Handler for MedGPT
Supports: OpenRouter with Llama 3.1 70B
"""

import os
import requests
import json
from typing import Optional, Dict
from dotenv import load_dotenv

load_dotenv()


class LLMHandler:
    """Enterprise-grade LLM handler with OpenRouter API"""
    
    def __init__(self):
        self.backend = "fallback"
        self.model_name = "Context Extractor"
        self.api_key = None
        self.error = None
        self._initialize()
    
    def _initialize(self):
        """Initialize OpenRouter with proper error handling"""
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        
        if not self.api_key:
            self.error = "OPENROUTER_API_KEY not found in environment"
            print(f"⚠️ {self.error}")
            return
        
        try:
            # Test connection
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "meta-llama/llama-3.1-70b-instruct",
                    "messages": [{"role": "user", "content": "test"}],
                    "max_tokens": 5
                },
                timeout=10
            )
            
            if response.status_code == 200:
                self.backend = "openrouter"
                self.model_name = "Llama 3.1 70B"
                self.error = None
                print("✅ OpenRouter initialized successfully")
            else:
                raise Exception(f"API returned status {response.status_code}")
            
        except Exception as e:
            self.error = f"OpenRouter initialization failed: {str(e)}"
            print(f"❌ {self.error}")
    
    def generate_answer(self, query: str, context: str, temperature: float = 0.3,
                       top_p: float = 0.9, max_tokens: int = 1024) -> str:
        """
        Generate medical answer from context
        
        Args:
            query: User's medical question
            context: Retrieved context from vectorstore
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            max_tokens: Maximum response length
            
        Returns:
            Generated answer
        """
        if self.backend == "fallback":
            return self._fallback_answer(query, context)
        
        try:
            system_prompt = """You are MedGPT, a professional medical knowledge assistant.

Your responsibilities:
- Provide accurate, evidence-based medical information
- Base all answers STRICTLY on the provided context
- Use clear, professional medical terminology
- Structure responses with proper paragraphs
- Include relevant clinical details (dosages, protocols, criteria)
- Cite specific information from the context
- If context is insufficient, clearly state limitations

CRITICAL: Never add information not present in the provided context."""

            user_prompt = f"""Medical Context:
{context[:4000]}

Clinical Question: {query}

Provide a comprehensive, evidence-based answer based ONLY on the context above."""

            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "meta-llama/llama-3.1-70b-instruct",
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "top_p": top_p
                },
                timeout=30
            )
            
            if response.status_code == 200:
                answer = response.json()["choices"][0]["message"]["content"].strip()
                return answer if answer else "Unable to generate response."
            else:
                raise Exception(f"API error: {response.status_code}")
            
        except Exception as e:
            print(f"⚠️ OpenRouter error: {e}")
            return self._fallback_answer(query, context)
    
    def stream_answer(self, query: str, context: str, temperature: float = 0.3, 
                     top_p: float = 0.9, max_tokens: int = 1024):
        """
        Stream the answer from OpenRouter
        
        Args:
            query: User's medical question
            context: Retrieved context from vectorstore
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            max_tokens: Maximum response length
            
        Yields:
            Tokens as they arrive
        """
        if self.backend == "fallback":
            # Fallback doesn't stream - just yield the whole answer
            yield self._fallback_answer(query, context)
            return
        
        try:
            system_prompt = """You are MedGPT, a professional medical knowledge assistant.

Your responsibilities:
- Provide accurate, evidence-based medical information
- Base all answers STRICTLY on the provided context
- Use clear, professional medical terminology
- Structure responses with proper paragraphs
- Include relevant clinical details (dosages, protocols, criteria)
- Cite specific information from the context
- If context is insufficient, clearly state limitations

CRITICAL: Never add information not present in the provided context."""

            user_prompt = f"""Medical Context:
{context[:4000]}

Clinical Question: {query}

Provide a comprehensive, evidence-based answer based ONLY on the context above."""

            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "meta-llama/llama-3.1-70b-instruct",
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "top_p": top_p,
                    "stream": True
                },
                timeout=60,
                stream=True
            )
            
            for line in response.iter_lines():
                if line:
                    line_text = line.decode('utf-8')
                    if line_text.startswith('data: '):
                        if line_text.strip() == 'data: [DONE]':
                            break
                        try:
                            data = json.loads(line_text[6:])
                            if 'choices' in data and len(data['choices']) > 0:
                                delta = data['choices'][0].get('delta', {})
                                content = delta.get('content', '')
                                if content:
                                    yield content
                        except json.JSONDecodeError:
                            continue
            
        except Exception as e:
            print(f"⚠️ OpenRouter streaming error: {e}")
            yield self._fallback_answer(query, context)
    
    def _fallback_answer(self, query: str, context: str) -> str:
        """
        Intelligent fallback using keyword extraction
        Used when OpenRouter is unavailable
        """
        sentences = []
        for line in context.split('\n'):
            line = line.strip()
            if len(line) > 50:
                for sent in line.split('. '):
                    sent = sent.strip()
                    if len(sent) > 50:
                        sentences.append(sent if sent.endswith('.') else sent + '.')
        
        query_words = set(query.lower().split())
        scored = []
        
        for sent in sentences[:30]:
            sent_lower = sent.lower()
            score = sum(1 for word in query_words if len(word) >= 4 and word in sent_lower)
            if score > 0:
                scored.append((score, sent))
        
        scored.sort(reverse=True, key=lambda x: x[0])
        top_sentences = [sent for _, sent in scored[:4]]
        
        if top_sentences:
            answer = ' '.join(top_sentences)
        else:
            answer = ' '.join(sentences[:3]) if sentences else context[:500]
        
        notice = f"""

---
⚠️ **Fallback Mode Active**

This response uses keyword extraction. For AI-enhanced answers:
1. Go to Settings → Secrets in Streamlit Cloud
2. Add: `OPENROUTER_API_KEY = "your_key"`
3. Get free key at: https://openrouter.ai

Error: {self.error or 'API key not configured'}
"""
        
        return answer + notice
    
    def get_status(self) -> Dict[str, str]:
        """Get current backend status"""
        return {
            "backend": self.backend,
            "model": self.model_name,
            "ready": self.backend != "fallback",
            "error": self.error
        }
