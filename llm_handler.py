"""
Professional LLM Handler for MedGPT
Supports: Groq (Primary), with intelligent fallback
"""

import os
from typing import Optional, Dict
from dotenv import load_dotenv

load_dotenv()


class LLMHandler:
    """Enterprise-grade LLM handler with Groq API"""
    
    def __init__(self):
        self.backend = "fallback"
        self.model_name = "Context Extractor"
        self.groq_client = None
        self.error = None
        self._initialize()
    
    def _initialize(self):
        """Initialize Groq with proper error handling"""
        groq_key = os.getenv("GROQ_API_KEY")
        
        if not groq_key:
            self.error = "GROQ_API_KEY not found in environment"
            print(f"⚠️ {self.error}")
            return
        
        try:
            from groq import Groq
            
            self.groq_client = Groq(api_key=groq_key)
            
            # Test connection with minimal token usage
            response = self.groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": "test"}],
                max_tokens=5,
                timeout=10
            )
            
            self.backend = "groq"
            self.model_name = "Llama 3.3 70B"
            self.error = None
            print("✅ Groq initialized successfully")
            
        except ImportError as e:
            self.error = f"Groq library not installed: {e}"
            print(f"❌ {self.error}")
        except Exception as e:
            self.error = f"Groq initialization failed: {str(e)}"
            print(f"❌ {self.error}")
    
    def generate_answer(self, query: str, context: str, max_tokens: int = 1024) -> str:
        """
        Generate medical answer from context
        
        Args:
            query: User's medical question
            context: Retrieved context from vectorstore
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

            response = self.groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                max_tokens=max_tokens,
                top_p=0.9,
                timeout=30
            )
            
            answer = response.choices[0].message.content.strip()
            return answer if answer else "Unable to generate response."
            
        except Exception as e:
            print(f"⚠️ Groq error: {e}")
            return self._fallback_answer(query, context)
    
    def _fallback_answer(self, query: str, context: str) -> str:
        """
        Intelligent fallback using keyword extraction
        Used when Groq is unavailable
        """
        # Split into sentences
        sentences = []
        for line in context.split('\n'):
            line = line.strip()
            if len(line) > 50:  # Filter short fragments
                for sent in line.split('. '):
                    sent = sent.strip()
                    if len(sent) > 50:
                        sentences.append(sent if sent.endswith('.') else sent + '.')
        
        # Score sentences by query relevance
        query_words = set(query.lower().split())
        scored = []
        
        for sent in sentences[:30]:
            sent_lower = sent.lower()
            # Count matching keywords (min 4 chars)
            score = sum(1 for word in query_words if len(word) >= 4 and word in sent_lower)
            if score > 0:
                scored.append((score, sent))
        
        # Sort by relevance and take top 4
        scored.sort(reverse=True, key=lambda x: x[0])
        top_sentences = [sent for _, sent in scored[:4]]
        
        if top_sentences:
            answer = ' '.join(top_sentences)
        else:
            # Return first substantial content
            answer = ' '.join(sentences[:3]) if sentences else context[:500]
        
        # Add notice
        notice = f"""

---
⚠️ **Fallback Mode Active**

This response uses keyword extraction. For AI-enhanced answers:
1. Go to Settings → Secrets in Streamlit Cloud
2. Add: `GROQ_API_KEY = "your_groq_key"`
3. Get free key at: https://console.groq.com

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
