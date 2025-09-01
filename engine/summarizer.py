import os
import re
from typing import List, Dict, Optional


def _sentences(text: str) -> List[str]:
    # Simple multilingual-ish sentence splitter (heuristic)
    text = re.sub(r"\s+", " ", text.strip())
    # Split on . ! ? or line breaks while keeping content
    parts = re.split(r"(?<=[\.!?])\s+|\n+", text)
    out = [s.strip() for s in parts if s and not s.isspace()]
    return out


def _extractive_summary(sentences: List[str], max_sentences: int = 5) -> str:
    if not sentences:
        return ""
    # Score: length-normalized + position prior (earlier gets small boost)
    scored = []
    n = len(sentences)
    for i, s in enumerate(sentences):
        length = len(s)
        score = (min(length, 240) / 240.0) + (1.0 - (i / max(n - 1, 1))) * 0.15
        scored.append((score, i, s))
    scored.sort(key=lambda x: x[0], reverse=True)
    top = sorted(scored[:max_sentences], key=lambda x: x[1])
    return " ".join([t[2] for t in top]).strip()


class Summarizer:
    def __init__(self,
                 mode: Optional[str] = None,
                 base_url: Optional[str] = None,
                 model: Optional[str] = None,
                 api_key: Optional[str] = None):
        self.mode = (mode or os.environ.get("SERAPI_SUMMARIZER_MODE", "extractive")).lower()
        self.base_url = base_url or os.environ.get("SERAPI_LLM_BASE_URL")
        self.model = model or os.environ.get("SERAPI_LLM_MODEL")
        self.api_key = api_key or os.environ.get("SERAPI_LLM_API_KEY")

    def summarize(self, segments: List[Dict], lang: Optional[str] = None, max_chars: int = 4000) -> str:
        text = " ".join([(seg.get("text") or "").strip() for seg in segments]).strip()
        if not text:
            return ""
        text = text[:max_chars]
        if self.mode == "llm" and self.base_url and self.model and self.api_key:
            try:
                return self._summarize_llm(text, lang)
            except Exception:
                # Fallback to extractive if LLM fails
                pass
        sents = _sentences(text)
        return _extractive_summary(sents)

    def _summarize_llm(self, text: str, lang: Optional[str]) -> str:
        # OpenAI-compatible Chat Completions (e.g., LM Studio)
        from openai import OpenAI  # lazy import
        client = OpenAI(base_url=self.base_url, api_key=self.api_key)
        sys_msg = (
            "You are a neutral meeting summarizer. Write a concise, factual summary without advice."
            if (lang or "en").startswith("en") else
            "Du bist ein neutraler Protokoll‑Zusammenfasser. Schreibe eine kurze, sachliche Zusammenfassung ohne Ratschläge."
        )
        user_msg = text
        resp = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.2,
            max_tokens=300,
        )
        return (resp.choices[0].message.content or "").strip()

