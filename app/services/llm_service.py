import logging
import asyncio
from typing import List, Dict, Any, AsyncGenerator

from google import genai
from google.genai import types

from app.core.config import settings

logger = logging.getLogger(__name__)


class LLMService:
    """
    Google GenAI SDK (google-genai) wrapper — supports both Gemini and Gemma models.

    Key compatibility note:
    - Gemini models (gemini-*): support the system_instruction field in GenerateContentConfig.
    - Gemma models (gemma-*): do NOT support system_instruction. The system rules must be
      injected as the first user+model turn in the conversation history instead.
    """

    def __init__(self, api_key: str = settings.GEMINI_API_KEY):
        self.client = genai.Client(api_key=api_key)
        self.model_id = settings.GEMINI_MODEL_NAME

        # Detect if this is a Gemma model — affects how system rules are injected.
        self.is_gemma = "gemma" in self.model_id.lower()

        # Strict grounding rules — the core anti-hallucination guardrails.
        self.system_instruction = (
            "You are a helpful and precise assistant for a Chat-with-PDF system.\n"
            "RULES (follow strictly):\n"
            "1. Answer ONLY using the provided CONTEXT text.\n"
            "2. Cite sources using [SOURCE_N] labels (e.g. 'As stated [SOURCE_1]...').\n"
            "3. If the context does not contain the answer, say EXACTLY: "
            "'The uploaded documents do not contain enough information to answer this question.'\n"
            "4. NEVER hallucinate, guess, or use external knowledge.\n"
            "5. Be concise and professional."
        )

        logger.info(f"LLMService initialized. Model: {self.model_id} | Gemma mode: {self.is_gemma}")

    def _build_config(self) -> types.GenerateContentConfig:
        """
        Build GenerateContentConfig.
        Gemma models do NOT support system_instruction — omit it for them.
        """
        if self.is_gemma:
            return types.GenerateContentConfig(temperature=0.0)
        return types.GenerateContentConfig(
            system_instruction=self.system_instruction,
            temperature=0.0,
        )

    def _prepend_system_as_turns(
        self, contents: List[types.Content]
    ) -> List[types.Content]:
        """
        For Gemma models: inject system rules as the first user+model exchange.
        This is the standard workaround since Gemma doesn't have a system role.
        The model 'acknowledges' the rules so they remain in the context window.
        """
        system_user_turn = types.Content(
            role="user",
            parts=[types.Part(text=f"SYSTEM RULES:\n{self.system_instruction}")],
        )
        system_model_ack = types.Content(
            role="model",
            parts=[types.Part(text="Understood. I will follow these rules strictly for all responses.")],
        )
        return [system_user_turn, system_model_ack] + contents

    async def get_hyde_query(self, query: str) -> str:
        """
        HyDE (Hypothetical Document Embeddings) expansion.
        Generates a hypothetical answer passage and embeds that instead of the
        raw query — dramatically improves dense retrieval recall on abstract questions.
        """
        prompt = (
            f"Write a concise paragraph that would answer the following question "
            f"if it appeared in a technical document: {query}"
        )
        try:
            response = await self.client.aio.models.generate_content(
                model=self.model_id,
                contents=prompt,
                config=self._build_config(),
            )
            return getattr(response, "text", None) or query
        except Exception as e:
            logger.warning(f"HyDE expansion failed: {e}. Falling back to original query.")
            return query

    async def generate_response(
        self,
        query: str,
        context_chunks: List[Dict[str, Any]],
        history: List[Dict[str, str]] = [],
    ) -> AsyncGenerator[str, None]:
        """
        Async streaming answer generator.

        Uses a single, flat prompt format — context + rules + question in one message.
        This is more reliable than multi-turn system injection, especially for
        smaller models like Gemma 3 1B that struggle with complex prompt structures.
        """

        # 1. Format retrieved chunks with SOURCE labels
        context_str = "\n\n".join(
            f"[SOURCE_{i + 1}] Page {c['page_num']} from '{c['filename']}':\n{c['text']}"
            for i, c in enumerate(context_chunks)
        )

        # 2. Build a simple, flat prompt that works reliably on small models.
        #    Rules are embedded directly in the user message — no system instruction,
        #    no multi-turn injection — just plain readable instructions.
        prompt = (
            f"You are a document assistant. Use ONLY the following document excerpts to answer the question.\n"
            f"If the answer is not in the excerpts, say: 'I could not find this information in the uploaded documents.'\n"
            f"Do not use any knowledge outside of what is provided below.\n"
            f"Always cite where you found the answer (e.g., '[SOURCE_1]').\n\n"
            f"--- DOCUMENT EXCERPTS ---\n{context_str}\n"
            f"--- END OF EXCERPTS ---\n\n"
            f"Question: {query}\n\n"
            f"Answer:"
        )

        # 3. For Gemini: wrap in types.Content with system instruction.
        #    For Gemma: pass as plain string — Gemma handles it as a single instruction-following turn.
        if self.is_gemma:
            # Simple string content works best for small instruction-tuned models
            contents = prompt
            config = self._build_config()
        else:
            contents = [types.Content(role="user", parts=[types.Part(text=prompt)])]
            config = self._build_config()

        # 4. Stream with exponential backoff retry for 429 errors.
        max_retries = 2
        for attempt in range(max_retries + 1):
            try:
                response_stream = await self.client.aio.models.generate_content_stream(
                    model=self.model_id,
                    contents=contents,
                    config=config,
                )
                yielded_something = False
                async for chunk in response_stream:
                    text = getattr(chunk, "text", None)
                    if text:
                        yielded_something = True
                        yield text

                if not yielded_something:
                    logger.warning("Stream yielded no text — falling back to non-streaming.")
                    response = await self.client.aio.models.generate_content(
                        model=self.model_id,
                        contents=contents,
                        config=config,
                    )
                    text = getattr(response, "text", None)
                    yield text if text else "The model returned an empty response. Please try again."
                return

            except Exception as e:
                err_str = str(e)
                is_rate_limit = "429" in err_str or "RESOURCE_EXHAUSTED" in err_str

                if is_rate_limit and attempt < max_retries:
                    wait_secs = 15 * (2 ** attempt)
                    logger.warning(f"Rate limited. Retrying in {wait_secs}s (attempt {attempt + 1}/{max_retries})")
                    await asyncio.sleep(wait_secs)
                    continue

                logger.error(f"Google GenAI error: {e}")
                if is_rate_limit:
                    yield (
                        "⚠️ **API quota exceeded.** The free tier has a daily limit. "
                        "Please wait a few hours for your quota to reset."
                    )
                else:
                    yield f"⚠️ **AI model error:** {e}"
                return



llm_service = LLMService()
