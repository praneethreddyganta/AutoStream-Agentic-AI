import os
import unittest
from unittest.mock import MagicMock, patch

from agent.llm import CustomGroqModel, get_fallback_llm


class TestLLMConfiguration(unittest.TestCase):
    def test_groq_adapter_uses_gpt_oss_20b_and_low_reasoning_effort(self):
        response = MagicMock()
        response.json.return_value = {
            "choices": [{"message": {"content": "Fallback response"}}]
        }

        with patch("agent.llm.requests.post", return_value=response) as post:
            model = CustomGroqModel(api_key="test-key")
            result = model.invoke("Hello")

        self.assertEqual(result.content, "Fallback response")
        payload = post.call_args.kwargs["json"]
        self.assertEqual(payload["model"], "openai/gpt-oss-20b")
        self.assertEqual(payload["reasoning_effort"], "low")
        self.assertEqual(payload["reasoning_format"], "hidden")
        self.assertEqual(payload["max_completion_tokens"], 1024)

    def test_gemini_is_primary_and_groq_is_called_only_after_primary_failure(self):
        response = MagicMock()
        response.json.return_value = {
            "choices": [{"message": {"content": "Groq fallback response"}}]
        }
        fake_gemini = MagicMock()
        fake_gemini.invoke.side_effect = RuntimeError("Gemini quota exceeded")

        with patch.dict(
            os.environ,
            {
                "GOOGLE_API_KEY": "test-google-key",
                "GROQ_API_KEY": "test-groq-key",
                "MODEL_NAME": "gemini-3.5-flash-lite",
                "GROQ_MODEL": "openai/gpt-oss-20b",
            },
            clear=True,
        ), patch(
            "langchain_google_genai.ChatGoogleGenerativeAI",
            return_value=fake_gemini,
        ) as gemini_constructor, patch(
            "agent.llm.requests.post",
            return_value=response,
        ) as groq_post:
            chain = get_fallback_llm()
            result = chain.invoke("Answer this question")

        gemini_constructor.assert_called_once_with(
            model="gemini-3.5-flash-lite",
            google_api_key="test-google-key",
            temperature=0,
            max_retries=0,
        )
        self.assertEqual(result.content, "Groq fallback response")
        groq_post.assert_called_once()
        self.assertEqual(
            groq_post.call_args.kwargs["json"]["model"],
            "openai/gpt-oss-20b",
        )

    def test_missing_gemini_key_fails_clearly(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, "GOOGLE_API_KEY is required"):
                get_fallback_llm()


if __name__ == "__main__":
    unittest.main()
