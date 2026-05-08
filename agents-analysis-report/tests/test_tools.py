"""Tests for agent tools."""

from unittest.mock import patch, MagicMock

from tools import retrieve_docs, summarize_text, web_search, reset_faiss_db


class TestRetrieveDocs:
    """Tests for the document retrieval tool."""

    def setup_method(self):
        reset_faiss_db()

    @patch("tools._get_faiss_db")
    def test_returns_matching_documents(self, mock_db):
        """retrieve_docs should return concatenated document texts."""
        mock_doc1 = MagicMock()
        mock_doc1.page_content = "AI regulation in the EU is evolving."
        mock_doc2 = MagicMock()
        mock_doc2.page_content = "The US takes a different approach."
        mock_db.return_value.similarity_search.return_value = [mock_doc1, mock_doc2]

        result = retrieve_docs.invoke("AI regulation")

        assert "EU" in result
        assert "US" in result
        mock_db.return_value.similarity_search.assert_called_once()

    @patch("tools._get_faiss_db")
    def test_returns_message_when_no_docs(self, mock_db):
        """retrieve_docs should handle empty results gracefully."""
        mock_db.return_value.similarity_search.return_value = []

        result = retrieve_docs.invoke("nonexistent topic xyz")

        assert "No relevant documents" in result

    @patch("tools._get_faiss_db", side_effect=Exception("FAISS not loaded"))
    def test_handles_faiss_load_error(self, mock_db):
        """retrieve_docs should return error message if FAISS fails."""
        result = retrieve_docs.invoke("test query")

        assert "error" in result.lower()


class TestSummarizeText:
    """Tests for the summarizer tool."""

    @patch("tools.ChatAnthropic")
    def test_returns_summary(self, mock_llm_class):
        """summarize_text should return the LLM's summary."""
        mock_response = MagicMock()
        mock_response.content = "- Point 1\n- Point 2\n- Point 3"
        mock_llm_class.return_value.invoke.return_value = mock_response

        result = summarize_text.invoke("Long text about AI regulation...")

        assert "Point 1" in result
        mock_llm_class.return_value.invoke.assert_called_once()

    @patch("tools.ChatAnthropic")
    def test_truncates_long_input(self, mock_llm_class):
        """summarize_text should handle very long inputs without crashing."""
        mock_response = MagicMock()
        mock_response.content = "Summary of very long text"
        mock_llm_class.return_value.invoke.return_value = mock_response

        long_text = "word " * 10000
        result = summarize_text.invoke(long_text)

        assert "Summary" in result


class TestWebSearch:
    """Tests for the DuckDuckGo web search tool."""

    @patch("tools._get_ddg_search")
    def test_returns_search_results(self, mock_get_ddg):
        """web_search should return results from DuckDuckGo."""
        mock_ddg = MagicMock()
        mock_ddg.invoke.return_value = "AI regulation news: EU passed AI Act in 2024."
        mock_get_ddg.return_value = mock_ddg

        result = web_search.invoke("AI regulation news")

        assert "AI regulation" in result

    @patch("tools._get_ddg_search")
    def test_handles_empty_results(self, mock_get_ddg):
        """web_search should handle empty results gracefully."""
        mock_ddg = MagicMock()
        mock_ddg.invoke.return_value = ""
        mock_get_ddg.return_value = mock_ddg

        result = web_search.invoke("xyznonexistenttopic123")

        assert "No web results" in result

    @patch("tools._get_ddg_search")
    def test_handles_search_error(self, mock_get_ddg):
        """web_search should catch exceptions and return error message."""
        mock_ddg = MagicMock()
        mock_ddg.invoke.side_effect = Exception("Rate limited")
        mock_get_ddg.return_value = mock_ddg

        result = web_search.invoke("test query")

        assert "error" in result.lower()