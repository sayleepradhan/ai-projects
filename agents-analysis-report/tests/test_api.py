"""Tests for the FastAPI endpoints."""

from unittest.mock import patch
from fastapi.testclient import TestClient

from agent import ReportResult, AgentStep
from api import app


client = TestClient(app)


class TestHealthEndpoint:
    """Tests for GET /api/health."""

    def test_health_returns_ok(self):
        response = client.get("/api/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"


class TestSyncReportEndpoint:
    """Tests for POST /api/report."""

    @patch("api.generate_report")
    def test_valid_request_returns_report(self, mock_gen):
        """A valid topic should return a structured report."""
        mock_gen.return_value = ReportResult(
            topic="AI overview",
            plan=[
                AgentStep(description="Research AI", result="Found info", status="done"),
            ],
            report="# AI Report\nContent here.",
        )

        response = client.post(
            "/api/report",
            json={"topic": "AI overview"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["topic"] == "AI overview"
        assert len(data["plan"]) == 1
        assert "AI Report" in data["report"]

    def test_empty_topic_returns_422(self):
        """A topic shorter than 5 chars should fail validation."""
        response = client.post("/api/report", json={"topic": "hi"})
        assert response.status_code == 422

    def test_missing_topic_returns_422(self):
        """Missing topic field should fail validation."""
        response = client.post("/api/report", json={})
        assert response.status_code == 422

    @patch("api.generate_report")
    def test_agent_error_returns_500(self, mock_gen):
        """If the agent fails entirely, return 500."""
        mock_gen.return_value = ReportResult(
            topic="bad topic",
            error="Planning failed: LLM down",
        )

        response = client.post(
            "/api/report",
            json={"topic": "bad topic that fails"},
        )

        assert response.status_code == 500


class TestAsyncReportEndpoint:
    """Tests for POST /api/report/async and GET /api/report/{task_id}."""

    @patch("api.generate_report")
    def test_async_returns_task_id(self, mock_gen):
        """POST /api/report/async should return a task_id."""
        mock_gen.return_value = ReportResult(topic="test", report="done")

        response = client.post(
            "/api/report/async",
            json={"topic": "async test topic"},
        )

        assert response.status_code == 200
        data = response.json()
        assert "task_id" in data
        assert data["status"] == "pending"

    def test_get_nonexistent_task_returns_404(self):
        """GET with a fake task_id should return 404."""
        response = client.get("/api/report/fake-id-12345")
        assert response.status_code == 404


class TestRequestValidation:
    """Edge case tests for request validation."""

    def test_topic_too_long_returns_422(self):
        """A topic exceeding 500 chars should fail validation."""
        response = client.post(
            "/api/report",
            json={"topic": "x" * 501},
        )
        assert response.status_code == 422

    def test_topic_at_boundary_is_accepted(self):
        """A topic of exactly 5 chars should be accepted."""
        with patch("api.generate_report") as mock_gen:
            mock_gen.return_value = ReportResult(
                topic="hello", report="Report", plan=[]
            )
            response = client.post(
                "/api/report",
                json={"topic": "hello"},
            )
            assert response.status_code == 200