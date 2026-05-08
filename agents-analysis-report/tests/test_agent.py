"""Tests for the Plan-and-Execute agent pipeline."""

from unittest.mock import patch, MagicMock

from agent import create_plan, synthesize_report, generate_report, ReportResult


class TestCreatePlan:
    """Tests for the planning phase."""

    @patch("agent.ChatAnthropic")
    def test_returns_list_of_steps(self, mock_llm_class):
        """create_plan should parse numbered steps from LLM output."""
        mock_response = MagicMock()
        mock_response.content = (
            "1. Search for AI regulation policies in the EU\n"
            "2. Search for AI regulation policies in the US\n"
            "3. Compare EU and US approaches\n"
            "4. Summarize key differences and trends"
        )
        mock_llm_class.return_value.invoke.return_value = mock_response

        steps = create_plan("AI regulation overview")

        assert len(steps) == 4
        assert "EU" in steps[0]
        assert "Summarize" in steps[3]

    @patch("agent.ChatAnthropic")
    def test_handles_bullet_points(self, mock_llm_class):
        """create_plan should parse bullet-pointed plans."""
        mock_response = MagicMock()
        mock_response.content = (
            "- Research topic A\n"
            "- Investigate topic B\n"
            "- Synthesize findings"
        )
        mock_llm_class.return_value.invoke.return_value = mock_response

        steps = create_plan("some topic")

        assert len(steps) == 3
        assert "Research" in steps[0]

    @patch("agent.ChatAnthropic")
    def test_handles_empty_response(self, mock_llm_class):
        """create_plan should return empty list if LLM returns nothing."""
        mock_response = MagicMock()
        mock_response.content = ""
        mock_llm_class.return_value.invoke.return_value = mock_response

        steps = create_plan("topic")

        assert steps == []


class TestSynthesizeReport:
    """Tests for the report synthesis phase."""

    @patch("agent.ChatAnthropic")
    def test_produces_markdown_report(self, mock_llm_class):
        """synthesize_report should return a Markdown string."""
        mock_response = MagicMock()
        mock_response.content = (
            "## Executive Summary\nAI regulation varies by country.\n\n"
            "## Key Findings\n- EU leads with the AI Act\n"
        )
        mock_llm_class.return_value.invoke.return_value = mock_response

        findings = [
            {"step": "Research EU", "result": "EU passed AI Act"},
            {"step": "Research US", "result": "US takes lighter approach"},
        ]

        report = synthesize_report("AI regulation", findings)

        assert "Executive Summary" in report
        assert isinstance(report, str)

    @patch("agent.ChatAnthropic")
    def test_handles_empty_findings(self, mock_llm_class):
        """synthesize_report should handle no findings gracefully."""
        mock_response = MagicMock()
        mock_response.content = "## Report\nInsufficient data to draw conclusions."
        mock_llm_class.return_value.invoke.return_value = mock_response

        report = synthesize_report("niche topic", [])

        assert isinstance(report, str)
        assert len(report) > 0


class TestGenerateReport:
    """Integration tests for the full pipeline."""

    @patch("agent.synthesize_report")
    @patch("agent.run_executor")
    @patch("agent.create_plan")
    def test_full_pipeline_success(self, mock_plan, mock_run_exec, mock_synth):
        """generate_report should run plan, execute, and synthesize."""
        mock_plan.return_value = ["Step A", "Step B"]
        mock_run_exec.return_value = "Result for step"
        mock_synth.return_value = "# Final Report\nGreat analysis."

        result = generate_report("Test topic")

        assert isinstance(result, ReportResult)
        assert result.topic == "Test topic"
        assert len(result.plan) == 2
        assert result.plan[0].status == "done"
        assert result.plan[1].status == "done"
        assert "Final Report" in result.report
        assert result.error == ""

    @patch("agent.create_plan", side_effect=Exception("LLM down"))
    def test_handles_planning_failure(self, mock_plan):
        """generate_report should capture planning errors."""
        result = generate_report("Test topic")

        assert "Planning failed" in result.error
        assert result.report == ""

    @patch("agent.synthesize_report")
    @patch("agent.run_executor")
    @patch("agent.create_plan")
    def test_handles_executor_failure(self, mock_plan, mock_run_exec, mock_synth):
        """generate_report should mark failed steps as error."""
        mock_plan.return_value = ["Step A"]
        mock_run_exec.side_effect = Exception("Tool crashed")
        mock_synth.return_value = "Partial report"

        result = generate_report("Test topic")

        assert result.plan[0].status == "error"
        assert "Error" in result.plan[0].result

    @patch("agent.synthesize_report")
    @patch("agent.run_executor")
    @patch("agent.create_plan")
    def test_on_step_callback_fires(self, mock_plan, mock_run_exec, mock_synth):
        """generate_report should call on_step for each plan step."""
        mock_plan.return_value = ["Step A", "Step B"]
        mock_run_exec.return_value = "ok"
        mock_synth.return_value = "Report"

        callback_log = []

        def tracker(idx, desc, status):
            callback_log.append((idx, status))

        generate_report("topic", on_step=tracker)

        # Each step fires "running" then "done" = 4 callbacks for 2 steps
        assert len(callback_log) == 4
        assert callback_log[0] == (0, "running")
        assert callback_log[1] == (0, "done")