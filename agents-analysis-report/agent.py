"""
Plan-and-Execute Agent for Analysis Report Creation.

The agent operates in three phases:
  1. PLAN      -- The LLM breaks the user's topic into a sequence of research steps.
  2. EXECUTE   -- Each step is carried out using the available tools.
  3. SYNTHESIZE -- The gathered information is compiled into a structured report.

This mirrors the Plan-and-Solve prompting approach described in the
"Building LLMs for Production" textbook, modernized for current LangChain APIs.
"""

from dataclasses import dataclass, field

from langchain_anthropic import ChatAnthropic
from langgraph.prebuilt import create_react_agent

import config
from tools import ALL_TOOLS


# ---------------------------------------------------------------------------
# Data classes for structured output
# ---------------------------------------------------------------------------
@dataclass
class AgentStep:
    """One step in the plan with its result."""
    description: str
    result: str = ""
    status: str = "pending"  # pending | running | done | error


@dataclass
class ReportResult:
    """Final output of the agent pipeline."""
    topic: str
    plan: list[AgentStep] = field(default_factory=list)
    report: str = ""
    error: str = ""


# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------
PLANNER_PROMPT = """\
You are a research planner. Given a topic, produce a numbered list of 3-6 
concrete research steps that an executor agent should follow to write a 
comprehensive analysis report on the topic.

Each step should be a specific action like:
- "Search for X in the document store"
- "Summarize findings about Y"
- "Compare perspectives on Z"

Respond ONLY with the numbered list. No preamble, no closing remarks.

Topic: {topic}
"""


def create_plan(topic: str) -> list[str]:
    """Use the LLM to generate a research plan for the given topic."""
    llm = ChatAnthropic(model=config.LLM_MODEL, temperature=0.0)
    response = llm.invoke(PLANNER_PROMPT.format(topic=topic))
    lines = response.content.strip().split("\n")
    steps = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        # Strip leading number/bullet
        for prefix in ("- ", "* "):
            if line.startswith(prefix):
                line = line[len(prefix):]
        if line and line[0].isdigit():
            # Remove "1. ", "2) " etc.
            idx = 0
            while idx < len(line) and (line[idx].isdigit() or line[idx] in ".)"):
                idx += 1
            line = line[idx:].strip()
        if line:
            steps.append(line)
    return steps


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------
EXECUTOR_SYSTEM = """\
You are a research executor. You have access to tools for searching a document 
store, summarizing text, and searching the web.

Given a specific research step, use the appropriate tools to gather information 
and return a concise, factual answer. Always prefer the document retriever first.
If the retriever returns insufficient results, try the web search tool.
After gathering raw information, use the summarizer if the text is long.

Be thorough but concise. Cite specific facts when available.
"""


def run_executor(step_description: str) -> str:
    """Run a single research step using the react agent."""
    llm = ChatAnthropic(model=config.LLM_MODEL, temperature=0.0)
    agent = create_react_agent(llm, ALL_TOOLS, prompt=EXECUTOR_SYSTEM)
    result = agent.invoke(
        {"messages": [{"role": "user", "content": step_description}]},
    )
    # Extract the final AI message content
    messages = result.get("messages", [])
    for msg in reversed(messages):
        if hasattr(msg, "content") and msg.content and not hasattr(msg, "tool_calls"):
            return msg.content
        if isinstance(msg, dict) and msg.get("content"):
            return msg["content"]
    return "No output from executor."


# ---------------------------------------------------------------------------
# Synthesizer
# ---------------------------------------------------------------------------
SYNTHESIS_PROMPT = """\
You are an expert analyst and technical writer. Using the research findings below, 
write a well-structured analysis report on the topic: "{topic}".

The report MUST follow this structure:
1. **Executive Summary** -- 2-3 sentence overview
2. **Key Findings** -- the main facts and insights, organized by theme
3. **Analysis** -- your interpretation of the findings, connections between themes
4. **Conclusion** -- summary of takeaways and potential implications

Use Markdown formatting. Be factual, concise, and insightful.

Research Findings:
{findings}
"""


def synthesize_report(topic: str, findings: list[dict]) -> str:
    """Compile step results into a structured Markdown report."""
    llm = ChatAnthropic(model=config.LLM_MODEL, temperature=0.2)
    findings_text = "\n\n".join(
        f"### Step: {f['step']}\n{f['result']}" for f in findings
    )
    response = llm.invoke(
        SYNTHESIS_PROMPT.format(topic=topic, findings=findings_text)
    )
    return response.content


# ---------------------------------------------------------------------------
# Full Pipeline
# ---------------------------------------------------------------------------
def generate_report(
    topic: str,
    on_step: callable = None,
) -> ReportResult:
    """
    Run the full Plan-and-Execute pipeline.

    Args:
        topic: The research topic/question.
        on_step: Optional callback(step_index, step_description, status) for
                 progress updates (used by the Streamlit frontend).

    Returns:
        ReportResult with the plan, individual step results, and final report.
    """
    result = ReportResult(topic=topic)

    # Phase 1: Plan
    try:
        step_descriptions = create_plan(topic)
        result.plan = [AgentStep(description=s) for s in step_descriptions]
    except Exception as e:
        result.error = f"Planning failed: {e}"
        return result

    # Phase 2: Execute each step
    findings = []

    for i, step in enumerate(result.plan):
        step.status = "running"
        if on_step:
            on_step(i, step.description, "running")

        try:
            output = run_executor(step.description)
            step.result = output
            step.status = "done"
            findings.append({"step": step.description, "result": step.result})
        except Exception as e:
            step.result = f"Error: {e}"
            step.status = "error"
            findings.append({"step": step.description, "result": step.result})

        if on_step:
            on_step(i, step.description, step.status)

    # Phase 3: Synthesize
    try:
        result.report = synthesize_report(topic, findings)
    except Exception as e:
        result.error = f"Synthesis failed: {e}"

    return result