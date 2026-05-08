"""
Streamlit frontend for the Analysis Report Agent.

Run with: streamlit run app.py
"""

import streamlit as st
from agent import generate_report, ReportResult

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Analysis Report Agent",
    page_icon="📊",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    .stApp { max-width: 1100px; margin: 0 auto; }
    .step-done { color: #2e7d32; }
    .step-running { color: #f57c00; }
    .step-error { color: #c62828; }
    .step-pending { color: #757575; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("📊 Statistics Canada Analysis Report Agent")
st.caption(
    "Enter a research topic related to Canadian housing, demographics, or "
    "socioeconomic trends. The agent will plan research steps, retrieve data "
    "from ingested Statistics Canada publications, and synthesize a structured report."
)

# ---------------------------------------------------------------------------
# Sidebar: settings
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("About")
    st.markdown(
        "**How it works**\n\n"
        "1. The *Planner* (Claude) breaks your topic into research steps\n"
        "2. The *Executor* runs each step using tools:\n"
        "   - Statistics Canada doc retriever (FAISS)\n"
        "   - DuckDuckGo web search\n"
        "   - Claude summarizer\n"
        "3. The *Synthesizer* compiles findings into a Markdown report"
    )
    st.divider()
    st.markdown(
        "**Data sources**\n\n"
        "Ingested from Statistics Canada publications on housing, "
        "rent, construction investment, co-residency, and housing need."
    )

# ---------------------------------------------------------------------------
# Main form
# ---------------------------------------------------------------------------
topic = st.text_area(
    "Research Topic",
    placeholder="e.g., Analyze Canada's housing affordability crisis using recent Statistics Canada data",
    height=80,
)

col1, col2 = st.columns([1, 5])
with col1:
    run_btn = st.button("Generate Report", type="primary", disabled=not topic)

# ---------------------------------------------------------------------------
# Run the agent
# ---------------------------------------------------------------------------
if run_btn and topic:
    # Containers for live updates
    plan_container = st.container()
    progress_bar = st.progress(0, text="Planning...")
    steps_container = st.container()
    report_container = st.container()

    # Step callback for live progress
    step_statuses = []

    def on_step(idx, desc, status):
        step_statuses.append((idx, desc, status))

    # Run the pipeline
    with st.spinner("Agent is working..."):
        result: ReportResult = generate_report(topic, on_step=on_step)

    # Display the plan
    with plan_container:
        st.subheader("Research Plan")
        for i, step in enumerate(result.plan):
            icon = {"done": "✅", "error": "❌", "running": "⏳", "pending": "⬜"}.get(
                step.status, "⬜"
            )
            st.markdown(f"{icon} **Step {i+1}:** {step.description}")

    progress_bar.progress(100, text="Complete")

    # Display step results in expanders
    with steps_container:
        st.subheader("Step Results")
        for i, step in enumerate(result.plan):
            with st.expander(f"Step {i+1}: {step.description}", expanded=False):
                if step.status == "error":
                    st.error(step.result)
                else:
                    st.markdown(step.result)

    # Display the final report
    with report_container:
        st.divider()
        st.subheader("Final Report")
        if result.error:
            st.warning(f"Agent encountered an issue: {result.error}")
        if result.report:
            st.markdown(result.report)

            # Download button
            st.download_button(
                label="Download Report (Markdown)",
                data=result.report,
                file_name="analysis_report.md",
                mime="text/markdown",
            )
        else:
            st.error("No report was generated.")