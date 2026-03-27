import streamlit as st
import streamlit.components.v1 as components
from arxiv_fetcher import fetch_arxiv_papers
from triple_extractor import (
    build_extraction_chain, extract_triples
)
from graph_builder import (
    build_knowledge_graph, get_graph_stats, filter_graph_by_degree
)
from visualizer import create_pyvis_graph, render_to_html


st.set_page_config(
    page_title="ArXiv Knowledge Graph Explorer",
    page_icon="graph_icon",
    layout="wide",
)


st.title("ArXiv Knowledge Graph Explorer")
st.markdown(
    "Extract entities and relationships from AI/ML research "
    "abstracts and visualize them as an interactive knowledge graph."
)


# ---- Sidebar controls ----
with st.sidebar:
    st.header("Settings")
    query = st.text_input(
        "ArXiv search query",
        value="transformer attention mechanism",
    )
    max_papers = st.slider("Number of papers", 1, 15, 5)
    min_degree = st.slider(
        "Min connections to display", 1, 5, 1,
        help="Filter out nodes with fewer connections than this."
    )
    use_physics = st.checkbox("Enable physics simulation", value=True)
    fetch_btn = st.button("Build Knowledge Graph", type="primary")


# ---- Session state ----
if "graph" not in st.session_state:
    st.session_state.graph = None
if "papers" not in st.session_state:
    st.session_state.papers = []
if "all_triples" not in st.session_state:
    st.session_state.all_triples = []


# ---- Main logic ----
if fetch_btn:
    with st.spinner("Fetching papers from ArXiv..."):
        papers = fetch_arxiv_papers(query, max_results=max_papers)


    if not papers:
        st.warning("No papers found. Try a different query.")
    else:
        st.session_state.papers = papers
        chain = build_extraction_chain()
        all_triples = []


        progress = st.progress(0, text="Extracting knowledge...")
        for i, paper in enumerate(papers):
            triples = extract_triples(
                paper.title, paper.abstract, chain=chain
            )
            all_triples.extend(triples)
            progress.progress(
                (i + 1) / len(papers),
                text=f"Processing {i+1}/{len(papers)}: {paper.title[:60]}..."
            )


        st.session_state.all_triples = all_triples
        G = build_knowledge_graph(all_triples)
        st.session_state.graph = G
        progress.empty()


# ---- Display results ----
if st.session_state.graph is not None:
    G = st.session_state.graph


    # Apply degree filter
    if min_degree > 1:
        G_display = filter_graph_by_degree(G, min_degree)
    else:
        G_display = G


    if G_display.number_of_nodes() == 0:
        st.info("No nodes meet the minimum connection threshold. "
                "Try lowering the filter.")
    else:
        # Stats row
        stats = get_graph_stats(G_display)
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Nodes", stats["nodes"])
        col2.metric("Edges", stats["edges"])
        col3.metric("Density", stats["density"])
        col4.metric("Components", stats["components"])


        # Render graph
        net = create_pyvis_graph(G_display, physics=use_physics)
        html = render_to_html(net, "temp_graph.html")
        components.html(html, height=620, scrolling=True)


        # Show most connected entities
        st.subheader("Most Connected Entities")
        for name, deg in stats["most_connected"]:
            st.write(f"**{name}**: {deg} connections")


    # Papers processed
    with st.expander("Papers processed"):
        for p in st.session_state.papers:
            st.markdown(f"**{p.title}** ({p.arxiv_id})")
            st.caption(p.abstract[:200] + "...")


    # Raw triples
    with st.expander("Extracted triples"):
        for s, p, o in st.session_state.all_triples:
            st.text(f"({s}, {p}, {o})")
