import pytest
import os




pytestmark = pytest.mark.skipif(
    not os.getenv("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY not set; skipping integration tests"
)




class TestEndToEnd:
    """Integration tests that hit real APIs.


    Run with: pytest tests/test_integration.py -v
    Requires ANTHROPIC_API_KEY to be set.
    """


    def test_arxiv_fetch_returns_papers(self):
        from src.arxiv_fetcher import fetch_arxiv_papers
        papers = fetch_arxiv_papers("attention mechanism", max_results=2)
        assert len(papers) > 0
        assert papers[0].title
        assert papers[0].abstract


    def test_full_pipeline(self):
        from src.arxiv_fetcher import fetch_arxiv_papers
        from src.triple_extractor import (
            build_extraction_chain, extract_triples
        )
        from src.graph_builder import build_knowledge_graph
        from src.visualizer import create_pyvis_graph, render_to_html


        papers = fetch_arxiv_papers("BERT transformer", max_results=1)
        assert len(papers) > 0


        chain = build_extraction_chain()
        triples = extract_triples(
            papers[0].title, papers[0].abstract, chain=chain
        )
        assert len(triples) > 0


        G = build_knowledge_graph(triples)
        assert G.number_of_nodes() > 0


        net = create_pyvis_graph(G)
        html = render_to_html(net, "/tmp/test_kg.html")
        assert "<html>" in html.lower() or "<div" in html.lower()
