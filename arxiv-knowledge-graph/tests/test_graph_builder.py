import pytest
import networkx as nx
from src.graph_builder import (
    build_knowledge_graph, get_graph_stats, filter_graph_by_degree
)

SAMPLE_TRIPLES = [
    ("BERT", "is a", "language model"),
    ("BERT", "uses", "self-attention"),
    ("GPT", "is a", "language model"),
    ("GPT", "uses", "self-attention"),
    ("Transformer", "introduced", "self-attention"),
]

class TestBuildKnowledgeGraph:
    def test_creates_graph_with_correct_node_count(self):
        G = build_knowledge_graph(SAMPLE_TRIPLES)
        # Bert, Gpt, Language Model, Self-Attention, Transformer
        assert G.number_of_nodes() == 5


    def test_creates_correct_number_of_edges(self):
        G = build_knowledge_graph(SAMPLE_TRIPLES)
        assert G.number_of_edges() == 5


    def test_normalizes_to_title_case(self):
        G = build_knowledge_graph([("bert", "is a", "MODEL")])
        assert "Bert" in G.nodes()
        assert "Model" in G.nodes()


    def test_extends_existing_graph(self):
        G1 = build_knowledge_graph(SAMPLE_TRIPLES[:2])
        G2 = build_knowledge_graph(SAMPLE_TRIPLES[2:], existing_graph=G1)
        assert G2.number_of_edges() == 5
        assert G2 is G1  # same object, mutated


    def test_empty_triples(self):
        G = build_knowledge_graph([])
        assert G.number_of_nodes() == 0




class TestFilterGraphByDegree:
    def test_filters_low_degree_nodes(self):
        G = build_knowledge_graph(SAMPLE_TRIPLES)
        filtered = filter_graph_by_degree(G, min_degree=2)
        # Only well-connected nodes survive
        for _, deg in filtered.degree():
            assert deg >= 2


    def test_high_filter_returns_fewer_nodes(self):
        G = build_knowledge_graph(SAMPLE_TRIPLES)
        filtered = filter_graph_by_degree(G, min_degree=3)
        assert filtered.number_of_nodes() <= G.number_of_nodes()

class TestGetGraphStats:
    def test_returns_expected_keys(self):
        G = build_knowledge_graph(SAMPLE_TRIPLES)
        stats = get_graph_stats(G)
        assert "nodes" in stats
        assert "edges" in stats
        assert "density" in stats
        assert "components" in stats
        assert "most_connected" in stats


    def test_most_connected_sorted_descending(self):
        G = build_knowledge_graph(SAMPLE_TRIPLES)
        stats = get_graph_stats(G)
        degrees = [d for _, d in stats["most_connected"]]
        assert degrees == sorted(degrees, reverse=True)
