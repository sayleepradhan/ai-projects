import pytest
import networkx as nx
from src.visualizer import create_pyvis_graph
from src.graph_builder import build_knowledge_graph


SAMPLE_TRIPLES = [
    ("BERT", "is a", "language model"),
    ("BERT", "uses", "self-attention"),
]


class TestCreatePyvisGraph:
    def test_returns_network_object(self):
        G = build_knowledge_graph(SAMPLE_TRIPLES)
        net = create_pyvis_graph(G)
        assert net is not None


    def test_node_count_matches(self):
        G = build_knowledge_graph(SAMPLE_TRIPLES)
        net = create_pyvis_graph(G)
        assert len(net.nodes) == G.number_of_nodes()


    def test_edge_count_matches(self):
        G = build_knowledge_graph(SAMPLE_TRIPLES)
        net = create_pyvis_graph(G)
        assert len(net.edges) == G.number_of_edges()


    def test_handles_empty_graph(self):
        G = nx.DiGraph()
        net = create_pyvis_graph(G)
        assert len(net.nodes) == 0


    def test_physics_disabled(self):
        G = build_knowledge_graph(SAMPLE_TRIPLES)
        net = create_pyvis_graph(G, physics=False)
        # Should not raise; physics toggled off
        assert net is not None
