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

    def test_nodes_with_zero_degree_no_division_error(self):
        """Nodes with no edges should not cause ZeroDivisionError.

        This happens when filter_graph_by_degree produces a subgraph
        where surviving nodes lose all their edges because their
        neighbors were removed. max_deg becomes 0, and the node
        sizing formula (degrees[node] / max_deg) triggers division
        by zero without the fix.
        """
        G = nx.DiGraph()
        G.add_node("Large Language Models")
        G.add_node("Research")
        G.add_node("Authors")
        # No edges, all degrees are 0
        net = create_pyvis_graph(G)
        assert len(net.nodes) == 3
        assert len(net.edges) == 0

    def test_single_isolated_node(self):
        """A graph with one node and no edges should render without error."""
        G = nx.DiGraph()
        G.add_node("Transformer")
        net = create_pyvis_graph(G)
        assert len(net.nodes) == 1

    def test_mixed_isolated_and_connected_nodes(self):
        """Graph with some connected and some isolated nodes."""
        G = nx.DiGraph()
        G.add_edge("A", "B", label="relates to")
        G.add_node("C")  # isolated
        G.add_node("D")  # isolated
        net = create_pyvis_graph(G)
        assert len(net.nodes) == 4
        assert len(net.edges) == 1