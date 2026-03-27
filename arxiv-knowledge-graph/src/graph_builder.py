import networkx as nx

def build_knowledge_graph(
    triples: list[tuple[str, str, str]],
    existing_graph: nx.DiGraph = None,
) -> nx.DiGraph:
    """Build or extend a knowledge graph from triples.

    Args:
        triples: List of (subject, predicate, object) tuples.
        existing_graph: Optional graph to extend.

    Returns:
        A NetworkX directed graph.
    """
    G = existing_graph if existing_graph is not None else nx.DiGraph()

    for subj, pred, obj in triples:
        # Normalize node names to title case for consistency
        subj_norm = subj.strip().title()
        obj_norm = obj.strip().title()


        G.add_node(subj_norm)
        G.add_node(obj_norm)
        G.add_edge(subj_norm, obj_norm, label=pred.strip())

    return G


def get_graph_stats(G: nx.DiGraph) -> dict:
    """Return summary statistics for the graph."""
    return {
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "density": round(nx.density(G), 4),
        "components": nx.number_weakly_connected_components(G),
        "most_connected": sorted(
            G.degree(), key=lambda x: x[1], reverse=True
        )[:5],
    }

def filter_graph_by_degree(
    G: nx.DiGraph,
    min_degree: int = 2,
) -> nx.DiGraph:
    """Return a subgraph containing only well-connected nodes.

    Useful for removing noise and focusing on central concepts.
    """
    nodes_to_keep = [
        n for n, d in G.degree() if d >= min_degree
    ]
    return G.subgraph(nodes_to_keep).copy()
