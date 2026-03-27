from pyvis.network import Network
import networkx as nx


def create_pyvis_graph(
    G: nx.DiGraph,
    height: str = "600px",
    width: str = "100%",
    physics: bool = True,
    notebook: bool = False,
) -> Network:
    """Convert a NetworkX graph to an interactive Pyvis network.


    Args:
        G: The NetworkX directed graph.
        height: Height of the visualization.
        width: Width of the visualization.
        physics: Whether to enable physics simulation.
        notebook: Whether running in a Jupyter notebook.


    Returns:
        A configured Pyvis Network object.
    """
    net = Network(
        height=height,
        width=width,
        directed=True,
        notebook=notebook,
        bgcolor="#ffffff",
        font_color="#1B2A4A",
    )


    # Scale node size by degree
    degrees = dict(G.degree())
    max_deg = max(degrees.values()) if degrees else 1


    for node in G.nodes():
        size = 15 + (degrees[node] / max_deg) * 35
        net.add_node(
            node,
            label=node,
            size=size,
            color="#0D9488",
            font={"size": 14, "face": "Arial"},
        )


    for u, v, data in G.edges(data=True):
        net.add_edge(
            u, v,
            title=data.get("label", ""),
            label=data.get("label", ""),
            color="#94A3B8",
            font={"size": 10, "face": "Arial", "color": "#64748B"},
            arrows="to",
        )


    if physics:
        net.set_options("""
        {
            "physics": {
                "forceAtlas2Based": {
                    "gravitationalConstant": -50,
                    "centralGravity": 0.01,
                    "springLength": 200
                },
                "solver": "forceAtlas2Based"
            }
        }
        """)
    else:
        net.toggle_physics(False)


    return net




def render_to_html(
    net: Network,
    output_path: str = "knowledge_graph.html",
) -> str:
    """Render the Pyvis network to an HTML file.


    Returns:
        The HTML content as a string.
    """
    net.save_graph(output_path)
    with open(output_path, "r", encoding="utf-8") as f:
        return f.read()
