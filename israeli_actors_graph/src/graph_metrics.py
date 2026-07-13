"""Per-period graph metrics for the Israeli actors graph (assignment Step 2).

Radius/diameter are undefined on a disconnected graph, so they're computed
on the largest connected component only (reusing utils.get_giant_component),
while density/clustering/degree stats use the full period graph.
"""
import os
import sys

import networkx as nx
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import utils


def compute_period_metrics(H: nx.Graph) -> dict:
    degrees = [d for _, d in H.degree()]
    components = sorted((len(c) for c in nx.connected_components(H)), reverse=True) if H.number_of_nodes() else [0]
    giant = utils.get_giant_component(H) if H.number_of_nodes() else H

    return {
        # basic
        "n_actors": H.number_of_nodes(),
        "n_collaborations": H.number_of_edges(),
        "density": nx.density(H),
        "avg_clustering": nx.average_clustering(H) if H.number_of_nodes() else float("nan"),
        # degree distribution
        "avg_degree": sum(degrees) / len(degrees) if degrees else 0,
        "max_degree": max(degrees) if degrees else 0,
        # connectivity
        "n_components": len(components),
        "largest_component_size": components[0],
        "largest_component_frac": components[0] / H.number_of_nodes() if H.number_of_nodes() else float("nan"),
        "radius": nx.radius(giant) if giant.number_of_nodes() > 1 else float("nan"),
        "diameter": nx.diameter(giant) if giant.number_of_nodes() > 1 else float("nan"),
    }


def metrics_table(periods: dict) -> pd.DataFrame:
    rows = {p: compute_period_metrics(H) for p, H in periods.items()}
    return pd.DataFrame(rows).T
