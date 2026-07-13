"""Community detection for the Israeli actors graph (assignment Step 3,
Period C only). Compares three algorithms already available in
networkx>=3.5 -- no extra dependencies needed.
"""
from collections import Counter

import networkx as nx
import pandas as pd
from sklearn.metrics import normalized_mutual_info_score

from graph_build import period_of_year

ALGORITHMS = ("louvain", "greedy_modularity", "label_propagation")


def run_algorithms(H: nx.Graph, seed: int = 42) -> dict:
    """Run all three community-detection algorithms; returns {name: [set(nodes), ...]}."""
    return {
        "louvain": nx.community.louvain_communities(H, weight="weight", seed=seed),
        "greedy_modularity": nx.community.greedy_modularity_communities(H, weight="weight"),
        "label_propagation": list(nx.community.asyn_lpa_communities(H, weight="weight", seed=seed)),
    }


def _labels_for_nmi(H: nx.Graph, communities: list) -> list:
    """Map each node in H to its community index, in H's node order."""
    node_to_label = {n: i for i, community in enumerate(communities) for n in community}
    return [node_to_label[n] for n in H.nodes()]


def compare_algorithms(H: nx.Graph, results: dict) -> pd.DataFrame:
    rows = []
    for name, communities in results.items():
        sizes = sorted((len(c) for c in communities), reverse=True)
        rows.append({
            "algorithm": name,
            "n_communities": len(communities),
            "modularity": nx.community.modularity(H, communities, weight="weight"),
            "largest_community": sizes[0] if sizes else 0,
            "median_community": sizes[len(sizes) // 2] if sizes else 0,
            "n_singletons": sum(1 for s in sizes if s == 1),
        })
    return pd.DataFrame(rows).set_index("algorithm")


def pairwise_nmi(H: nx.Graph, results: dict) -> pd.DataFrame:
    """Normalized mutual information between every pair of algorithms'
    partitions -- how much the algorithms agree on community assignment."""
    names = list(results.keys())
    labels = {name: _labels_for_nmi(H, results[name]) for name in names}
    matrix = pd.DataFrame(index=names, columns=names, dtype=float)
    for a in names:
        for b in names:
            matrix.loc[a, b] = normalized_mutual_info_score(labels[a], labels[b])
    return matrix


def characterize_communities(H: nx.Graph, communities: list, period: str = "C", top_n: int = 5, min_size: int = 3) -> pd.DataFrame:
    """For each community (size >= min_size): dominant genres, active-year
    range, and top-degree actors, restricted to credits from `period` so the
    characterization reflects the period being analyzed."""
    rows = []
    for community in sorted(communities, key=len, reverse=True):
        if len(community) < min_size:
            continue

        sub_degrees = {n: H.degree(n) for n in community}
        top_actors = sorted(sub_degrees, key=sub_degrees.get, reverse=True)[:top_n]
        top_names = [H.nodes[n].get("display_name", n) for n in top_actors]

        genre_counter = Counter()
        years = []
        for n in community:
            for c in H.nodes[n].get("credits", []):
                if period_of_year(c["year"]) != period:
                    continue
                years.append(c["year"])
                if c["genre"]:
                    genre_counter[c["genre"]] += 1

        top_genres = [g for g, _ in genre_counter.most_common(3)]
        rows.append({
            "size": len(community),
            "top_actors": ", ".join(top_names),
            "top_genres": ", ".join(top_genres) if top_genres else "(none listed)",
            "year_range": f"{min(years)}-{max(years)}" if years else "n/a",
        })

    return pd.DataFrame(rows)
