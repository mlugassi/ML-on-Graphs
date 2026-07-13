"""Centrality analysis for the Israeli actors graph (assignment Step 4).

Degree/betweenness/closeness are computed unweighted (each collaboration
counts once, regardless of how many films two actors shared) since
networkx's shortest-path-based algorithms treat `weight` as a *distance*
(lower = closer), which is the opposite of what our collaboration-count
weight means. Eigenvector centrality is computed *weighted*, since there
`weight` correctly acts as tie *strength* (more shared films = stronger
influence), which is the right semantics for that algorithm.
"""
import os
import sys

import networkx as nx
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import utils
from graph_build import period_of_year


def compute_all_centralities(H: nx.Graph) -> dict:
    # Eigenvector centrality is ambiguous on a disconnected graph (the
    # eigenvector could concentrate in any one component), so it's computed
    # on the largest connected component only; actors outside it get NaN.
    giant = utils.get_giant_component(H)
    eigenvector_giant = nx.eigenvector_centrality_numpy(giant, weight="weight")

    return {
        "degree": nx.degree_centrality(H),
        "betweenness": nx.betweenness_centrality(H, weight=None),
        "closeness": nx.closeness_centrality(H),
        "eigenvector": {n: eigenvector_giant.get(n, float("nan")) for n in H.nodes},
    }


def centrality_table(H: nx.Graph, centralities: dict, period: str | None = None) -> pd.DataFrame:
    """Build the combined centrality table. If `period` is given, adds
    `n_films` = number of that actor's film credits *in this period*
    (distinct from `n_costars`, the number of distinct co-stars -- an actor
    can be in few films with large ensemble casts, or many films with small
    casts, so these answer different questions)."""
    df = pd.DataFrame(centralities)
    df["display_name"] = [H.nodes[n].get("display_name", n) for n in df.index]
    df["n_costars"] = [H.degree(n) for n in df.index]
    df["n_collaborations"] = [H.degree(n, weight="weight") for n in df.index]

    if period is not None:
        def n_films(n):
            return sum(1 for c in H.nodes[n].get("credits", []) if period_of_year(c["year"]) == period)
        df["n_films"] = [n_films(n) for n in df.index]

    # Average shortest-path distance within the largest connected component
    # -- a more intuitive companion to the raw closeness score ("closeness"
    # is 1/(avg distance among *reachable* nodes), not obviously readable
    # on its own).
    giant = utils.get_giant_component(H)
    avg_dist = {}
    for n in giant.nodes:
        lengths = nx.shortest_path_length(giant, source=n)
        others = [d for target, d in lengths.items() if target != n]
        avg_dist[n] = sum(others) / len(others) if others else float("nan")
    df["avg_distance_giant_component"] = [avg_dist.get(n, float("nan")) for n in df.index]

    # For eigenvector centrality: name each actor's single most-important
    # costar (highest eigenvector score among their direct neighbors) --
    # makes "connected to other important actors" concrete rather than an
    # abstract number.
    eig = centralities["eigenvector"]
    top_costar = {}
    for n in H.nodes:
        neighbors = list(H.neighbors(n))
        scored = [(nbr, eig.get(nbr, float("nan"))) for nbr in neighbors]
        scored = [(nbr, s) for nbr, s in scored if s == s]  # drop NaN
        if scored:
            best_nbr, best_score = max(scored, key=lambda t: t[1])
            top_costar[n] = H.nodes[best_nbr].get("display_name", best_nbr)
        else:
            top_costar[n] = None
    df["most_important_costar"] = [top_costar.get(n) for n in df.index]

    return df


def top_n(df: pd.DataFrame, metric: str, n: int = 15, extra_cols: list | None = None) -> pd.DataFrame:
    cols = ["display_name", metric, "n_costars", "n_collaborations"]
    if extra_cols:
        cols += [c for c in extra_cols if c not in cols]
    return df.sort_values(metric, ascending=False).head(n)[cols]


def community_bridging_score(H: nx.Graph, communities: list) -> pd.Series:
    """For each node: how many *distinct* communities its neighbors belong
    to. A node whose co-stars span many different communities is acting as
    a bridge between them, regardless of its raw betweenness value."""
    node_to_community = {n: i for i, c in enumerate(communities) for n in c}
    scores = {}
    for n in H.nodes:
        neighbor_communities = {node_to_community[nbr] for nbr in H.neighbors(n)}
        scores[n] = len(neighbor_communities)
    return pd.Series(scores, name="n_distinct_neighbor_communities")


def graph_center(H: nx.Graph) -> tuple[pd.DataFrame, int]:
    """The graph-theoretic center: nodes whose eccentricity (max distance to
    any other reachable node) equals the graph's radius, computed on the
    largest connected component. This is a distinct concept from closeness
    centrality -- closeness ranks *everyone* by average distance, while the
    center is the small set of nodes that literally minimize the *worst-case*
    distance to any other actor ("who is at the center of the network")."""
    giant = utils.get_giant_component(H)
    eccentricity = nx.eccentricity(giant)
    radius = min(eccentricity.values())
    center_nodes = [n for n, e in eccentricity.items() if e == radius]

    df = pd.DataFrame({
        "display_name": [H.nodes[n]["display_name"] for n in center_nodes],
        "eccentricity": [eccentricity[n] for n in center_nodes],
        "n_costars": [H.degree(n) for n in center_nodes],
    }).sort_values("display_name").reset_index(drop=True)
    return df, radius


def big_stars_and_their_costars(H: nx.Graph, eigenvector_scores: dict, n_stars: int = 8, costars_per_star: int = 10) -> pd.DataFrame:
    """"Who works with the big stars": take the top-N actors by eigenvector
    centrality (the "big stars" -- highly connected to other important
    actors) and list each one's own most-frequent direct co-stars. Distinct
    from the eigenvector ranking itself, which answers "who *is* well
    connected to important actors" rather than "who specifically works with
    each big star"."""
    ranked_stars = sorted(
        ((n, s) for n, s in eigenvector_scores.items() if s == s),
        key=lambda kv: kv[1], reverse=True,
    )[:n_stars]

    rows = []
    for star, star_score in ranked_stars:
        star_name = H.nodes[star]["display_name"]
        neighbors = sorted(H.neighbors(star), key=lambda n: H[star][n].get("weight", 1), reverse=True)
        for nbr in neighbors[:costars_per_star]:
            rows.append({
                "big_star": star_name,
                "big_star_eigenvector": star_score,
                "costar": H.nodes[nbr]["display_name"],
                "n_shared_films": H[star][nbr].get("weight", 1),
            })
    return pd.DataFrame(rows)


def bridging_detail(H: nx.Graph, communities: list, node, top_k: int = 5) -> str:
    """Human-readable summary of which specific communities a node bridges:
    for each distinct community among its neighbors, the community's size
    and its highest-degree ("anchor") member -- makes bridging concrete
    instead of just a count."""
    node_to_community = {n: i for i, c in enumerate(communities) for n in c}
    by_community = {}
    for nbr in H.neighbors(node):
        cid = node_to_community[nbr]
        by_community.setdefault(cid, []).append(nbr)

    parts = []
    for cid, members_seen in sorted(by_community.items(), key=lambda kv: -len(communities[kv[0]])):
        community = communities[cid]
        anchor = max(community, key=lambda n: H.degree(n))
        anchor_name = H.nodes[anchor].get("display_name", anchor)
        parts.append(f"community #{cid} (size {len(community)}, anchored by {anchor_name})")
        if len(parts) >= top_k:
            break
    return "; ".join(parts)
