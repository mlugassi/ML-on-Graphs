"""Feature computation for link prediction on the Israeli actors graph.

All feature functions accept a *training* graph G and a list of (u, v)
node-id pairs, computing everything strictly from G to avoid data leakage.

Public API
----------
topological_features(G, pairs)                    -> pd.DataFrame
centrality_features(pairs, centralities)          -> pd.DataFrame
svd_node_embeddings(G, n_components, seed)        -> dict[node, np.ndarray]
node2vec_node_embeddings(G, ...)                  -> dict[node, np.ndarray]
cosine_similarity_feature(pairs, emb, prefix)     -> pd.DataFrame
hadamard_features(pairs, emb, prefix)             -> pd.DataFrame
build_feature_matrix(G, pairs, centralities, ...) -> (np.ndarray, list[str])
"""
from __future__ import annotations

import warnings

import networkx as nx
import numpy as np
import pandas as pd
from scipy.sparse import lil_matrix
from sklearn.decomposition import TruncatedSVD


# ---------------------------------------------------------------------------
# Topological features
# ---------------------------------------------------------------------------

def topological_features(
    G: nx.Graph,
    pairs: list[tuple],
    precomputed_sp: dict | None = None,
) -> pd.DataFrame:
    """Compute Common Neighbors, Jaccard, Adamic-Adar, Preferential
    Attachment, and shortest-path distance for each (u, v) pair.

    Parameters
    ----------
    G              : training graph (nx.Graph) — the historical snapshot
    pairs          : list of (u, v) node-id tuples
    precomputed_sp : optional dict {(u,v): distance} for fast lookups;
                     if None, shortest path is computed on-demand via BFS.

    Returns
    -------
    pd.DataFrame with columns:
        u, v, common_neighbors, jaccard, adamic_adar,
        pref_attachment, shortest_path
    """
    valid = [(u, v) for u, v in pairs if G.has_node(u) and G.has_node(v)]

    # Batch-compute networkx topological scores (efficient generator-based API)
    cn_map  = {(u, v): len(list(nx.common_neighbors(G, u, v))) for u, v in valid}
    jac_map = {(u, v): s for u, v, s in nx.jaccard_coefficient(G, valid)}
    aa_map  = {(u, v): s for u, v, s in nx.adamic_adar_index(G, valid)}
    pa_map  = {(u, v): s for u, v, s in nx.preferential_attachment(G, valid)}

    # Shortest-path distance (use precomputed dict if available)
    sp_map: dict[tuple, int] = {}
    for u, v in valid:
        if precomputed_sp is not None:
            sp_map[(u, v)] = precomputed_sp.get((u, v), precomputed_sp.get((v, u), -1))
        else:
            try:
                sp_map[(u, v)] = nx.shortest_path_length(G, u, v)
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                sp_map[(u, v)] = -1   # unreachable / different component

    rows = []
    for u, v in pairs:
        ok = G.has_node(u) and G.has_node(v)
        rows.append({
            "u":                u,
            "v":                v,
            "common_neighbors": cn_map.get((u, v), 0)   if ok else 0,
            "jaccard":          jac_map.get((u, v), 0.0) if ok else 0.0,
            "adamic_adar":      aa_map.get((u, v), 0.0)  if ok else 0.0,
            "pref_attachment":  pa_map.get((u, v), 0)    if ok else 0,
            "shortest_path":    sp_map.get((u, v), -1)   if ok else -1,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Centrality features
# ---------------------------------------------------------------------------

def centrality_features(
    pairs: list[tuple],
    centralities: dict[str, dict],
) -> pd.DataFrame:
    """Degree / betweenness / closeness / eigenvector centrality for both
    endpoints of each pair, plus their product and absolute difference.

    Parameters
    ----------
    pairs        : list of (u, v) node-id tuples
    centralities : {metric_name: {node_id: score}}

    Returns
    -------
    pd.DataFrame — columns:
        u, v,
        u_<metric>, v_<metric>, prod_<metric>, diff_<metric>
        for each metric in centralities.
    """
    rows = []
    for u, v in pairs:
        row: dict = {"u": u, "v": v}
        for metric, scores in centralities.items():
            us = scores.get(u, float("nan"))
            vs = scores.get(v, float("nan"))
            row[f"u_{metric}"] = us
            row[f"v_{metric}"] = vs
            both = (us == us) and (vs == vs)   # NaN-safe check
            row[f"prod_{metric}"] = us * vs       if both else float("nan")
            row[f"diff_{metric}"] = abs(us - vs)  if both else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# SVD node embeddings  (Matrix Factorisation baseline)
# ---------------------------------------------------------------------------

def svd_node_embeddings(
    G: nx.Graph,
    n_components: int = 32,
    seed: int = 42,
) -> dict:
    """Truncated SVD of the weighted adjacency matrix.

    A classic matrix-factorisation baseline for link prediction: the SVD
    captures the principal directions of co-occurrence in the graph, and
    nodes with similar neighborhoods end up with similar embeddings.

    Returns
    -------
    dict : node_id -> np.ndarray of shape (n_components,)
    """
    nodes = list(G.nodes())
    n     = len(nodes)
    idx   = {node: i for i, node in enumerate(nodes)}

    A = lil_matrix((n, n), dtype=float)
    for u, v, d in G.edges(data=True):
        w = float(d.get("weight", 1.0))
        A[idx[u], idx[v]] = w
        A[idx[v], idx[u]] = w

    k   = min(n_components, n - 1)
    svd = TruncatedSVD(n_components=k, random_state=seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        emb = svd.fit_transform(A.tocsr())       # shape (n, k)

    return {node: emb[idx[node]] for node in nodes}


# ---------------------------------------------------------------------------
# Node2Vec embeddings
# ---------------------------------------------------------------------------

def node2vec_node_embeddings(
    G: nx.Graph,
    dimensions: int = 64,
    walk_length: int = 20,
    num_walks: int = 50,
    p: float = 1.0,
    q: float = 1.0,
    seed: int = 42,
) -> dict:
    """Node2Vec random-walk embeddings.

    The return-parameter p and exploration-parameter q are set to 1 (default)
    which gives the standard DeepWalk-like uniform random walk; adjust p<1 for
    BFS-like (community-focused) walks and q<1 for DFS-like (structural-role)
    walks.

    The graph is converted to a simple undirected graph before fitting
    (Node2Vec does not support multi-edges).

    Raises ImportError if the ``node2vec`` package is not installed.

    Returns
    -------
    dict : node_id -> np.ndarray of shape (dimensions,)
    """
    try:
        from node2vec import Node2Vec  # type: ignore[import]
    except ImportError as exc:
        raise ImportError(
            "The 'node2vec' package is required. "
            "Install it with:  pip install node2vec"
        ) from exc

    G_simple = nx.Graph(G)   # collapse multi-edges; keep max weight

    n2v = Node2Vec(
        G_simple,
        dimensions=dimensions,
        walk_length=walk_length,
        num_walks=num_walks,
        p=p, q=q,
        workers=1,          # 1 worker avoids multi-process issues on Windows
        seed=seed,
        quiet=True,
    )
    model = n2v.fit(window=10, min_count=1, batch_words=4)

    zero = np.zeros(dimensions, dtype=float)
    emb: dict = {}
    for node in G.nodes():
        try:
            emb[node] = model.wv[str(node)]
        except KeyError:
            emb[node] = zero.copy()
    return emb


# ---------------------------------------------------------------------------
# Edge features derived from node embeddings
# ---------------------------------------------------------------------------

def _safe_emb(node, emb_dict: dict, dim: int) -> np.ndarray:
    """Return the embedding for ``node``, or a zero vector if missing."""
    e = emb_dict.get(node)
    return e if e is not None else np.zeros(dim, dtype=float)


def cosine_similarity_feature(
    pairs: list[tuple],
    emb_dict: dict,
    prefix: str = "",
) -> pd.DataFrame:
    """Cosine similarity and dot product between node embeddings.

    These *scalar* summary features are transfer-safe: they do not depend
    on the absolute orientation of the embedding space.  A model trained on
    cosine-similarity features from one graph snapshot generalises to another
    snapshot's embeddings without coordinate-alignment issues.
    """
    if not emb_dict:
        return pd.DataFrame(index=range(len(pairs)))

    dim = next(iter(emb_dict.values())).shape[0]
    cos_vals, dot_vals = [], []
    for u, v in pairs:
        ue = _safe_emb(u, emb_dict, dim)
        ve = _safe_emb(v, emb_dict, dim)
        norm = np.linalg.norm(ue) * np.linalg.norm(ve)
        cos_vals.append(float(np.dot(ue, ve) / norm) if norm > 1e-10 else 0.0)
        dot_vals.append(float(np.dot(ue, ve)))
    return pd.DataFrame({
        f"{prefix}cosine_sim":  cos_vals,
        f"{prefix}dot_product": dot_vals,
    })


def hadamard_features(
    pairs: list[tuple],
    emb_dict: dict,
    prefix: str = "",
) -> pd.DataFrame:
    """Hadamard (element-wise product) of node embedding vectors.

    Produces one column per embedding dimension.  These are NOT transfer-safe
    across different embedding spaces (different graphs / model runs), so the
    same embedding model must be used at train and inference time.  Use
    ``cosine_similarity_feature`` for cross-snapshot robustness.
    """
    if not emb_dict:
        return pd.DataFrame(index=range(len(pairs)))

    dim  = next(iter(emb_dict.values())).shape[0]
    rows = []
    for u, v in pairs:
        ue = _safe_emb(u, emb_dict, dim)
        ve = _safe_emb(v, emb_dict, dim)
        rows.append(ue * ve)
    return pd.DataFrame(rows, columns=[f"{prefix}{i}" for i in range(dim)])


# ---------------------------------------------------------------------------
# Master feature-matrix builder
# ---------------------------------------------------------------------------

def build_feature_matrix(
    G: nx.Graph,
    pairs: list[tuple],
    centralities: dict,
    svd_emb: dict | None = None,
    n2v_emb: dict | None = None,
    precomputed_sp: dict | None = None,
    years_ahead: int | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Assemble the full feature matrix for a list of (u, v) pairs.

    Feature groups included
    -----------------------
    * Topological (5)  : CN, Jaccard, Adamic-Adar, Pref. Attachment, SP dist.
    * Centrality (16)  : degree/betweenness/closeness/eigenvector
                         × {u_val, v_val, product, abs_diff}
    * SVD (2)          : cosine similarity + dot product  [if svd_emb given]
    * Node2Vec (2)     : cosine similarity + dot product  [if n2v_emb given]
    * years_ahead (1)  : prediction horizon in years      [if years_ahead given]
                         Enables multi-horizon training and inference —
                         the same model can score 1-year vs 5-year predictions
                         simply by changing this value at inference time.

    Parameters
    ----------
    G              : nx.Graph — snapshot used for feature computation
    pairs          : list of (u, v) node-id tuples
    centralities   : {metric_name: {node_id: score}}
    svd_emb        : optional SVD node embeddings
    n2v_emb        : optional Node2Vec node embeddings
    precomputed_sp : optional {(u,v): dist} for fast shortest-path lookup
    years_ahead    : optional int — how many years ahead we are predicting.
                     Added as a constant column so the model learns how
                     prediction difficulty scales with the horizon.

    Returns
    -------
    X             : np.ndarray, shape (len(pairs), n_features), NaN-free
    feature_names : list[str]
    """
    topo = topological_features(G, pairs, precomputed_sp=precomputed_sp)
    topo_cols = ["common_neighbors", "jaccard", "adamic_adar",
                 "pref_attachment", "shortest_path"]

    cent      = centrality_features(pairs, centralities)
    cent_cols = [c for c in cent.columns if c not in ("u", "v")]

    parts = [
        topo[topo_cols].reset_index(drop=True),
        cent[cent_cols].reset_index(drop=True),
    ]

    if svd_emb:
        parts.append(
            cosine_similarity_feature(pairs, svd_emb, prefix="svd_")
            .reset_index(drop=True)
        )
    if n2v_emb:
        parts.append(
            cosine_similarity_feature(pairs, n2v_emb, prefix="n2v_")
            .reset_index(drop=True)
        )

    if years_ahead is not None:
        ya_df = pd.DataFrame(
            {"years_ahead": [float(years_ahead)] * len(pairs)}
        ).reset_index(drop=True)
        parts.append(ya_df)

    X_df = pd.concat(parts, axis=1).fillna(0.0)
    return X_df.values.astype(float), list(X_df.columns)
