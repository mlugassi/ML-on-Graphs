"""Temporal link prediction pipeline for the Israeli actors graph.

Temporal design
---------------
G_early  : films 1991 <= year <= TRAIN_CUTOFF (default 2015)
             features computed from this graph for model selection
G_mid    : films 1991 <= year <= VAL_CUTOFF   (default 2020)
             features computed from this graph for true holdout eval
G_full   : all films in the dataset
             used for finding test positives and future predictions

Model-selection phase
    - Features from G_early
    - Positive pairs  = new edges formed in (TRAIN_CUTOFF, VAL_CUTOFF]
    - Negative pairs  = non-edges in G_early AND in G_mid
    - Split 80 / 20 -> train_split / val_split to compare LR vs. RF
    - Best model selected by AUC-ROC on val_split

True holdout evaluation
    - Features from G_mid
    - Positive pairs  = new edges formed after VAL_CUTOFF in G_full
    - Negative pairs  = non-edges through the end of the observation window
    - Best model evaluated on this unseen test set

Exploratory future predictions
    - Retrain best model on full model-selection data (features from G_mid)
    - Score all currently-unconnected pairs in G_full
    - Rank by predicted probability as a "watch list"
"""
from __future__ import annotations

import os
import sys
import warnings
from itertools import combinations

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Add repo root (for utils.py) and src (for graph_build, link_features)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.dirname(__file__))

import graph_build
import link_features as lf
import utils

# Default year boundaries
YEAR_START   = 1990     # first year included (1990 inclusive)
TRAIN_CUTOFF = 2015     # G_early: 1990 <= year <= this
VAL_CUTOFF   = 2020     # G_mid:   1990 <= year <= this
NEG_RATIO    = 1.5      # negatives per positive
RANDOM_SEED  = 42


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def build_graph_for_years(
    cast_edges: pd.DataFrame,
    year_min: int,
    year_max: int,
) -> nx.Graph:
    """Build a simple weighted actor co-appearance graph from films
    with ``year_min <= year <= year_max``.

    Reuses ``graph_build.build_full_graph`` on a filtered slice of
    ``cast_edges``, then collapses multi-edges to a weighted simple graph
    (weight = number of shared films).

    Returns an empty nx.Graph if no films fall in the specified range.
    """
    subset = cast_edges[
        (cast_edges["year"] >= year_min) & (cast_edges["year"] <= year_max)
    ].copy()

    if subset.empty:
        return nx.Graph()

    G_multi = graph_build.build_full_graph(subset)

    H = nx.Graph()
    for n, data in G_multi.nodes(data=True):
        H.add_node(n, **data)
    for u, v, edata in G_multi.edges(data=True):
        if H.has_edge(u, v):
            H[u][v]["weight"] += 1
            H[u][v]["films"].append({
                "title":      edata["title"],
                "year":       edata["year"],
                "movie_slug": edata.get("movie_slug", ""),
            })
        else:
            H.add_edge(u, v, weight=1, films=[{
                "title":      edata["title"],
                "year":       edata["year"],
                "movie_slug": edata.get("movie_slug", ""),
            }])
    return H


# ---------------------------------------------------------------------------
# Positive / negative pair construction
# ---------------------------------------------------------------------------

def get_new_pairs(G_before: nx.Graph, G_after: nx.Graph) -> list[tuple]:
    """Return pairs that became connected in G_after but were NOT connected
    in G_before, restricted to actors who already existed in G_before.

    Actors who debuted after the G_before snapshot cannot be predicted by
    the model (a known limitation noted in the report).
    """
    known = set(G_before.nodes())
    positives = set()
    for u, v in G_after.edges():
        if u in known and v in known and not G_before.has_edge(u, v):
            positives.add((min(u, v), max(u, v)))
    return list(positives)


def sample_negatives(
    G: nx.Graph,
    n_samples: int,
    exclude_pairs: set | None = None,
    min_degree: int = 1,
    seed: int = RANDOM_SEED,
) -> list[tuple]:
    """Sample non-edges from G as negative link-prediction examples.

    Only considers nodes with degree >= ``min_degree`` (active actors, not
    one-off appearances with no co-stars).  All pairs in ``exclude_pairs``
    are excluded — pass the set of positive pairs so that no true positive
    ends up in the negative sample.

    Parameters
    ----------
    G            : training graph
    n_samples    : number of negative pairs to return
    exclude_pairs: additional pairs to exclude (e.g. future positives)
    min_degree   : minimum degree to be eligible as a candidate node
    seed         : random seed for reproducibility

    Returns
    -------
    list of (u, v) tuples (canonicalised as (min, max))
    """
    rng = np.random.default_rng(seed)
    eligible = [n for n in G.nodes() if G.degree(n) >= min_degree]

    existing  = {(min(u, v), max(u, v)) for u, v in G.edges()}
    excluded  = exclude_pairs or set()
    seen: set = set()
    negatives: list[tuple] = []

    max_attempts = n_samples * 200
    attempts = 0
    while len(negatives) < n_samples and attempts < max_attempts:
        i, j = rng.choice(len(eligible), size=2, replace=False)
        u, v  = eligible[i], eligible[j]
        pair  = (min(u, v), max(u, v))
        if pair not in existing and pair not in excluded and pair not in seen:
            negatives.append(pair)
            seen.add(pair)
        attempts += 1

    if len(negatives) < n_samples:
        warnings.warn(
            f"sample_negatives: returned {len(negatives)} of {n_samples} "
            "requested.  Try reducing neg_ratio or min_degree.",
            stacklevel=2,
        )
    return negatives


# ---------------------------------------------------------------------------
# Centrality computation (thin wrapper for link-prediction context)
# ---------------------------------------------------------------------------

def compute_centralities(H: nx.Graph) -> dict:
    """Compute the four centrality measures used as link-prediction features.

    Eigenvector centrality is only defined on the largest connected component;
    actors outside it receive 0 (not NaN) so the feature matrix stays clean.
    """
    giant = utils.get_giant_component(H)
    try:
        eig_giant = nx.eigenvector_centrality_numpy(giant, weight="weight")
    except Exception:
        eig_giant = {}

    return {
        "degree":      nx.degree_centrality(H),
        "betweenness": nx.betweenness_centrality(H, weight=None),
        "closeness":   nx.closeness_centrality(H),
        "eigenvector": {n: eig_giant.get(n, 0.0) for n in H.nodes()},
    }


# ---------------------------------------------------------------------------
# Dataset assembly
# ---------------------------------------------------------------------------

def build_link_dataset(
    G_feat: nx.Graph,
    positive_pairs: list[tuple],
    neg_ratio: float = NEG_RATIO,
    svd_emb: dict | None = None,
    n2v_emb: dict | None = None,
    exclude_from_neg: set | None = None,
    seed: int = RANDOM_SEED,
) -> tuple[np.ndarray, np.ndarray, list[tuple], list[str], dict]:
    """Build (X, y, pairs, feature_names, centralities) for a set of
    positive pairs, sampling negatives from G_feat.

    Parameters
    ----------
    G_feat          : graph used for ALL feature computation (no leakage)
    positive_pairs  : list of (u, v) positive examples (label = 1)
    neg_ratio       : negative samples per positive
    svd_emb         : optional SVD node embeddings dict
    n2v_emb         : optional Node2Vec node embeddings dict
    exclude_from_neg: additional pairs excluded from negative sampling
                      (pass future positives to ensure true negatives)
    seed            : random seed

    Returns
    -------
    X             : np.ndarray, shape (n_pairs, n_features)
    y             : np.ndarray of {0, 1}
    pairs         : list of (u, v) in the same row-order as X / y
    feature_names : list[str]
    centralities  : dict (reuse for further analysis in the notebook)
    """
    n_neg      = max(1, int(len(positive_pairs) * neg_ratio))
    pos_set    = {(min(u, v), max(u, v)) for u, v in positive_pairs}
    excl       = pos_set | (exclude_from_neg or set())

    neg_pairs  = sample_negatives(G_feat, n_neg, exclude_pairs=excl, seed=seed)

    all_pairs  = [(min(u, v), max(u, v)) for u, v in positive_pairs] + neg_pairs
    y          = np.array([1] * len(positive_pairs) + [0] * len(neg_pairs))

    cents      = compute_centralities(G_feat)
    X, names   = lf.build_feature_matrix(G_feat, all_pairs, cents,
                                          svd_emb=svd_emb, n2v_emb=n2v_emb)
    return X, y, all_pairs, names, cents


# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------

def train_logistic_regression(
    X: np.ndarray,
    y: np.ndarray,
    seed: int = RANDOM_SEED,
) -> tuple:
    """Train a Logistic Regression classifier with standard scaling.

    Returns (model, scaler) — always pass the same scaler when calling
    evaluate_model or predict_proba on this model.
    """
    scaler  = StandardScaler()
    X_sc    = scaler.fit_transform(X)
    model   = LogisticRegression(
        max_iter=2000, random_state=seed, class_weight="balanced", C=1.0
    )
    model.fit(X_sc, y)
    return model, scaler


def train_random_forest(
    X: np.ndarray,
    y: np.ndarray,
    seed: int = RANDOM_SEED,
) -> RandomForestClassifier:
    """Train a Random Forest classifier (no scaling needed)."""
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(X, y)
    return model


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_model(
    model,
    X: np.ndarray,
    y: np.ndarray,
    scaler: StandardScaler | None = None,
    threshold: float = 0.5,
) -> dict:
    """Return a metrics dict: Precision, Recall, F1, AUC-ROC, Avg. Precision.

    Parameters
    ----------
    model     : fitted sklearn estimator
    X         : feature matrix (unscaled; scaler is applied internally)
    y         : true labels (0/1)
    scaler    : if not None, applies scaler.transform(X) before predicting
    threshold : decision threshold for Precision / Recall / F1
    """
    X_in  = scaler.transform(X) if scaler is not None else X
    proba = model.predict_proba(X_in)[:, 1]
    pred  = (proba >= threshold).astype(int)

    return {
        "precision":      float(precision_score(y, pred,  zero_division=0)),
        "recall":         float(recall_score(y, pred,     zero_division=0)),
        "f1":             float(f1_score(y, pred,         zero_division=0)),
        "auc_roc":        float(roc_auc_score(y, proba)),
        "avg_precision":  float(average_precision_score(y, proba)),
    }


def roc_curve_data(
    model,
    X: np.ndarray,
    y: np.ndarray,
    scaler: StandardScaler | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (fpr, tpr, thresholds) for plotting the ROC curve."""
    X_in  = scaler.transform(X) if scaler is not None else X
    proba = model.predict_proba(X_in)[:, 1]
    return roc_curve(y, proba)


def pr_curve_data(
    model,
    X: np.ndarray,
    y: np.ndarray,
    scaler: StandardScaler | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (precision, recall, thresholds) for plotting the PR curve."""
    X_in  = scaler.transform(X) if scaler is not None else X
    proba = model.predict_proba(X_in)[:, 1]
    return precision_recall_curve(y, proba)


# ---------------------------------------------------------------------------
# Error analysis
# ---------------------------------------------------------------------------

def error_analysis(
    model,
    X: np.ndarray,
    y: np.ndarray,
    pairs: list[tuple],
    G_train: nx.Graph,
    n_examples: int = 20,
    scaler: StandardScaler | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build DataFrames of the top false positives and top false negatives.

    False positives (FP): model predicted a link — it didn't form.
        Sorted by predicted score descending (highest-confidence wrong guesses).
    False negatives (FN): a link DID form — model missed it.
        Sorted by predicted score ascending (most confidently missed links).

    Each row includes actor display names, predicted score, number of common
    neighbors (main reason for high-score predictions), and the films that
    actually connected the pair (for FN, from G_after — not computed here,
    added in the notebook from the full graph).
    """
    X_in  = scaler.transform(X) if scaler is not None else X
    proba = model.predict_proba(X_in)[:, 1]

    def display(node):
        if G_train.has_node(node):
            return G_train.nodes[node].get("display_name", node)
        return str(node)

    def cn(u, v):
        if G_train.has_node(u) and G_train.has_node(v):
            return len(list(nx.common_neighbors(G_train, u, v)))
        return 0

    df = pd.DataFrame({
        "u":               [p[0] for p in pairs],
        "v":               [p[1] for p in pairs],
        "u_name":          [display(p[0]) for p in pairs],
        "v_name":          [display(p[1]) for p in pairs],
        "y_true":          y,
        "y_prob":          proba,
        "common_neighbors": [cn(p[0], p[1]) for p in pairs],
    })

    fp = (df[df["y_true"] == 0]
          .sort_values("y_prob", ascending=False)
          .head(n_examples)
          [["u_name", "v_name", "y_prob", "common_neighbors"]]
          .reset_index(drop=True))

    fn = (df[df["y_true"] == 1]
          .sort_values("y_prob", ascending=True)
          .head(n_examples)
          [["u_name", "v_name", "y_prob", "common_neighbors"]]
          .reset_index(drop=True))

    return fp, fn


# ---------------------------------------------------------------------------
# Future predictions  (exploratory)
# ---------------------------------------------------------------------------

def predict_future_collaborations(
    model,
    G: nx.Graph,
    centralities: dict,
    svd_emb: dict | None = None,
    n2v_emb: dict | None = None,
    n_top: int = 50,
    min_degree: int = 2,
    max_candidates: int = 15_000,
    seed: int = RANDOM_SEED,
    scaler: StandardScaler | None = None,
) -> pd.DataFrame:
    """Score unconnected actor pairs and return the top-N most likely
    future collaborations as a ranked "watch list".

    Since the full non-edge set is O(n²), we restrict candidates to active
    actors (degree >= min_degree) and cap at max_candidates pairs sampled
    randomly when the eligible set is too large.

    Parameters
    ----------
    model          : fitted sklearn estimator
    G              : the current full graph
    centralities   : pre-computed centralities from compute_centralities(G)
    svd_emb / n2v_emb : optional node embeddings from G
    n_top          : number of top predictions to return
    min_degree     : minimum degree to be considered an "active" actor
    max_candidates : cap on total candidate pairs (efficiency guard)
    seed           : random seed for the candidate sample
    scaler         : optional scaler (required if model is Logistic Regression)

    Returns
    -------
    pd.DataFrame with columns:
        u_name, v_name, score, common_neighbors,
        u_degree, v_degree
    """
    rng         = np.random.default_rng(seed)
    active      = [n for n in G.nodes() if G.degree(n) >= min_degree]
    existing    = {(min(u, v), max(u, v)) for u, v in G.edges()}

    # Build candidate non-edge pairs (sample if too many)
    if len(active) * (len(active) - 1) // 2 <= max_candidates:
        candidates = [(min(u, v), max(u, v))
                      for u, v in combinations(active, 2)
                      if (min(u, v), max(u, v)) not in existing]
    else:
        seen: set = set()
        candidates = []
        attempts   = 0
        target     = min(max_candidates, len(active) ** 2 // 4)
        while len(candidates) < target and attempts < target * 10:
            i, j = rng.choice(len(active), size=2, replace=False)
            pair  = (min(active[i], active[j]), max(active[i], active[j]))
            if pair not in existing and pair not in seen:
                candidates.append(pair)
                seen.add(pair)
            attempts += 1

    if not candidates:
        return pd.DataFrame()

    X, _ = lf.build_feature_matrix(
        G, candidates, centralities,
        svd_emb=svd_emb, n2v_emb=n2v_emb,
    )

    X_in  = scaler.transform(X) if scaler is not None else X
    proba = model.predict_proba(X_in)[:, 1]

    deg   = dict(G.degree())
    df    = pd.DataFrame({
        "u":               [p[0] for p in candidates],
        "v":               [p[1] for p in candidates],
        "u_name":          [G.nodes[p[0]].get("display_name", p[0]) for p in candidates],
        "v_name":          [G.nodes[p[1]].get("display_name", p[1]) for p in candidates],
        "score":           proba,
        "common_neighbors":[len(list(nx.common_neighbors(G, p[0], p[1])))
                            for p in candidates],
        "u_degree":        [deg.get(p[0], 0) for p in candidates],
        "v_degree":        [deg.get(p[1], 0) for p in candidates],
    })

    return (df.sort_values("score", ascending=False)
              .head(n_top)
              .reset_index(drop=True))
