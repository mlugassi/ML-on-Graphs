"""Build the Israeli actors co-appearance graph from scraped cast data.

Node attributes are derived ONLY from film-page data already collected
(genres acted in, lead/supporting billing, directors worked with) -- no
separate scrape of individual actors' own Wikipedia pages is done, per
project decision.
"""
import json
import re
from collections import Counter

import networkx as nx
import pandas as pd

LEAD_BILLING_CUTOFF = 2  # billing_position < this => counted as a "lead" credit

PERIOD_BOUNDARIES = {
    "A": (None, 1970),   # year <= 1970
    "B": (1970, 1990),   # 1970 < year <= 1990
    "C": (1990, None),   # year > 1990
}


def period_of_year(year: int) -> str:
    if year <= 1970:
        return "A"
    if year <= 1990:
        return "B"
    return "C"


def _normalize_name(name: str) -> str:
    """Normalize a plain-text actor name for exact-match deduplication."""
    name = name.strip()
    name = re.sub(r"\s+", " ", name)
    name = name.replace("׳", "'").replace("״", '"')
    return name


def _actor_id(actor_slug, actor_name: str) -> str:
    """Canonical node key: wiki slug when available, else a normalized-name
    fallback key (exact-match dedup only -- no fuzzy matching, no visiting
    the actor's own page, per project decision)."""
    if isinstance(actor_slug, str) and actor_slug:
        return actor_slug
    return f"__name__:{_normalize_name(actor_name)}"


def _parse_directors(directors_json) -> list[dict]:
    if not isinstance(directors_json, str) or not directors_json:
        return []
    try:
        return json.loads(directors_json)
    except (json.JSONDecodeError, TypeError):
        return []


def load_cast_edges(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.dropna(subset=["year", "actor_name"]).copy()
    df["year"] = df["year"].astype(int)
    df["period"] = df["year"].apply(period_of_year)
    return df


def build_full_graph(cast_edges: pd.DataFrame) -> nx.MultiGraph:
    """Build the canonical actor graph: one node per actor, one edge per
    co-starring pair per film (so repeat collaborations across films are
    preserved as distinct parallel edges, each carrying that film's info).
    """
    G = nx.MultiGraph()

    for movie_slug, film_rows in cast_edges.groupby("movie_slug"):
        film_rows = film_rows.sort_values("billing_position")
        year = int(film_rows["year"].iloc[0])
        title = film_rows["movie_title"].iloc[0]
        genre = film_rows["genre"].iloc[0] if pd.notna(film_rows["genre"].iloc[0]) else ""
        directors = _parse_directors(film_rows["directors"].iloc[0])
        director_names = [d["name"] for d in directors]

        cast_in_film = []
        for _, row in film_rows.iterrows():
            actor_id = _actor_id(row["actor_slug"], row["actor_name"])
            is_lead = row["billing_position"] < LEAD_BILLING_CUTOFF
            cast_in_film.append((actor_id, row["actor_name"], row["billing_position"], is_lead))

            if not G.has_node(actor_id):
                G.add_node(
                    actor_id,
                    display_name=row["actor_name"],
                    credits=[],
                )
            G.nodes[actor_id]["credits"].append({
                "movie_slug": movie_slug,
                "title": title,
                "year": year,
                "genre": genre,
                "billing_position": int(row["billing_position"]),
                "is_lead": is_lead,
                "directors": director_names,
            })

        for i in range(len(cast_in_film)):
            for j in range(i + 1, len(cast_in_film)):
                a_id, a_name, a_pos, a_lead = cast_in_film[i]
                b_id, b_name, b_pos, b_lead = cast_in_film[j]
                if a_id == b_id:
                    continue  # same person credited twice under different links
                G.add_edge(a_id, b_id, movie_slug=movie_slug, title=title, year=year, genre=genre)

    for node, data in G.nodes(data=True):
        credits = data["credits"]
        data["n_films"] = len(credits)
        data["genres"] = dict(Counter(c["genre"] for c in credits if c["genre"]))
        data["directors_worked_with"] = sorted({d for c in credits for d in c["directors"]})
        data["n_lead"] = sum(1 for c in credits if c["is_lead"])
        data["n_supporting"] = sum(1 for c in credits if not c["is_lead"])
        data["active_years"] = sorted({c["year"] for c in credits})
        data["first_year"] = min(data["active_years"])
        data["last_year"] = max(data["active_years"])

    return G


def period_subgraph(G: nx.MultiGraph, period: str) -> nx.Graph:
    """Simple weighted graph for one period: nodes = actors with >=1 credit
    in the period (isolated actors included, per project decision); edges =
    collaborations in films from that period, collapsed with weight =
    collaboration count and a list of the contributing films.
    """
    assert period in PERIOD_BOUNDARIES

    nodes = [
        n for n, data in G.nodes(data=True)
        if any(period_of_year(c["year"]) == period for c in data["credits"])
    ]

    H = nx.Graph()
    for n in nodes:
        H.add_node(n, **G.nodes[n])

    for u, v, edata in G.edges(data=True):
        if period_of_year(edata["year"]) != period:
            continue
        if H.has_edge(u, v):
            H[u][v]["weight"] += 1
            H[u][v]["films"].append({"title": edata["title"], "year": edata["year"], "movie_slug": edata["movie_slug"]})
        else:
            H.add_edge(u, v, weight=1, films=[{"title": edata["title"], "year": edata["year"], "movie_slug": edata["movie_slug"]}])

    return H


if __name__ == "__main__":
    import os
    import sys

    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")

    cast_edges_path = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "cast_edges.csv")
    cast_edges = load_cast_edges(cast_edges_path)
    print(f"Loaded {len(cast_edges)} cast rows across {cast_edges['movie_slug'].nunique()} films.")

    G = build_full_graph(cast_edges)
    print(f"Full graph: {G.number_of_nodes()} actors, {G.number_of_edges()} collaboration edges.")

    for period in ["A", "B", "C"]:
        H = period_subgraph(G, period)
        isolated = sum(1 for n in H.nodes if H.degree(n) == 0)
        print(f"Period {period}: {H.number_of_nodes()} actors ({isolated} isolated), {H.number_of_edges()} unique collaborations.")
