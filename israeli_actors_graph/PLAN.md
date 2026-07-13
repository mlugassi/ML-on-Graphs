# Israeli Actors Graph — Data Collection, Analysis & Link Prediction

## Context

This is a course assignment for "Machine Learning on Graphs" (Azrieli College) — build a co-appearance graph of Israeli-film actors from Wikipedia, analyze its evolution across three time periods, detect communities, compute centrality, and build/evaluate a temporal link-prediction model. AI-tool use is explicitly permitted by the assignment. Per the user's direction: I write and verify all code (running small-scale checks to prove each piece works), but the user drives the pace — no large-scale scrape or long run happens without their explicit go-ahead. The user will write the final PDF report and record the video themselves; my deliverable is working code, real intermediate data/results from verification runs, notebooks, and figures they can pull into the report.

This work lives in a **new subfolder in the current repo** (`ML-on-Graphs`), reusing existing conventions (`utils.py` helpers like `get_giant_component`, `print_graph_info`, `log_log_plot`; `networkx>=3.5`, `node2vec`, `torch_geometric`, `scikit-learn` already in `requirements.txt`).

Source page confirmed by inspection: `he.wikipedia.org/wiki/סרטי_קולנוע_ישראליים` is a **list article** (not a raw MediaWiki category) with ~400-450 films across decade tables (columns: year, title [wikilinked], screenwriter, director, cast, genre, awards, notes). The cast column there is capped at ~4-6 names — short of the "first 10" requirement — so per-film pages must be visited individually for the full cast (from each film's infobox, first 10 credited actors in page order).

"Israeli actor" = any actor credited in an Israeli film from this list (per user's clarification) — no separate nationality verification needed.

## Folder structure

```
israeli_actors_graph/
  README.md
  data/
    raw/            # cached API/HTML responses (gitignored — re-fetchable)
    processed/       # movies.csv, cast_edges.csv, actors_nodes.csv (checked in)
  src/
    scraping.py       # fetch list page + film pages via MediaWiki API, parse infobox cast
    graph_build.py     # build full MultiGraph, actor ID resolution/cleaning, period splitting
    graph_metrics.py    # per-period basic/connectivity/degree-distribution metrics
    communities.py       # community detection (Louvain/greedy-modularity/label-prop) + comparison
    centrality.py          # degree/betweenness/closeness/eigenvector centrality
    link_features.py     # CN, Jaccard, Adamic-Adar, Pref. Attachment, shortest path, embeddings
    link_prediction.py  # temporal train/val/test construction, model training/eval, future predictions
  notebooks/
    01_data_collection.ipynb
    02_graph_construction.ipynb
    03_temporal_period_analysis.ipynb
    04_community_detection.ipynb
    05_centrality_analysis.ipynb
    06_link_prediction.ipynb
  figures/
```
Add to root `requirements.txt`: `requests`, `beautifulsoup4`, `lxml`, `tqdm`. (Louvain is available via `networkx>=3.5`'s built-in `louvain_communities` — no extra dependency needed.)

## Phase 1 — Data collection (`scraping.py`, `01_data_collection.ipynb`)

- Fetch the list article via the MediaWiki API (`action=parse`, `he.wikipedia.org/w/api.php`) rather than scraping raw rendered pages directly — more stable, avoids template-rendering quirks, plays nicer with rate limits.
- Parse each decade table (via BeautifulSoup on the returned HTML) into rows: `year, title, wiki_slug, screenwriter, director, genre, awards, notes`.
- For each film's `wiki_slug`, fetch its own article (same API), extract cast from the infobox field (`שחקנים`/`כוכבים`), falling back to a "שחקנים" section if the infobox lacks one. Take the first 10 actors in page order, each keyed by their **Wikipedia page slug** (not raw display text) to avoid duplicate-name issues; actors without a wikilink (occasional plain-text trailing name) get a normalized-text fallback ID, flagged in a review log.
- Cache every raw API response to `data/raw/` so re-runs are resumable and don't re-hit Wikipedia.
- Be polite: custom User-Agent, small delay between requests, `tqdm` progress bar.
- **Verification before full run**: test the parser against ~5-10 known films (e.g. סאלח שבתי, אסקימו לימון) and confirm cast/year extraction is correct, then stop and report back before scraping all ~450 films.
- Output: `data/processed/movies.csv` (film metadata) and `data/processed/cast_edges.csv` (film → actor rows).

## Phase 2 — Graph construction (`graph_build.py`, `02_graph_construction.ipynb`)

- Build one canonical `nx.MultiGraph`: node = actor (attrs: slug, display name, list of films); edge = one per co-starring pair per film (attrs: title, year), so repeat collaborations across films/periods are preserved distinctly.
- Period split (half-open intervals to avoid double-counting at boundaries): **A** = year ≤ 1970, **B** = 1970 < year ≤ 1990, **C** = year > 1990. Will confirm this boundary convention is reasonable before proceeding — flag to user in case they intended different edges.
- For each period, derive a simple weighted `nx.Graph` (collapse multi-edges → `weight` = collaboration count, keep film list as edge attr) for metric computation, using only edges (films) whose year falls in that period. Node set = actors appearing in at least one such edge.
- Data cleaning: dedupe near-identical display names sharing normalized text but different slugs (log for manual spot-check), drop self-loop artifacts if any.

## Phase 3 — Per-period metrics (`graph_metrics.py`, `03_temporal_period_analysis.ipynb`)

For each of A/B/C, compute and tabulate:
- Basic: node count, edge count, density, average clustering coefficient.
- Connectivity: number of connected components, largest component size, radius & diameter **of the largest component** (undefined on disconnected graphs — reusing `get_giant_component` from `utils.py`).
- Degree distribution: histogram (`log_log_plot` from `utils.py`), mean degree, max degree.
- A comparison table/plot across A→B→C (numbers only — narrative commentary is left for the report, per the user's plan to write it themselves).

## Phase 4 — Community detection (Period C only) (`communities.py`, `04_community_detection.ipynb`)

- Run and compare 3 algorithms available without new dependencies: `nx.community.louvain_communities`, `nx.community.greedy_modularity_communities`, `nx.community.label_propagation_communities`.
- Compare: modularity score, number of communities, size distribution, pairwise agreement (e.g. normalized mutual information).
- Characterize each Louvain community (primary algorithm) using data already collected: dominant genres (from `movies.csv`), active-year range, top-degree actors as representative members.

## Phase 5 — Centrality analysis (`centrality.py`, `05_centrality_analysis.ipynb`)

- On Period C's graph (largest component where required): degree, betweenness (`k`-sampled if graph is large), closeness, eigenvector centrality (`eigenvector_centrality_numpy` for robustness on sparse/disconnected structure).
- Report top-15 actors per metric with names, and a short cross-metric comparison (e.g. high-degree vs. high-betweenness overlap).

## Phase 6 — Link prediction (`link_features.py`, `link_prediction.py`, `06_link_prediction.ipynb`)

Interpretation of the assignment's train/test design (will flag to user for confirmation before the big run):
1. **Train graph**: edges (films) with year in [1990, 2020].
2. **Temporal train/val split** within that range (e.g. train on edges before 2015, validate on new edges 2015-2020) to select the best model/features without touching the true holdout.
3. **True holdout test**: actual new edges formed in (2020, 2025] (already present in the full scraped dataset) — pairs connected there but not connected as of 2020, restricted to node pairs that already existed in the 1990-2020 graph (an actor debuting after 2020 can't be a candidate — this is a real model limitation worth flagging in the report).
4. **Negative sampling**: sample non-edges (as of 2020) that remain non-edges through 2025, balanced against positives (e.g. degree-stratified sampling to avoid a trivially easy classifier).
5. **Features** (computed only from the 1990-2020 graph, no leakage): Common Neighbors, Jaccard, Adamic-Adar, Preferential Attachment, shortest-path distance, node centrality (degree/betweenness/closeness/eigenvector) for both endpoints, `node2vec` embeddings (Hadamard/L1/L2-combined as edge features), and a simple matrix-factorization baseline (`TruncatedSVD` on the adjacency matrix) as an alternative embedding source.
6. **Models**: Logistic Regression and Random Forest (via `scikit-learn`, already a dependency) compared on validation AUC; best one carried to the holdout.
7. **Evaluation on the 2020-2025 holdout**: Precision, Recall, F1, AUC-ROC. Error analysis: top-scored false positives (predicted, didn't happen) and highest-missed false negatives (happened, scored low) with discussion of likely causes.
8. **Exploratory final step**: retrain the best model on the full graph through the latest scraped year, score all currently-unconnected pairs, and surface a ranked "watch list" of plausible future collaborations — this has no ground truth yet, framed as a speculative/discussion output for the report.

## Execution approach

I will build each phase's code and run small-scale verification (a handful of films/nodes) to prove correctness, then stop and report results before running anything at full scale (full ~450-film scrape, full graph construction, full model training). The user drives when each phase moves from "verified on a sample" to "run for real." Report and video are the user's responsibility — my output is code, cached/processed data, notebooks, and figures with factual captions/metrics, not narrative report prose.

## Verification

- Phase 1: manually check parsed cast/year against a few known film pages.
- Phase 2-3: sanity-check node/edge counts per period against manual eyeballing of the source tables (e.g. period A should be small, given few pre-1970 Israeli films).
- Phase 4-5: cross-check that top-centrality actors match well-known prolific Israeli actors as a plausibility check.
- Phase 6: confirm no data leakage (features computed strictly from pre-2020 graph), confirm holdout evaluation numbers are stable across reruns with a fixed random seed.
