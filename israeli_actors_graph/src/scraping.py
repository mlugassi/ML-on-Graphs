"""Wikipedia data collection for the Israeli actors graph.

Fetches the Israeli-films list article via the MediaWiki API and parses its
decade tables into a movies DataFrame (year, title, wiki slug, crew, genre).
Per-film cast extraction (visiting each film's own page) is a separate step.
"""
import os
import re
import time
from urllib.parse import unquote, urlparse

import requests
from bs4 import BeautifulSoup

WIKI_API_URL = "https://he.wikipedia.org/w/api.php"
LIST_PAGE_TITLE = "סרטי קולנוע ישראליים"
USER_AGENT = "IsraeliActorsGraphResearch/1.0 (educational course project; contact via GitHub)"
RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
REQUEST_DELAY_SECONDS = 1.0
MAX_RETRIES = 5

_session = requests.Session()
_session.headers.update({"User-Agent": USER_AGENT})


def _cache_path(page_title: str) -> str:
    safe_name = re.sub(r"[^\w\-]", "_", page_title)
    return os.path.join(RAW_DIR, f"{safe_name}.html")


def fetch_page_html(page_title: str, use_cache: bool = True) -> str:
    """Fetch the rendered HTML body of a Wikipedia page via the MediaWiki API."""
    cache_path = _cache_path(page_title)
    if use_cache and os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            return f.read()

    params = {
        "action": "parse",
        "page": page_title,
        "prop": "text",
        "format": "json",
        "formatversion": "2",
    }

    for attempt in range(MAX_RETRIES):
        response = _session.get(WIKI_API_URL, params=params, timeout=30)
        if response.status_code == 429:
            wait = float(response.headers.get("Retry-After", 2 ** attempt * 2))
            time.sleep(wait)
            continue
        response.raise_for_status()
        break
    else:
        response.raise_for_status()

    payload = response.json()
    if "error" in payload:
        raise ValueError(f"MediaWiki API error for '{page_title}': {payload['error']}")
    html = payload["parse"]["text"]

    os.makedirs(RAW_DIR, exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        f.write(html)
    time.sleep(REQUEST_DELAY_SECONDS)
    return html


def _slug_from_href(href: str) -> str | None:
    """Extract the canonical page-title slug from an internal wiki link href."""
    if not href or href.startswith("#"):
        return None
    path = urlparse(href).path
    if not path.startswith("/wiki/"):
        return None
    slug = unquote(path[len("/wiki/"):])
    if ":" in slug.split("/")[0] and slug.split(":")[0] in {
        "קטגוריה", "Category", "תבנית", "Template", "קובץ", "File", "עזרה", "Help",
    }:
        return None
    return slug


def _parse_cell_links(cell) -> list[dict]:
    """Parse a table cell into an ordered list of {name, slug} entries.

    Handles both wikilinked names and occasional trailing plain-text names,
    splitting on commas/newlines outside of links.
    """
    entries = []
    seen_texts = set()
    for a in cell.find_all("a"):
        name = a.get_text(strip=True)
        if not name:
            continue
        slug = _slug_from_href(a.get("href", ""))
        entries.append({"name": name, "slug": slug})
        seen_texts.add(name)

    # Plain-text fallback: names in the cell text not covered by a link.
    full_text = cell.get_text(separator=",", strip=True)
    for part in re.split(r"[,\n]", full_text):
        name = part.strip()
        if name and name not in seen_texts and not any(name in e["name"] or e["name"] in name for e in entries):
            entries.append({"name": name, "slug": None})
            seen_texts.add(name)

    return entries


def parse_movie_list_tables(html: str) -> list[dict]:
    """Parse all wikitables on the list page into movie records."""
    soup = BeautifulSoup(html, "lxml")
    tables = soup.find_all("table", class_="wikitable")

    movies = []
    for table in tables:
        header_cells = [th.get_text(strip=True) for th in table.find_all("tr")[0].find_all(["th", "td"])]
        col_index = {name: i for i, name in enumerate(header_cells)}

        def find_col(*keywords):
            for name, i in col_index.items():
                if any(kw in name for kw in keywords):
                    return i
            return None

        year_col = find_col("שנה", "שנת")
        title_col = find_col("שם") or find_col("סרט")
        writer_col = find_col("תסריט")
        director_col = find_col("בימוי", "במאי")
        cast_col = find_col("שחקנ")
        genre_col = find_col("ז'אנר", "סוגה")
        awards_col = find_col("פרס")
        notes_col = find_col("הער")

        rows = table.find_all("tr")[1:]
        for row in rows:
            cells = row.find_all(["td", "th"])
            if not cells or title_col is None or title_col >= len(cells):
                continue

            title_cell = cells[title_col]
            title_links = _parse_cell_links(title_cell)
            title_text = title_cell.get_text(strip=True)
            title_slug = title_links[0]["slug"] if title_links else None

            if not title_text:
                continue

            year_text = cells[year_col].get_text(strip=True) if year_col is not None and year_col < len(cells) else ""
            year_match = re.search(r"\d{4}", year_text)
            year = int(year_match.group()) if year_match else None

            movies.append({
                "title": title_text,
                "wiki_slug": title_slug,
                "year": year,
                "screenwriters": _parse_cell_links(cells[writer_col]) if writer_col is not None and writer_col < len(cells) else [],
                "directors": _parse_cell_links(cells[director_col]) if director_col is not None and director_col < len(cells) else [],
                "cast_from_list": _parse_cell_links(cells[cast_col]) if cast_col is not None and cast_col < len(cells) else [],
                "genre": cells[genre_col].get_text(strip=True) if genre_col is not None and genre_col < len(cells) else "",
                "awards": cells[awards_col].get_text(strip=True) if awards_col is not None and awards_col < len(cells) else "",
                "notes": cells[notes_col].get_text(strip=True) if notes_col is not None and notes_col < len(cells) else "",
            })

    return movies


def collect_movie_list(use_cache: bool = True) -> list[dict]:
    """Fetch and parse the full Israeli-films list page into movie records."""
    html = fetch_page_html(LIST_PAGE_TITLE, use_cache=use_cache)
    return parse_movie_list_tables(html)


CAST_FIELD_KEYWORDS = ("שחקנ", "כוכב")
MAX_CAST_SIZE = 10


def _split_br_separated_links(cell) -> list[dict]:
    """Parse a `<td>` whose entries are separated by `<br>` (infobox style),
    falling back to comma/newline splitting for any trailing plain text.
    """
    return _parse_cell_links(cell)


def _extract_cast_from_infobox(soup: BeautifulSoup) -> list[dict] | None:
    infobox = soup.find("table", class_="infobox")
    if infobox is None:
        return None
    for row in infobox.find_all("tr"):
        header = row.find("th")
        cell = row.find("td")
        if header is None or cell is None:
            continue
        header_text = header.get_text(strip=True)
        if any(kw in header_text for kw in CAST_FIELD_KEYWORDS):
            entries = _split_br_separated_links(cell)
            # Infobox cast fields sometimes hold a placeholder like "ראו למטה"
            # (see below) pointing at a cast list elsewhere in the article,
            # rather than actual names. If nothing resolved to a real
            # wikilink, treat it as unresolved so the section fallback runs.
            if entries and not any(e["slug"] for e in entries):
                return None
            return entries
    return None


def _extract_cast_from_section(soup: BeautifulSoup) -> list[dict] | None:
    """Fallback: find a 'שחקנים' section heading and parse the following list/paragraph."""
    heading = None
    for tag in soup.find_all(["h2", "h3"]):
        heading_text = tag.get_text(strip=True)
        if any(kw in heading_text for kw in CAST_FIELD_KEYWORDS):
            heading = tag
            break
    if heading is None:
        return None

    entries = []
    seen_texts = set()
    for sibling in heading.find_all_next():
        if sibling.name in ("h2", "h3"):
            break
        if sibling.name == "li":
            for a in sibling.find_all("a"):
                name = a.get_text(strip=True)
                if name and name not in seen_texts:
                    entries.append({"name": name, "slug": _slug_from_href(a.get("href", ""))})
                    seen_texts.add(name)
        if len(entries) >= MAX_CAST_SIZE:
            break
    return entries or None


def extract_film_cast(html: str) -> dict:
    """Extract up to MAX_CAST_SIZE cast members from a film's own article HTML.

    Returns {"cast": [{"name", "slug"}, ...], "source": "infobox"|"section"|"none"}.
    """
    soup = BeautifulSoup(html, "lxml")

    cast = _extract_cast_from_infobox(soup)
    source = "infobox"
    if not cast:
        cast = _extract_cast_from_section(soup)
        source = "section"
    if not cast:
        return {"cast": [], "source": "none"}

    return {"cast": cast[:MAX_CAST_SIZE], "source": source}


def collect_film_cast(wiki_slug: str, use_cache: bool = True) -> dict:
    """Fetch a film's own page and extract its cast."""
    page_title = wiki_slug.replace("_", " ")
    html = fetch_page_html(page_title, use_cache=use_cache)
    return extract_film_cast(html)


CAST_EDGES_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "cast_edges.csv")
EXTRACTION_LOG_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "cast_extraction_log.csv")


def build_cast_edges_from_movie_list(movies_df) -> "pd.DataFrame":
    """Build cast_edges rows directly from the cast column already parsed off
    the list page (movies_df['cast_from_list']) -- no per-film page visits.

    This is the default/fast path: the list page's cast column is shorter
    than a film's own infobox (a handful of names rather than up to 10), but
    it requires zero additional HTTP requests since it's already been
    fetched and parsed as part of collect_movie_list().

    A film's own article page (wiki_slug) is NOT required here -- the cast
    column comes from the list table itself, which is populated regardless
    of whether the film has its own article. Films without a wiki_slug get
    a synthetic movie identifier (title+year) instead of being dropped.
    """
    import json

    import pandas as pd

    rows = []
    for _, movie in movies_df.iterrows():
        movie_id = movie["wiki_slug"] if pd.notna(movie["wiki_slug"]) else f"__title__:{movie['title']}__{movie['year']}"
        try:
            cast = json.loads(movie["cast_from_list"]) if isinstance(movie["cast_from_list"], str) else []
        except json.JSONDecodeError:
            cast = []
        for position, actor in enumerate(cast):
            rows.append({
                "movie_slug": movie_id,
                "movie_title": movie["title"],
                "year": movie["year"],
                "genre": movie["genre"],
                "directors": movie["directors"],
                "cast_source": "list_table",
                "billing_position": position,
                "actor_name": actor["name"],
                "actor_slug": actor["slug"],
            })

    return pd.DataFrame(rows)


def collect_all_film_casts(movies_df, checkpoint_every: int = 25, resume: bool = True) -> "pd.DataFrame":
    """Scrape cast lists for every unique film with a valid wiki_slug.

    Resumable: films whose wiki_slug already appears in the on-disk
    cast_edges.csv (from a prior run) are skipped. Progress is checkpointed
    to disk every `checkpoint_every` films so an interrupted run loses at
    most that many films of work (the underlying per-page HTML is cached
    regardless, so re-fetching is cheap either way).
    """
    import pandas as pd

    films = movies_df[movies_df["wiki_slug"].notna()].drop_duplicates(subset="wiki_slug")

    done_slugs = set()
    edge_rows = []
    if resume and os.path.exists(CAST_EDGES_PATH):
        existing = pd.read_csv(CAST_EDGES_PATH)
        edge_rows = existing.to_dict("records")
        done_slugs = set(existing["movie_slug"].unique())

    log_rows = []
    if resume and os.path.exists(EXTRACTION_LOG_PATH):
        log_rows = pd.read_csv(EXTRACTION_LOG_PATH).to_dict("records")
        done_slugs |= set(r["movie_slug"] for r in log_rows)

    todo = films[~films["wiki_slug"].isin(done_slugs)]
    print(f"{len(films)} films total, {len(done_slugs)} already done, {len(todo)} to scrape.")

    os.makedirs(os.path.dirname(CAST_EDGES_PATH), exist_ok=True)

    for i, (_, movie) in enumerate(todo.iterrows(), start=1):
        slug = movie["wiki_slug"]
        try:
            result = collect_film_cast(slug)
            for position, actor in enumerate(result["cast"]):
                edge_rows.append({
                    "movie_slug": slug,
                    "movie_title": movie["title"],
                    "year": movie["year"],
                    "genre": movie["genre"],
                    "directors": movie["directors"],
                    "cast_source": result["source"],
                    "billing_position": position,
                    "actor_name": actor["name"],
                    "actor_slug": actor["slug"],
                })
            status = "ok" if result["cast"] else "no_cast_found"
            log_rows.append({"movie_slug": slug, "title": movie["title"], "status": status, "n_cast": len(result["cast"]), "source": result["source"]})
        except Exception as e:
            log_rows.append({"movie_slug": slug, "title": movie["title"], "status": f"error: {e}", "n_cast": 0, "source": ""})

        if i % checkpoint_every == 0 or i == len(todo):
            pd.DataFrame(edge_rows).to_csv(CAST_EDGES_PATH, index=False, encoding="utf-8-sig")
            pd.DataFrame(log_rows).to_csv(EXTRACTION_LOG_PATH, index=False, encoding="utf-8-sig")
            print(f"[{i}/{len(todo)}] checkpointed: {len(edge_rows)} cast rows so far.")

    return pd.DataFrame(edge_rows)


if __name__ == "__main__":
    import sys

    import pandas as pd

    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")

    records = collect_movie_list()
    df = pd.DataFrame(records)
    print(f"Parsed {len(df)} movie rows from {df['title'].nunique()} unique titles.")
    print(df[["title", "wiki_slug", "year", "genre"]].head(15).to_string(index=False))
