"""ffanalytics' source_scrapes.R, in Python.

Ten sites publish fantasy football projections and no two of them agree, on
the numbers or on how to serve them. ffanalytics scrapes all ten and this is
that file translated: one function per source, each returning what the R's
returns -- a table per position, with the site's columns renamed to the
shared vocabulary in scrape_columns and every player carrying the MFL id
that lets the sources be compared at all.

    scrape_cbs             an HTML table, column names hidden in tooltips
    scrape_nfl             an HTML table, two header rows
    scrape_fantasysharks   a CSV, addressed by a numbered "segment"
    scrape_numberfire      two HTML tables, side by side, joined by row
    scrape_walterfootball  an Excel workbook, a sheet per position
    scrape_fleaflicker     an HTML table, twenty players at a time
    scrape_fftoday         an HTML table inside two other tables
    scrape_fantasypros     an HTML table, ids in the row's class attribute
    scrape_rtsports        JSON, a stats object per player
    scrape_espn            JSON, stats numbered rather than named
    scrape_fanduel         GraphQL, and the successor to NumberFire

A table is a list of dicts here rather than a data frame, one dict per
player, which is the shape the rest of draftsim already reads (see adp.py)
and what `--out` writes as CSV. Columns are typed the way R's `type.convert`
types them -- whole column at a time, so one "N/A" leaves the column as text
rather than making that one player's yardage a string among floats.

Two things every scrape does, because the R does:

    two seconds between pages, which is what the sites ask for and what the
    R prints a message about before it starts

    an hour's cache under DRAFTSIM_CACHE (or ~/.cache/draftsim), so that
    working on the pipeline is not repeatedly working the sites. Pass
    refresh=True, or --refresh, to insist on a fresh one

A failure on one position is caught and reported, and the other positions
still come back -- ffanalytics' `lapply_safe`, which matters more here than
it sounds: these are ten sites of varying liveliness, and two of them have
already moved. As of this writing fantasy.nfl.com redirects to nfl.com and
numberfire.com to fanduel.com, so those two scrapes come back empty however
faithfully they are translated; FanDuel's GraphQL is where NumberFire's
projections went, and that scrape works.

The one deliberate departure from the R is in mfl_ids: ffanalytics matches a
site's own player ids against a crosswalk in its binary package data before
falling back on matching by name, and that crosswalk is not published in any
readable form. Names, positions and teams do the work here. See mfl_ids for
what that costs.
"""

import argparse
import csv
import io
import json
import os
import re
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

from . import htmldom, xlsxread
from .adp import cache_dir
from .mfl_ids import get_mfl_id, scrape_week, scrape_year
from .scrape_columns import (
    CBS,
    ESPN_TEAM_NUMS,
    FANDUEL,
    FANTASYPROS,
    FANTASYSHARKS,
    FANTASYSHARKS_DUPES,
    FFTODAY,
    FLEAFLICKER,
    NFL,
    NFL_POS_IDX,
    NUMBERFIRE,
    NUMBERFIRE_IDP,
    RTS,
    RTS_POS_IDX,
    TEAM_CORRECTIONS,
    WALTERFOOTBALL,
    espn_stats,
    rename,
)

POSITIONS = ("QB", "RB", "WR", "TE", "K", "DST")
IDP = ("DL", "LB", "DB")
USER_AGENT = (
    "ffanalytics R package "
    "(https://github.com/FantasyFootballAnalytics/ffanalytics)"
)
TIMEOUT = 60
DELAY = 2  # what the sites ask between pages, and what the R waits
MAX_AGE = 60 * 60  # an hour, as ffanalytics caches a scrape
NA = ("", "NA", "N/A", "-", "—", "--")
_INTEGER = re.compile(r"[+-]?\d+$")


# ---------------------------------------------------------------- fetching


def fetch(url, headers=None, data=None, timeout=TIMEOUT):
    """The bytes at a URL, identifying ourselves as the R package does."""
    head = {"User-Agent": USER_AGENT, "Accept-Encoding": "identity"}
    head.update(headers or {})
    request = urllib.request.Request(url, data=data, headers=head)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def read_html(url, **kwargs):
    """read_html: a page, parsed."""
    raw = fetch(url, **kwargs)
    return htmldom.parse(raw.decode("utf-8", "replace"))


def read_json(url, **kwargs):
    """resp_body_json: a page that answers in JSON."""
    return json.loads(fetch(url, **kwargs))


def announce(source, position, url):
    """What the R prints as it goes, so a slow scrape says where it is."""
    print("Scraping %s %s projections from\n  %s" % (source, position, url))


# ------------------------------------------------------------------ tables


def convert(rows, skip=("id", "src_id")):
    """type.convert: type each column, whole, leaving the id columns alone.

    R types a column, not a cell, and that is worth keeping: a stray "N/A"
    in one player's row leaves the whole column as text rather than making
    that player's yardage a string where everyone else has a float. Ids stay
    text however numeric they look, because a leading zero is meaningful.
    """
    columns = {name for row in rows for name in row}
    for name in columns - set(skip):
        values = [_na(row.get(name)) for row in rows]
        typed = _column(values)
        for row, value in zip(rows, typed, strict=True):
            if name in row:
                row[name] = value
    return rows


def _na(value):
    """A cell, with the site's several ways of writing "nothing" as None."""
    if value is None:
        return None
    if isinstance(value, str):
        value = value.strip()
        return None if value in NA else value
    return value


def _column(values):
    """One column as ints, or as floats, or left as it came."""
    present = [v for v in values if v is not None]
    if not present or any(
        not isinstance(v, (str, int, float)) for v in present
    ):
        return values
    if all(_whole(v) for v in present):
        return [None if v is None else int(v) for v in values]
    try:
        return [None if v is None else float(v) for v in values]
    except (TypeError, ValueError):
        return values


def _whole(value):
    """A count rather than a rate -- what R would read as an integer."""
    if isinstance(value, bool):
        return False
    if isinstance(value, int):
        return True
    return isinstance(value, str) and _INTEGER.match(value.strip()) is not None


def positive(rows, column="site_pts"):
    """The R keeps only players a site expects to score."""
    kept = []
    for row in rows:
        points = row.get(column)
        if isinstance(points, (int, float)) and points > 0:
            kept.append(row)
    return kept


def drop_empty(rows):
    """Filter(function(x) any(!is.na(x))): drop columns nobody filled in."""
    keep = {n for row in rows for n, v in row.items() if v is not None}
    return [{n: v for n, v in row.items() if n in keep} for row in rows]


def relabel(row, mapping):
    """The same row under the shared column names, order untouched."""
    return dict(zip(rename(list(row), mapping), row.values(), strict=True))


def named(values, names):
    """`names<-`: a row of cells under the column names it belongs to."""
    if len(values) != len(names):
        raise ValueError(
            "row has %d cells, %d column names" % (len(values), len(names))
        )
    return dict(zip(names, values, strict=True))


def uniquify(names, replacements):
    """Rename repeats positionally, as the R does after its column map.

    FantasySharks calls both the rushing and the receiving column ">= 50yds"
    and the map can only claim one of them; the second is named here.
    """
    out, seen, spare = [], set(), list(replacements)
    for name in names:
        if name in seen and spare:
            out.append(spare.pop(0))
        else:
            seen.add(name)
            out.append(name)
    return out


def extract(text, pattern, groups):
    """tidyr::extract: pull columns out of one, or leave them empty."""
    match = re.match(pattern, text or "", re.DOTALL)
    if match is None:
        return dict.fromkeys(groups)
    return dict(zip(groups, match.groups(), strict=True))


def add_ids(rows, source, **kwargs):
    """Give every row its MFL id and say which site it came from."""
    ids = get_mfl_id(**kwargs) if rows else []
    for row, mfl in zip(rows, ids, strict=True):
        row["id"] = mfl
        row["data_src"] = source
    return rows


def column(rows, name):
    return [row.get(name) for row in rows]


def front(rows, names):
    """dplyr::select(id, src_id, everything()): put the keys first."""
    out = []
    for row in rows:
        ordered = {n: row.get(n) for n in names if n in row}
        ordered.update(row)
        out.append(ordered)
    return out


# ------------------------------------------------------------------ caching


def _defaults(season, week):
    return (
        scrape_year() if season is None else season,
        scrape_week(season) if week is None else week,
    )


def cached(source, positions, season, week, refresh, build):
    """An hour's worth of a scrape, kept the way ffanalytics keeps one.

    The cache is only used when it holds every position asked for, which is
    ffanalytics' rule: ask for more than last time and the whole thing is
    scraped again rather than answered in part.
    """
    path = Path(cache_dir()) / ("ffa_%s_%s_%s.json" % (source, season, week))
    if not refresh and _fresh(path):
        held = json.loads(path.read_text(encoding="utf-8"))
        if all(pos in held for pos in positions):
            print("Using the %s scrape cached under %s" % (source, path))
            return {pos: held[pos] for pos in positions}
    scraped = build()
    _write_json(path, scraped)
    return scraped


def _fresh(path):
    return path.exists() and time.time() - path.stat().st_mtime < MAX_AGE


def _write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data), encoding="utf-8")
    os.replace(tmp, path)


def each_position(positions, scrape_one, delay=DELAY):
    """lapply_safe over the positions, pausing between pages as R does.

    One position failing -- a site down, a table moved -- costs that
    position and not the scrape.
    """
    out = {}
    for i, position in enumerate(positions):
        if i:
            time.sleep(delay)
        try:
            out[position] = scrape_one(position)
        except Exception as error:  # noqa: BLE001 - R prints and carries on
            print("  %s failed: %s" % (position, error), file=sys.stderr)
            out[position] = []
    return out


# ---------------------------------------------------------------------- CBS

CBS_URL = (
    "https://www.cbssports.com/fantasy/football/stats/%s/%s/%s/"
    "projections/nonppr/"
)
CBS_TABLE = "#TableBase > div > div > table"
CBS_PLAYER = (
    "table > tbody > tr > td:nth-child(1) > "
    "span.CellPlayerName--long > span > a"
)
CBS_NAME = (
    r".*?\s{2,}[A-Z]{1,3}\s{2,}[A-Z]{2,3}\s{2,}"
    r"(.*?)\s{2,}(.*?)\s{2,}(.*)"
)


def scrape_cbs(positions=POSITIONS, season=None, week=None, refresh=False):
    """CBS: one page per position, the column names inside the tooltips."""
    season, week = _defaults(season, week)
    print("\nThe CBS scrape uses a 2 second delay between pages")
    weekly = "restofseason" if week in (0, "ros") else week

    def build():
        return each_position(
            positions, lambda pos: _cbs_position(pos, season, weekly)
        )

    return cached("cbs", positions, season, week, refresh, build)


def _cbs_position(position, season, weekly):
    url = CBS_URL % (position, season, weekly)
    announce("CBS", position, url)
    page = read_html(url)
    names = rename(_cbs_names(page), CBS)
    ids = _cbs_ids(page, position)
    body = page.select_one(CBS_TABLE + " > tbody")
    rows = [named(cells, names) for cells in body.table()]
    if position == "DST":
        rows = _cbs_dst(rows, ids)
    else:
        rows = _cbs_players(rows, ids)
    return positive(convert(rows))


def _cbs_names(page):
    """The header row, whose useful half is the tooltip under each label."""
    header = page.select_one(CBS_TABLE + " > thead > tr.TableBase-headTr")
    parts = re.split(r"\n|\t", header.text2())
    return [p for p in parts if re.search("[A-Z]", p)]


def _cbs_ids(page, position):
    if position == "DST":
        hrefs = [a.attr("href") for a in page.select("span.TeamName a")]
        return [re.sub(r".*?([A-Z]{2,3}).*", r"\1", h or "") for h in hrefs]
    hrefs = [a.attr("href") for a in page.select(CBS_PLAYER)]
    return [re.sub(r".*?([0-9]+).*", r"\1", h or "") for h in hrefs]


def _cbs_players(rows, ids):
    for row, src_id in zip(rows, ids, strict=True):
        row.update(extract(row["player"], CBS_NAME, ("player", "pos", "team")))
        row["src_id"] = src_id
    return add_ids(
        rows,
        "CBS",
        id_col=ids,
        player_name=column(rows, "player"),
        pos=column(rows, "pos"),
        team=column(rows, "team"),
    )


def _cbs_dst(rows, ids):
    teams = rename(ids, TEAM_CORRECTIONS)
    for row, team in zip(rows, teams, strict=True):
        row["team"] = team
        row["pos"] = "DST"
        row["src_id"] = None
    return add_ids(rows, "CBS", pos="DST", team=teams)


# ---------------------------------------------------------------------- NFL

NFL_URL = "https://fantasy.nfl.com/research/projections"
NFL_COUNTS = {"QB": 42, "RB": 100, "WR": 150, "TE": 60, "K": 64, "DST": 32}
NFL_NAME = r"(.*?)\s+\b(QB|RB|WR|TE|K)\b.*?([A-Z]{2,3})"


def scrape_nfl(positions=POSITIONS, season=None, week=None, refresh=False):
    """NFL.com: as the R reads it -- see the note about the redirect."""
    season, week = _defaults(season, week)
    print("\nThe NFL.com scrape uses a 2 second delay between pages")

    def build():
        return each_position(
            positions, lambda pos: _nfl_position(pos, season, week)
        )

    return cached("nfl", positions, season, week, refresh, build)


def _nfl_url(position, season, week):
    query = [
        "position=%d" % NFL_POS_IDX[position],
        "count=%d" % NFL_COUNTS[position],
        "sort=projectedPts",
        "statCategory=projectedStats",
        "statSeason=%s" % season,
    ]
    if week == 0:
        query.append("statType=seasonProjectedStats")
    else:
        query.extend(["statType=weekProjectedStats", "statWeek=%s" % week])
    return NFL_URL + "?" + "&".join(query)


def _nfl_position(position, season, week):
    url = _nfl_url(position, season, week)
    announce("NFL", position, url)
    page = read_html(url)
    head = page.select_one("table > thead")
    body = page.select_one("table > tbody")
    if head is None or body is None:
        raise ValueError("no projections table at %s" % url)
    names = _nfl_names(head)
    ids = [
        re.sub(".*=", "", a.attr("href") or "")
        for a in page.select("table td:first-child a.playerName")
    ]
    rows = [named(cells, names) for cells in body.table()]
    return _nfl_clean(rows, ids, position)


def _nfl_names(head):
    rows = head.table()
    paired = ["%s %s" % (a, b) for a, b in zip(rows[0], rows[1], strict=True)]
    return rename([p.strip() for p in paired], NFL)


def _nfl_clean(rows, ids, position):
    for row, src_id in zip(rows, ids, strict=True):
        if position == "DST":
            row["team"] = re.sub(r"\s+DEF$", "", row.get("team") or "")
            row["pos"] = "DST"
        else:
            row.update(
                extract(
                    row["player"], NFL_NAME, ("player", "pos", "team")
                )
            )
        if position in ("RB", "WR", "TE"):
            row.pop("pass_int", None)
        row.pop("opp", None)
        row["src_id"] = src_id
    rows = positive(convert(rows))
    rows = add_ids(
        rows,
        "NFL",
        id_col=column(rows, "src_id"),
        player_name=None if position == "DST" else column(rows, "player"),
        pos=column(rows, "pos"),
        team=column(rows, "team"),
    )
    order = ("id", "src_id", "player", "pos", "team")
    return drop_empty(front(rows, order))


# ------------------------------------------------------------- FantasySharks

SHARKS_URL = (
    "https://www.fantasysharks.com/apps/bert/forecasts/projections.php"
    "?csv=1&Sort=&League=-1&Position=%d&scoring=1&Segment=%d&uid=4"
)
SHARKS_POS = {
    "QB": 1,
    "RB": 2,
    "WR": 4,
    "TE": 5,
    "K": 7,
    "DST": 6,
    "DL": 8,
    "LB": 9,
    "DB": 10,
}
# The site addresses a season by a "segment" number, and there is no rule to
# it; ffanalytics keeps the list, so this keeps the list.
SHARKS_SEGMENT = {
    2026: 874,
    2025: 842,
    2024: 810,
    2023: 778,
    2022: 746,
    2021: 714,
    2020: 682,
    2019: 650,
    2018: 618,
    2017: 586,
}


def scrape_fantasysharks(
    positions=POSITIONS + IDP, season=None, week=None, refresh=False
):
    """FantasySharks: a CSV per position, if you know the segment number."""
    season, week = _defaults(season, week)
    print("\nThe FantasySharks scrape uses a 2 second delay between pages")
    segment = _sharks_segment(season, week)

    def build():
        return each_position(
            positions, lambda pos: _sharks_position(pos, segment)
        )

    return cached("fantasysharks", positions, season, week, refresh, build)


def _sharks_segment(season, week):
    year = SHARKS_SEGMENT.get(season)
    if year is None:
        raise ValueError("no FantasySharks segment for %s" % season)
    if week == "ros":
        return 813
    if week == 0:
        return year
    return year + int(week) + 8


def _sharks_position(position, segment):
    url = SHARKS_URL % (SHARKS_POS[position], segment)
    announce("FantasySharks", position, url)
    text = fetch(url).decode("utf-8", "replace")
    table = list(csv.reader(io.StringIO(text)))
    names = uniquify(rename(table[0], FANTASYSHARKS), FANTASYSHARKS_DUPES)
    names = _sharks_names(names, position)
    rows = [named(cells, names) for cells in table[1:] if any(cells)]
    for row in rows:
        row.pop("Rank", None)
        row["data_src"] = "FantasySharks"
        if position == "DST":
            row["id"] = "%04d" % int(float(row["id"]))
    return positive(convert(rows))


def _sharks_names(names, position):
    if position == "K":
        names = ["fg_att" if n == "pass_att" else n for n in names]
    if position == "DST":
        names = ["dst_int" if n == "pass_int" else n for n in names]
    if position in IDP:
        names = [re.sub(r"^(dst|pass)_", "idp_", n) for n in names]
    return names


# --------------------------------------------------------------- NumberFire

NUMBERFIRE_URL = "https://www.numberfire.com/nfl/fantasy/%s/%s"
NUMBERFIRE_POS = {
    "QB": "qb",
    "RB": "rb",
    "WR": "wr",
    "TE": "te",
    "K": "k",
    "DST": "d",
    "LB": "idp",
}
NUMBERFIRE_NAME = r"(.*?)\n.*\n.*?([A-Z]{1,3}),\s*([A-Z]{2,3})"
_INTERVAL = re.compile(r"(?<=[\d.])-")


def scrape_numberfire(
    positions=POSITIONS, season=None, week=None, refresh=False
):
    """NumberFire: two tables, side by side, joined row by row.

    numberfire.com now redirects to FanDuel's research pages, where none of
    these tables exist, so this comes back empty. `scrape_fanduel` is where
    the same projections live.
    """
    season, week = _defaults(season, week)
    print("\nThe numberFire scrape uses a 2 second delay between pages")
    site_positions = _numberfire_positions(positions)

    def build():
        scraped = each_position(
            site_positions, lambda pos: _numberfire_position(pos, week)
        )
        return _numberfire_split(scraped, positions)

    return cached("numberfire", positions, season, week, refresh, build)


def _numberfire_positions(positions):
    """The IDP page holds every defender, so it is fetched once as LB."""
    if any(pos in IDP for pos in positions):
        return [pos for pos in positions if pos not in IDP] + ["LB"]
    return list(positions)


def _numberfire_position(position, week):
    page = "remaining-projections" if week in (0, "ros") else (
        "fantasy-football-projections"
    )
    url = NUMBERFIRE_URL % (page, NUMBERFIRE_POS[position])
    announce("NumberFire", position, url)
    doc = read_html(url)
    ids = [
        (a.attr("href") or "").rstrip("/").rsplit("/", 1)[-1]
        for a in doc.select("td[class='player'] a")
    ]
    tables = doc.select("table.projection-table")
    if len(tables) < 2:
        raise ValueError("no projection tables at %s" % url)
    players = _numberfire_players(tables[0])
    stats = _numberfire_stats(tables[1], position)
    return _numberfire_join(players, stats, ids, position)


def _numberfire_players(table):
    rows = table.table()[1:]
    out = []
    for cells in rows:
        got = extract(cells[0], NUMBERFIRE_NAME, ("player", "pos", "team"))
        out.append(got)
    return out


def _numberfire_stats(table, position):
    rows = table.table()
    names = ["%s %s" % (a, b) for a, b in zip(rows[0], rows[1], strict=True)]
    names = [n.strip() for n in names]
    out = []
    for cells in rows[2:]:
        row = named(cells, names)
        if position not in ("LB", "DB"):
            row = _numberfire_interval(row)
        if position == "QB":
            row = _numberfire_split_pair(row, "Passing C/A", "/", (
                "pass_comp",
                "pass_att",
            ))
        out.append(
            {k: re.sub(r"N/A|\$|#", "", str(v)) for k, v in row.items()}
        )
    return out


def _numberfire_interval(row):
    """The confidence interval arrives as one "12.3-24.5" cell.

    Split on a hyphen that follows a digit, not on the first one there is,
    so an interval whose lower bound is negative still comes apart in the
    right place -- which is what the R's substitution is careful about too.
    """
    text = str(row.pop("numberFire CI", "") or "")
    parts = _INTERVAL.split(text, maxsplit=1) + [""]
    return {**row, "Lower": parts[0].strip(), "Upper": parts[1].strip()}


def _numberfire_split_pair(row, name, sep, parts):
    text = str(row.pop(name, "") or "")
    return {**row, **_split_pair(text, sep, parts)}


def _split_pair(text, sep, parts):
    left, _, right = text.partition(sep)
    return dict(zip(parts, (left.strip(), right.strip()), strict=True))


def _numberfire_join(players, stats, ids, position):
    rows = []
    for player, stat, src_id in zip(players, stats, ids, strict=False):
        row = {**player, **stat, "src_id": src_id}
        rows.append(row)
    mapping = NUMBERFIRE_IDP if position in IDP else NUMBERFIRE
    rows = [relabel(row, mapping) for row in rows]
    rows = add_ids(
        rows,
        "NumberFire",
        id_col=ids,
        player_name=column(rows, "player"),
        pos=column(rows, "pos"),
        team=column(rows, "team"),
    )
    rows = convert(rows)
    return positive(rows) if any("site_pts" in r for r in rows) else rows


def _numberfire_split(scraped, positions):
    """The one IDP table back into the positions that were asked for."""
    if not any(pos in IDP for pos in positions):
        return scraped
    defenders = scraped.pop("LB", [])
    out = {p: rows for p, rows in scraped.items() if p not in IDP}
    for position in positions:
        if position in IDP:
            out[position] = [r for r in defenders if r.get("pos") == position]
    return out


# ------------------------------------------------------------ WalterFootball

WALTERFOOTBALL_URL = "http://walterfootball.com/fantasy%srankingsexcel.xlsx"
WALTERFOOTBALL_SHEETS = {
    "QB": "QBs",
    "RB": "RBs",
    "WR": "WRs",
    "TE": "TEs",
    "K": "Ks",
}
WALTERFOOTBALL_KEEP = re.compile(
    r"^Pass|^Rush|^Catch|^Rec|^Reg TD$|^Int|^FG|^XP|name$|^player"
    r"|^Team$|^Pos|^Bye",
    re.IGNORECASE,
)


def scrape_walterfootball(
    positions=("QB", "RB", "WR", "TE", "K"),
    season=None,
    week=None,
    refresh=False,
):
    """WalterFootball: not a page but a workbook, a sheet per position."""
    season, week = _defaults(season, week)
    url = WALTERFOOTBALL_URL % season

    def build():
        with tempfile.NamedTemporaryFile(
            suffix=".xlsx", delete=False
        ) as handle:
            handle.write(fetch(url))
            path = handle.name
        try:
            return each_position(
                positions,
                lambda pos: _walterfootball_position(path, pos, url),
                delay=0,
            )
        finally:
            os.unlink(path)

    return cached("walterfootball", positions, season, week, refresh, build)


def _walterfootball_position(path, position, url):
    announce("WalterFootball", position, url)
    sheet = xlsxread.read_sheet(path, WALTERFOOTBALL_SHEETS[position])
    rows = xlsxread.to_records(sheet)
    rows = [_walterfootball_row(row, position) for row in rows]
    rows = _walterfootball_keep(rows)
    rows = [relabel(row, WALTERFOOTBALL) for row in rows]
    rows = add_ids(
        rows,
        "WalterFootball",
        player_name=column(rows, "player"),
        pos=column(rows, "pos"),
    )
    return _walterfootball_tds(convert(rows))


def _walterfootball_row(row, position):
    """One player: a full name, and the misspelling the R patches."""
    first = row.get("First Name", "")
    if position in ("QB", "WR") and first == "Marcua":
        first = "Marcus"
    last = row.get("Last Name", "")
    out = {"Player": ("%s %s" % (first, last)).strip()}
    out.update(row)
    out["First Name"] = first
    out["Bye"] = row.get("Bye", row.get("BYE", ""))
    out.pop("BYE", None)
    out["position"] = out.pop("Pos", "")
    return out


def _walterfootball_keep(rows):
    """The columns the R selects, minus any nobody in the sheet filled in."""
    filled = {n for row in rows for n, v in row.items() if str(v).strip()}
    keep = [
        n
        for n in (rows[0] if rows else {})
        if n in filled
        and (WALTERFOOTBALL_KEEP.search(n) or n in ("position", "Player"))
        and n not in ("Last Name", "First Name")
    ]
    return [{n: row.get(n) for n in keep} for row in rows]


def _walterfootball_tds(rows):
    """Split the one "REG TD" column back into rushing and receiving.

    WalterFootball counts a player's touchdowns without saying how he scored
    them, so ffanalytics apportions them by yards -- a back with all his
    yards on the ground keeps all his touchdowns there. With only one of the
    two yardage columns in the sheet there is nothing to apportion, and the
    column is simply renamed.
    """
    names = {n for row in rows for n in row}
    if "reg_tds" not in names:
        return rows
    has = [n for n in ("rush_yds", "rec_yds") if n in names]
    if len(has) == 2:
        return [_walterfootball_share(row) for row in rows]
    if len(has) == 1:
        target = has[0].replace("_yds", "_tds")
        for row in rows:
            row[target] = row.pop("reg_tds")
    return rows


def _walterfootball_share(row):
    rush, rec = row.get("rush_yds"), row.get("rec_yds")
    tds = row.pop("reg_tds")
    if None in (rush, rec, tds):
        row["rush_tds"], row["rec_tds"] = None, None
    elif rush + rec == 0:
        row["rush_tds"], row["rec_tds"] = 0, 0
    else:
        row["rush_tds"] = rush / (rush + rec) * tds
        row["rec_tds"] = rec / (rush + rec) * tds
    return row


# --------------------------------------------------------------- FleaFlicker

FLEAFLICKER_URL = (
    "https://www.fleaflicker.com/nfl/leaders?week=%s&statType=7&sortMode=7"
    "&position=%d&tableOffset=%d"
)
FLEAFLICKER_POS = {
    "QB": 4,
    "RB": 1,
    "WR": 2,
    "TE": 8,
    "K": 16,
    "DST": 256,
    "DE": 2048,
    "DT": 64,
    "LB": 128,
    "CB": 512,
    "S": 1024,
}
FLEAFLICKER_PAGES = {
    "K": 2,
    "DST": 2,
    "QB": 2,
    "DT": 4,
    "TE": 5,
    "DE": 6,
    "LB": 6,
    "S": 6,
    "RB": 6,
    "CB": 6,
    "WR": 6,
}
FLEAFLICKER_DST = r"(.*)\s+D/ST\s+([A-Z]{2,3}).*?(\d+).*"
FLEAFLICKER_NAME = r"(.*?)\s+(.*?)\s+(.*?)\s+(.*?)\s+.*(\d+)\)$"
# The site's own IDP positions, and the ffanalytics groups they roll up into.
FLEAFLICKER_GROUPS = {"DL": ("DE", "DT"), "DB": ("CB", "S")}


def scrape_fleaflicker(
    positions=POSITIONS + IDP, season=None, week=None, refresh=False
):
    """FleaFlicker: twenty players a page, until the points run out."""
    season, week = _defaults(season, week)
    wanted = _fleaflicker_positions(positions)

    def build():
        scraped = each_position(
            wanted, lambda pos: _fleaflicker_position(pos, week)
        )
        return _fleaflicker_group(scraped, positions)

    return cached("fleaflicker", positions, season, week, refresh, build)


def _fleaflicker_positions(positions):
    out = []
    for position in positions:
        out.extend(FLEAFLICKER_GROUPS.get(position, (position,)))
    return out


def _fleaflicker_position(position, week):
    rows, offset = [], 0
    for page in range(FLEAFLICKER_PAGES[position]):
        if page:
            time.sleep(DELAY)
        url = FLEAFLICKER_URL % (week, FLEAFLICKER_POS[position], offset)
        if not page:
            announce("FleaFlicker", position, url)
        found = _fleaflicker_page(url, position)
        rows.extend(found)
        points = [r.get("site_pts") for r in found]
        thin = [p for p in points if isinstance(p, (int, float))]
        if len(found) < 20 or (thin and min(thin) <= 1):
            break
        offset += 20
    return convert(rows)


def _fleaflicker_page(url, position):
    doc = read_html(url)
    table = doc.select_one("#body-center-main table")
    if table is None:
        raise ValueError("no leaders table at %s" % url)
    ids = [
        re.sub(r".*-(\d+)$", r"\1", a.attr("href") or "")
        for a in doc.select("a.player-text")
    ]
    rows = [r for r in table.table() if not _fleaflicker_footer(r)]
    names = _fleaflicker_names(rows[0], rows[1], position)
    body = [named(cells, names) for cells in rows[2:]]
    body = drop_empty(convert([_na_row(r) for r in body]))
    return _fleaflicker_players(body, ids, position)


def _fleaflicker_footer(row):
    return any(re.search("Previous.*Next", cell or "") for cell in row)


def _fleaflicker_names(group, sub, position):
    names = ["%s %s" % (a, b) for a, b in zip(group, sub, strict=True)]
    names = [re.sub(r"Week\s+\d+|Projected", "", n) for n in names]
    names = [re.sub(r"\s+", " ", n).strip() for n in names]
    if position == "K":
        for index, name in zip(
            (9, 10, 12, 13),
            ("fg_att", "fg_pct", "xp_att", "xp_pct"),
            strict=True,
        ):
            if index < len(names):
                names[index] = name
    names = rename(names, FLEAFLICKER)
    return [n or "...%d" % i for i, n in enumerate(names)]


def _na_row(row):
    return {n: _na(v) for n, v in row.items()}


def _fleaflicker_players(rows, ids, position):
    for row, src_id in zip(rows, ids, strict=False):
        row["src_id"] = src_id
    if position == "DST":
        return _fleaflicker_dst(rows, ids)
    firsts, lasts = [], []
    for row in rows:
        got = extract(
            re.sub(r"^Q(?=[A-Z])", "", row.get("player") or ""),
            FLEAFLICKER_NAME,
            ("first_name", "last_name", "pos", "team", "bye"),
        )
        firsts.append(got["first_name"])
        lasts.append(got["last_name"])
        name = "%s %s" % (got["first_name"], got["last_name"])
        row["player"] = name.strip()
        row.update({k: v for k, v in got.items() if "name" not in k})
    return add_ids(
        rows,
        "FleaFlicker",
        id_col=ids,
        player_name=column(rows, "player"),
        first=firsts,
        last=lasts,
        pos=column(rows, "pos"),
        team=column(rows, "team"),
    )


def _fleaflicker_dst(rows, ids):
    """A defense is one cell: "Bills D/ST BUF (7)"."""
    for row in rows:
        row.update(
            extract(
                row.get("player"), FLEAFLICKER_DST, ("player", "team", "bye")
            )
        )
        row["pos"] = "DST"
    return add_ids(
        rows,
        "FleaFlicker",
        id_col=ids,
        pos="DST",
        player_name=column(rows, "team"),
        team=column(rows, "team"),
    )


def _fleaflicker_group(scraped, positions):
    """DE and DT are one DL here, CB and S one DB, as ffanalytics has it."""
    out = {}
    for position in positions:
        parts = FLEAFLICKER_GROUPS.get(position, (position,))
        rows, seen = [], set()
        for part in parts:
            for row in scraped.get(part, []):
                key = row.get("id") or id(row)
                if key not in seen:
                    seen.add(key)
                    rows.append({**row, "pos": position})
        out[position] = rows
    return out


# ------------------------------------------------------------------ FFToday

FFTODAY_SEASON = (
    "https://www.fftoday.com/rankings/playerproj.php?Season=%s&PosID=%d"
    "&LeagueID=1&order_by=FFPts&sort_order=DESC&cur_page=%d"
)
FFTODAY_WEEK = (
    "https://www.fftoday.com/rankings/playerwkproj.php?Season=%s&GameWeek=%s"
    "&PosID=%d&LeagueID=1&order_by=FFPts&sort_order=DESC&cur_page=%d"
)
FFTODAY_POS = {
    "QB": 10,
    "RB": 20,
    "WR": 30,
    "TE": 40,
    "DL": 50,
    "LB": 60,
    "DB": 70,
    "K": 80,
    "DST": 99,
}
FFTODAY_PAGES = {
    "QB": 1,
    "TE": 1,
    "K": 1,
    "DST": 1,
    "RB": 2,
    "WR": 3,
    "DL": 3,
    "DB": 3,
    "LB": 3,
}


def scrape_fftoday(
    positions=POSITIONS + IDP, season=None, week=None, refresh=False
):
    """FFToday: an old page, and the projections table is the third one."""
    season, week = _defaults(season, week)
    print("\nThe FFToday scrape uses a 2 second delay between pages")
    if week and int(week) > 18:
        week = int(week) + 2
    wanted = list(positions)
    if week:
        wanted = [p for p in wanted if p not in ("DST",) + IDP]

    def build():
        return each_position(
            wanted, lambda pos: _fftoday_position(pos, season, week)
        )

    return cached("fftoday", positions, season, week, refresh, build)


def _fftoday_url(position, season, week, page):
    if not week:
        return FFTODAY_SEASON % (season, FFTODAY_POS[position], page)
    return FFTODAY_WEEK % (season, week, FFTODAY_POS[position], page)


def _fftoday_position(position, season, week):
    rows = []
    for page in range(FFTODAY_PAGES[position]):
        time.sleep(DELAY)
        url = _fftoday_url(position, season, week, page)
        if not page:
            announce("FFToday", position, url)
        rows.extend(_fftoday_page(url, position, week))
    return rows


def _fftoday_page(url, position, week):
    doc = read_html(url)
    ids = _fftoday_ids(doc, position)
    table = doc.select_one("table table table")
    if table is None:
        raise ValueError("no projections table at %s" % url)
    cells = [[c.replace(",", "").replace("%", "") for c in r]
             for r in table.table()]
    names = _fftoday_names(cells[0], cells[1], position)
    rows = [named(r, names) for r in cells[2:]]
    return _fftoday_clean(rows, ids, position, week)


def _fftoday_ids(doc, position):
    if position == "DST":
        hrefs = [
            a.attr("href") or ""
            for a in doc.select("a[href *='stats/players']")
        ]
        found = [re.sub(r".*?=(\d{4}).*", r"\1", h) for h in hrefs]
        return [f for f in found if re.search(r"\d{4}", f)]
    hrefs = [
        a.attr("href") or ""
        for a in doc.select("a[href *='stats/players/']")
    ]
    return [h.split("?")[0].rstrip("/").split("/")[-2] for h in hrefs]


def _fftoday_names(group, sub, position):
    sub = [re.sub(r"^(.*?)\n.*", r"\1", cell, flags=re.DOTALL) for cell in sub]
    names = ["%s %s" % (a, b) for a, b in zip(group, sub, strict=True)]
    names = rename([n.strip() for n in names], FFTODAY)
    if position in IDP:
        names = [re.sub(r"(dst|pass)_", "idp_", n) for n in names]
    return names


def _fftoday_clean(rows, ids, position, week):
    for row, src_id in zip(rows, ids, strict=True):
        row["pos"] = position
        row["src_id"] = src_id
        row.pop("chg", None)
        if week:
            row["opp"] = (row.get("opp") or "").replace("@", "")
        if "bye" in row:
            row["bye"] = (row["bye"] or "").replace("-", "")
    rows = convert(rows)
    dst = position == "DST"
    # A defense's row names the team in full -- "Houston Texans" -- which is
    # how the player table names it too, so it goes in as the player's name.
    # The R passes only the site's id here and leans on its crosswalk, which
    # is the one thing this cannot borrow; see mfl_ids.
    return add_ids(
        rows,
        "FFToday",
        id_col=ids,
        player_name=column(rows, "team" if dst else "player"),
        team=None if dst else column(rows, "team"),
        pos=column(rows, "pos"),
    )


# --------------------------------------------------------------- FantasyPros

FANTASYPROS_URL = "https://www.fantasypros.com/nfl/projections/%s.php?week=%s"
FANTASYPROS_NAME = r"(.*)\s+([A-Z]{2,3})"


def scrape_fantasypros(
    positions=POSITIONS, season=None, week=None, refresh=False
):
    """FantasyPros: a clean table, with the player's id in the row class."""
    season, week = _defaults(season, week)
    print("\nThe FantasyPros scrape uses a 2 second delay between pages")
    scrape = week if week else "draft"

    def build():
        return each_position(
            positions, lambda pos: _fantasypros_position(pos, scrape)
        )

    return cached("fantasypros", positions, season, week, refresh, build)


def _fantasypros_position(position, week):
    url = FANTASYPROS_URL % (position.lower(), week)
    announce("FantasyPros", position, url)
    doc = read_html(url)
    names = _fantasypros_names(doc, position)
    ids = _fantasypros_ids(doc)
    body = doc.select_one("table > tbody")
    if body is None:
        raise ValueError("no projections table at %s" % url)
    rows = [
        named([c.replace(",", "") for c in cells], names)
        for cells in body.table()
    ]
    return _fantasypros_clean(rows, ids, position)


def _fantasypros_names(doc, position):
    head = doc.select_one("table > thead")
    if position in ("K", "DST"):
        names = head.text2().split("\t")
    else:
        rows = head.table()
        names = [
            "%s %s" % (a, b)
            for a, b in zip(rows[0], rows[1], strict=True)
        ]
    return rename([n.strip() for n in names], FANTASYPROS)


def _fantasypros_ids(doc):
    classes = [tr.attr("class") for tr in doc.select("table > tbody > tr")]
    found = [
        re.sub(r".*?(\d{4,6}).*", r"\1", c) for c in classes if c is not None
    ]
    return [f for f in found if f.isdigit()]


def _fantasypros_clean(rows, ids, position):
    for row, src_id in zip(rows, ids, strict=True):
        row["src_id"] = src_id
        row["pos"] = position
        if position != "DST":
            row.update(
                extract(row["player"], FANTASYPROS_NAME, ("player", "team"))
            )
    dst = position == "DST"
    # As with FFToday, a defense arrives named in full and that is enough to
    # match it; the R passes the site's id alone and needs its crosswalk.
    rows = add_ids(
        rows,
        "FantasyPros",
        id_col=ids,
        player_name=column(rows, "player"),
        team=None if dst else column(rows, "team"),
        pos=column(rows, "pos"),
    )
    return positive(convert(rows))


# ----------------------------------------------------------------- RTSports

RTSPORTS_URL = (
    "https://www.freedraftguide.com/football/"
    "draft-guide-rankings-provider.php?POS=%d"
)
_INFO = ("player_id", "stats_id", "name", "nfl_team")


def scrape_rtsports(
    positions=POSITIONS, season=None, week=0, refresh=False
):
    """RTSports: JSON, and season-long only -- there is no weekly page."""
    season, week = _defaults(season, week)
    if week:
        raise ValueError("RTSports projections are only available for week 0")
    print("\nThe RTSports scrape uses a 5 second delay between pages")

    def build():
        return each_position(positions, _rtsports_position, delay=5)

    return cached("rtsports", positions, season, week, refresh, build)


def _rtsports_position(position):
    url = RTSPORTS_URL % RTS_POS_IDX[position]
    announce("RTSports", position, url)
    players = list(read_json(url).get("player_list", {}).values())
    stats = _rtsports_drop_constant([p.get("stats") or {} for p in players])
    rows = [
        {**_rtsports_info(p), **{k: v for k, v in s.items() if k not in _INFO}}
        for p, s in zip(players, stats, strict=True)
    ]
    if position in ("RB", "WR", "TE"):
        rows = _rtsports_pass_atts(rows)
    rows = [relabel(row, RTS) for row in rows]
    rows = convert(rows)
    rows = rows if position == "DST" else positive(rows)
    return _rtsports_ids(rows, position)


def _rtsports_info(player):
    """Who the player is, which the stats object repeats and leaves blank."""
    return {name: player.get(name) for name in _INFO}


def _rtsports_drop_constant(rows):
    """Drop the stats every player shares.

    The site sends every stat it has for every position, so a quarterback
    arrives with a column of zeroes for field goals and a blank one for the
    name the player record already carries. What varies is what was
    projected.
    """
    if not rows:
        return rows
    keys = list(dict.fromkeys([k for row in rows for k in row]))
    varies = [k for k in keys if len({str(r.get(k)) for r in rows}) > 1]
    return [{k: r.get(k) for k in varies if k in r} for r in rows]


def _rtsports_pass_atts(rows):
    """A receiver's passing attempts are all zero, so they get dropped."""
    if any("pass_yds" in r for r in rows) and not any(
        "pass_atts" in r for r in rows
    ):
        for row in rows:
            row["pass_atts"] = 0
    return rows


def _rtsports_ids(rows, position):
    stats_ids = column(rows, "stats_id")
    for row in rows:
        row["pos"] = position
        row.pop("stats_id", None)
        row["src_id"] = None if row.get("src_id") is None else str(
            row["src_id"]
        )
    rows = add_ids(
        rows,
        "RTSports",
        id_col=stats_ids,
        player_name=column(rows, "player"),
        team=column(rows, "team"),
        pos=column(rows, "pos"),
    )
    return front(rows, ("id", "src_id", "pos", "data_src"))


# --------------------------------------------------------------------- ESPN

ESPN_DEFAULTS = (
    "https://lm-api-reads.fantasy.espn.com/apis/v3/games/ffl/seasons/%s"
    "/segments/0/leaguedefaults/3?scoringPeriodId=0&view=kona_player_info"
)
ESPN_LEAGUE = (
    "https://lm-api-reads.fantasy.espn.com/apis/v3/games/ffl/seasons/%s"
    "/segments/0/leagues/%s?scoringPeriodId=%s&view=kona_player_info"
)
ESPN_SLOTS = {
    "QB": 0,
    "RB": 2,
    "WR": 4,
    "TE": 6,
    "K": 17,
    "DST": 16,
    "DT": 8,
    "DE": 9,
    "LB": 10,
    "DL": 11,
    "CB": 12,
    "DB": 14,
}
# The IDP pages are only served in the context of a league, and ffanalytics
# supplies this one when the caller has none of their own.
ESPN_LEAGUE_ID = 1595759
ESPN_LIMITS = {
    "QB": 42,
    "RB": 100,
    "WR": 150,
    "TE": 60,
    "K": 35,
    "DST": 32,
    "DL": 90,
    "DB": 60,
    "LB": 60,
}


def scrape_espn(
    positions=POSITIONS,
    season=None,
    week=None,
    refresh=False,
    espn_league_id=ESPN_LEAGUE_ID,
):
    """ESPN: the API its own projections page calls, filter header and all."""
    season, week = _defaults(season, week)
    print("\nThe ESPN scrape uses a 2 second delay between pages")
    wanted = list(positions)
    if any(pos in IDP for pos in wanted) and espn_league_id is None:
        print("Must provide a valid espn_league_id to get DL, LB, and DB")
        wanted = [pos for pos in wanted if pos not in IDP]

    def build():
        return each_position(
            wanted,
            lambda pos: _espn_position(pos, season, week, espn_league_id),
        )

    return cached("espn", positions, season, week, refresh, build)


def _espn_filter(position, season, week, limit):
    """The X-Fantasy-Filter header, which is where the query really lives."""
    split = 0 if week == 0 else 1
    return json.dumps(
        {
            "players": {
                "filterSlotIds": {"value": [ESPN_SLOTS[position]]},
                "filterStatsForSourceIds": {"value": [1]},
                "filterStatsForSplitTypeIds": {"value": [split]},
                "sortAppliedStatTotal": {
                    "sortAsc": False,
                    "sortPriority": 3,
                    "value": "11%s%s" % (season, week),
                },
                "sortDraftRanks": {
                    "sortPriority": 2,
                    "sortAsc": True,
                    "value": "PPR",
                },
                "sortPercOwned": {"sortAsc": False, "sortPriority": 4},
                "limit": limit,
                "offset": 0,
                "filterRanksForScoringPeriodIds": {"value": [2]},
                "filterRanksForRankTypes": {"value": ["PPR"]},
                "filterRanksForSlotIds": {
                    "value": [0, 2, 4, 6, 17, 16, 15]
                },
                "filterStatsForTopScoringPeriodIds": {
                    "value": 2,
                    "additionalValue": [
                        "00%s" % season,
                        "10%s" % season,
                        "11%s%s" % (season, week),
                        "02%s" % season,
                    ],
                },
            }
        },
        separators=(",", ":"),
    )


def _espn_position(position, season, week, league_id):
    idp = position in IDP
    url = (
        ESPN_LEAGUE % (season, league_id, week)
        if idp
        else ESPN_DEFAULTS % season
    )
    announce("ESPN", position, "https://fantasy.espn.com/football/players"
             "/projections")
    stats = espn_stats("dst" if idp else "idp")
    headers = {
        "Accept": "application/json",
        "X-Fantasy-Source": "kona",
        "X-Fantasy-Filter": _espn_filter(position, season, week, ESPN_LIMITS[
            position
        ]),
    }
    players = read_json(url, headers=headers).get("players", [])
    rows = [_espn_row(p, position, stats) for p in players]
    rows = [r for r in rows if r is not None]
    return _espn_ids(convert(rows), position)


def _espn_row(entry, position, stats):
    player = entry.get("player") or {}
    projections = player.get("stats") or []
    if not projections:
        return None
    row = {}
    for number, value in (projections[0].get("stats") or {}).items():
        if number in stats:
            row[stats[number]] = round(value)
    row["src_id"] = str(entry.get("id"))
    row["player"] = player.get("fullName")
    row["team"] = ESPN_TEAM_NUMS.get(str(player.get("proTeamId")))
    row["pos"] = position
    return row


def _espn_ids(rows, position):
    dst = position == "DST"
    rows = add_ids(
        rows,
        "ESPN",
        id_col=None if dst else column(rows, "src_id"),
        player_name=None if dst else column(rows, "player"),
        pos=column(rows, "pos"),
        team=column(rows, "team"),
    )
    return front(rows, ("id", "src_id", "pos", "player", "team"))


# ------------------------------------------------------------------ FanDuel

FANDUEL_URL = "https://fdresearch-api.fanduel.com/graphql"
FANDUEL_GROUPS = {
    "NFL_SKILL": ("QB", "RB", "WR", "TE"),
    "NFL_KICKER": ("K",),
    "NFL_D_ST": ("DST",),
}
FANDUEL_QUERY = """
query GetProjections($input: ProjectionsInput!) {
  getProjections(input: $input) {
    ... on NflSkill {
      player { numberFireId name position }
      team { numberFireId name abbreviation }
      salary value completionsAttempts passingYards passingTouchdowns
      interceptionsThrown rushingAttempts rushingYards rushingTouchdowns
      receptions targets receivingYards receivingTouchdowns fantasy
      positionRank overallRank opponentDefensiveRank
    }
    ... on NflKicker {
      player { numberFireId name position }
      team { numberFireId name abbreviation }
      salary value extraPointsAttempted extraPointsMade fieldGoalsAttempted
      fieldGoalsMade fieldGoalsMade0To19 fieldGoalsMade20To29
      fieldGoalsMade30To39 fieldGoalsMade40To49 fieldGoalsMade50Plus
      fantasy positionRank opponentDefensiveRank
    }
    ... on NflDefenseSt {
      player { numberFireId name position }
      team { numberFireId name abbreviation }
      salary value pointsAllowed yardsAllowed sacks interceptions
      fumblesRecovered touchdowns fantasy positionRank
      opponentOffensiveRank
    }
  }
}
"""


def scrape_fanduel(
    positions=POSITIONS, season=None, week=None, refresh=False
):
    """FanDuel: one GraphQL call per position group, and NumberFire's heir."""
    season, week = _defaults(season, week)
    kind = "WEEKLY" if week else "REMAINING"
    print(
        "\nScraping FanDuel projections for %s..." % ", ".join(positions)
    )
    groups = [
        group
        for group, members in FANDUEL_GROUPS.items()
        if any(pos in members for pos in positions)
    ]

    def build():
        rows = []
        for group in groups:
            rows.extend(_fanduel_group(group, kind))
        return _fanduel_split(rows, positions)

    return cached("fanduel", positions, season, week, refresh, build)


def _fanduel_group(group, kind):
    payload = json.dumps(
        {
            "query": FANDUEL_QUERY,
            "variables": {
                "input": {"type": kind, "position": group, "sport": "NFL"}
            },
            "operationName": "GetProjections",
        }
    ).encode()
    headers = {"Content-Type": "application/json", "Accept": "*/*",
               "Origin": "https://www.fanduel.com"}
    answer = json.loads(fetch(FANDUEL_URL, headers=headers, data=payload))
    found = (answer.get("data") or {}).get("getProjections") or []
    return [_fanduel_row(entry) for entry in found]


def _fanduel_row(entry):
    flat = {}
    for name, value in entry.items():
        if isinstance(value, dict):
            flat.update({"%s_%s" % (name, k): v for k, v in value.items()})
        else:
            flat[name] = value
    if "completionsAttempts" in flat:
        pair = _split_pair(str(flat.pop("completionsAttempts") or ""), "/", (
            "pass_comp",
            "pass_att",
        ))
        flat.update(pair)
    row = relabel(flat, FANDUEL)
    row["pos"] = "DST" if row.get("pos") == "D" else row.get("pos")
    row["src_id"] = None if row.get("src_id") is None else str(row["src_id"])
    return row


def _fanduel_split(rows, positions):
    rows = convert(rows)
    rows = add_ids(
        rows,
        "FanDuel",
        id_col=column(rows, "src_id"),
        player_name=column(rows, "player"),
        team=column(rows, "team"),
        pos=column(rows, "pos"),
    )
    rows = front(rows, ("id", "src_id", "player", "pos", "team"))
    out = {}
    for position in positions:
        wanted = [r for r in rows if r.get("pos") == position]
        for row in wanted:
            row.pop("salary", None)
            row.pop("value", None)
        out[position] = drop_empty(wanted)
    return out


# ------------------------------------------------------- not scraped at all


def scrape_fantasyfootballnerd(**kwargs):
    """Not implemented in ffanalytics either -- the R says so and stops."""
    del kwargs
    print(
        "\nThe FantasyFootballNerd scrape is not implemeted yet"
        "--we are working on it"
    )
    return {}


def scrape_fantasydata(**kwargs):
    """Behind a paywall, as the R notes."""
    del kwargs
    print(
        "\nThe FantasyData scrape is behind a paywall and is not supported "
        "at this time"
    )
    return {}


def scrape_yahoo(**kwargs):
    """Deprecated: Yahoo publishes FantasyPros' projections now."""
    del kwargs
    print(
        "\nThe Yahoo scrape is no longer supported because they now use "
        "FantasyPros projections"
    )
    return {}


SOURCES = {
    "cbs": scrape_cbs,
    "espn": scrape_espn,
    "fanduel": scrape_fanduel,
    "fantasydata": scrape_fantasydata,
    "fantasyfootballnerd": scrape_fantasyfootballnerd,
    "fantasypros": scrape_fantasypros,
    "fantasysharks": scrape_fantasysharks,
    "fftoday": scrape_fftoday,
    "fleaflicker": scrape_fleaflicker,
    "nfl": scrape_nfl,
    "numberfire": scrape_numberfire,
    "rtsports": scrape_rtsports,
    "walterfootball": scrape_walterfootball,
    "yahoo": scrape_yahoo,
}

# When each source is worth asking. ffanalytics carries a `draft` and a
# `weekly` flag on every scrape's signature and scrape_data reads them off
# before it calls one: FleaFlicker and FanDuel are weekly, WalterFootball
# and RTSports publish a season and never a week.
#
# For FleaFlicker this matters more than it sounds. Ask it for a season and
# it answers with one game, and one game's points land in the same average
# as everyone else's season -- which is what a board quietly reading nine
# points low is made of. FanDuel is marked weekly too, though what it
# returns for week 0 is rest-of-season and on the right scale; it is left
# out of a draft board because ffanalytics leaves it out, which is what
# keeps this board the same board as the published one.
AVAILABLE = {
    "cbs": (True, True),
    "espn": (True, True),
    "fanduel": (False, True),
    "fantasydata": (True, True),
    "fantasyfootballnerd": (True, True),
    "fantasypros": (True, True),
    "fantasysharks": (True, True),
    "fftoday": (True, True),
    "fleaflicker": (False, True),
    "nfl": (True, True),
    "numberfire": (True, True),
    "rtsports": (True, False),
    "walterfootball": (True, False),
    "yahoo": (True, True),
}

# What each source has to offer, so asking for a position it never carries
# is skipped rather than fetched and thrown away.
COVERAGE = {
    # The three that scrape nothing are listed so that asking for one still
    # gets the R's explanation of why it is not there.
    "fantasydata": POSITIONS,
    "fantasyfootballnerd": POSITIONS,
    "yahoo": POSITIONS,
    "cbs": POSITIONS,
    "espn": POSITIONS + IDP,
    "fanduel": POSITIONS,
    "fantasypros": POSITIONS,
    "fantasysharks": POSITIONS + IDP,
    "fftoday": POSITIONS + IDP,
    "fleaflicker": POSITIONS + IDP,
    "nfl": POSITIONS,
    "numberfire": POSITIONS + IDP,
    "rtsports": POSITIONS,
    "walterfootball": ("QB", "RB", "WR", "TE", "K"),
}


def scrape(sources, positions, season=None, week=None, refresh=False):
    """Every source asked for, each with the positions it actually has.

    A source that does not publish for the week in question is skipped
    rather than asked and averaged in, which is scrape_data's rule.
    """
    week = scrape_week(season) if week is None else week
    out = {}
    for source in _sources(sources):
        wanted = [p for p in positions if p in COVERAGE.get(source, ())]
        if not wanted or not _publishes(source, week):
            continue
        try:
            out[source] = SOURCES[source](
                positions=wanted, season=season, week=week, refresh=refresh
            )
        except Exception as error:  # noqa: BLE001 - one site is not the run
            print("%s failed: %s" % (source, error), file=sys.stderr)
            out[source] = {}
    return out


def _publishes(source, week):
    """Does this source publish for that week? scrape_data's own check."""
    draft, weekly = AVAILABLE.get(source, (True, True))
    if week == 0 and not draft:
        print("\nDraft data not available for %s" % source)
        return False
    if week and not weekly:
        print("\nWeekly data not available for %s" % source)
        return False
    return True


def _sources(sources):
    """NumberFire is FanDuel now, and ffanalytics substitutes it too."""
    if "numberfire" not in sources:
        return list(sources)
    print(
        "\nHeads up! NumberFire is now FanDuel... Using FanDuel for scrape"
    )
    kept = [s for s in sources if s != "numberfire"]
    return kept if "fanduel" in kept else kept + ["fanduel"]


def write_csv(rows, path):
    """One position of one source, as a CSV with every column it has."""
    names = list(dict.fromkeys([n for row in rows for n in row]))
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=names)
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Scrape fantasy football projections, as ffanalytics does"
    )
    parser.add_argument(
        "--source",
        nargs="+",
        default=[
            "cbs", "espn", "fanduel", "fantasyfootballnerd", "fantasypros",
            "fantasysharks", "fftoday", "fleaflicker", "nfl", "rtsports",
            "walterfootball",
        ],
        choices=sorted(SOURCES),
        help="which sites to scrape",
    )
    parser.add_argument(
        "--pos", nargs="+", default=list(POSITIONS), help="which positions"
    )
    parser.add_argument("--season", type=int, help="default: this season")
    parser.add_argument("--week", type=int, help="0 for season-long")
    parser.add_argument(
        "--refresh", action="store_true", help="ignore the hour's cache"
    )
    parser.add_argument("--out", help="write a CSV per source and position")
    args = parser.parse_args(argv)

    scraped = scrape(
        args.source, args.pos, args.season, args.week, args.refresh
    )
    for source, tables in sorted(scraped.items()):
        for position, rows in tables.items():
            matched = len([r for r in rows if r.get("id")])
            print(
                "%-16s %-4s %4d players, %4d with an id"
                % (source, position, len(rows), matched)
            )
            if args.out and rows:
                write_csv(
                    rows, Path(args.out) / ("%s_%s.csv" % (source, position))
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
