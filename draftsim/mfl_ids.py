"""One id for a player, whatever the site that just named him calls him.

Every source in source_scrapes has its own id space -- CBS numbers Josh Allen
2181054, FFToday 16228, FantasySharks 13589 -- so a scrape is six lists of
strangers until they are keyed to something shared. ffanalytics keys them to
MyFantasyLeague's id, in `get_mfl_id`, and this is that function.

The reference it matches against is the player table ffanalytics publishes,

    https://s3.us-east-2.amazonaws.com/ffanalytics/packagedata/player_table.csv

which is fetched once and kept under DRAFTSIM_CACHE (or ~/.cache/draftsim)
for a day, so a scrape of six sites costs one download and a machine that is
offline still has yesterday's names.

Matching is by name, position and team, all of it flattened first --
uppercased so `D/ST` and `JAX` can be corrected to `DST` and `JAC`, then
lowercased, stripped of a trailing Jr/Sr/III/Defense, then stripped of
punctuation and spaces, so that `Marvin Harrison Jr.`, `Marvin Harrison Jr`
and `MARVIN HARRISON` all arrive as `marvinharrison`. Then five passes, each
looser than the last:

    name + position + team,  last + position + team,  name + team,
    name + position,  first + position + team

A pass only fills in the players still unmatched, and only where the key is
unique on both sides -- if two players in the reference share it, neither is
taken, which is what keeps the two Josh Allens apart rather than guessing.
Defenses skip all of that: a DST is matched on team alone.

One difference from the R, and it is worth stating plainly. ffanalytics also
carries a crosswalk of site ids -- `player_ids`, cbs_id to mfl id and so on
-- and tries it before any of this. That table lives in the package's binary
`sysdata.rda` and is not published in any form this can read, so the
crosswalk is a parameter here (`crosswalk=`) rather than a built-in, and with
none supplied every source goes down the name-matching path. That is the same
path R takes for every player its crosswalk misses, which on a fresh scrape
is every rookie; what is lost is the handful the crosswalk would have caught
and a name cannot -- a player two sites spell differently, or a name shared
by two active players at the same position.
"""

import csv
import io
import json
import os
import re
import time
import urllib.request
from datetime import date, datetime, timedelta
from pathlib import Path

from .adp import cache_dir
from .scrape_columns import POS_CORRECTIONS, TEAM_CORRECTIONS

PLAYER_TABLE = (
    "https://s3.us-east-2.amazonaws.com/ffanalytics/packagedata/"
    "player_table.csv"
)
SCHEDULE = (
    "https://api.myfantasyleague.com/%d/export?TYPE=nflSchedule&W=ALL&JSON=1"
)
USER_AGENT = (
    "ffanalytics R package "
    "(https://github.com/FantasyFootballAnalytics/ffanalytics)"
)
TIMEOUT = 60
MAX_AGE = 24 * 60 * 60  # a day; the table changes about that often
SEASON_STARTS = 4  # before April, "this season" is still last calendar year

_SUFFIX = re.compile(r"\s+(defense|jr|sr|[iv]+)\.?$")
_PUNCT = re.compile(r"[!-/:-@\[-`{-~]+|\s+")

# The passes get_mfl_id makes, in the order it makes them.
COMBOS = (
    ("player_name", "pos", "team"),
    ("last", "pos", "team"),
    ("player_name", "team"),
    ("player_name", "pos"),
    ("first", "pos", "team"),
)

_TABLE = None
_WEEKS = {}


def flatten(value):
    """A name, team or position reduced to what a match is made on."""
    if value is None:
        return None
    text = str(value).upper()
    text = POS_CORRECTIONS.get(text, text)
    text = TEAM_CORRECTIONS.get(text, text)
    return _PUNCT.sub("", _SUFFIX.sub("", text.lower()))


def fetch(url, timeout=TIMEOUT):
    """The bytes at a URL, asking as the R package asks."""
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def player_table(refresh=False):
    """ffanalytics' player table: every player, with his MFL id.

    Kept for a day under the cache, and a stale copy is better than none if
    the fetch fails, so an outage costs accuracy rather than the scrape.
    """
    global _TABLE
    if _TABLE is not None and not refresh:
        return _TABLE
    path = Path(cache_dir()) / "ffa_player_table.csv"
    fresh = path.exists() and time.time() - path.stat().st_mtime < MAX_AGE
    if refresh or not fresh:
        try:
            raw = fetch(PLAYER_TABLE)
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            tmp.write_bytes(raw)
            os.replace(tmp, path)
        except OSError:
            if not path.exists():
                raise
    text = path.read_text(encoding="utf-8", errors="replace")
    _TABLE = list(csv.DictReader(io.StringIO(text)))
    return _TABLE


def _reference():
    """The player table, flattened the same way the scraped names are."""
    rows = []
    for row in player_table():
        first = flatten(row["first_name"])
        last = flatten(row["last_name"])
        rows.append(
            {
                "id": row["id"],
                "player_name": flatten(
                    "%s %s" % (row["first_name"], row["last_name"])
                ),
                "first": first,
                "last": last,
                "pos": flatten(row["position"]),
                "team": flatten(row["team"]),
            }
        )
    return rows


def _unique(reference, combo):
    """A key -> id lookup, with any key two players share thrown out."""
    seen, out = set(), {}
    for row in reference:
        key = "".join([row[part] or "" for part in combo])
        if key in seen:
            out.pop(key, None)
        else:
            seen.add(key)
            out[key] = row["id"]
    return out


def _teams(reference):
    """Team -> id for defenses, first row wins as `match()` does.

    The player table lists the 32 defenses before anybody else, which is why
    matching a DST on team alone lands on the defense and not on the first
    guard who plays for them.
    """
    out = {}
    for row in reference:
        if row["team"]:
            out.setdefault(row["team"], row["id"])
    return out


def _columns(player_name, first, last, pos, team, length):
    """The scraped side, flattened, with what was not given filled in."""
    if player_name is not None:
        names = _vector(player_name, length)
        if first is None:
            first = [re.sub(r"\s+.*$", "", n or "") for n in names]
        if last is None:
            last = [re.sub(r"^.*?\s+", "", n or "") for n in names]
    given = {
        "player_name": player_name,
        "first": first,
        "last": last,
        "pos": pos,
        "team": team,
    }
    out = {}
    for name, values in given.items():
        if values is not None:
            out[name] = [flatten(v) for v in _vector(values, length)]
    return out


def _vector(value, length):
    """A column of `length`, recycling a single value as R does."""
    if isinstance(value, (str, int, float)) or value is None:
        return [value] * length
    values = list(value)
    return values * length if len(values) == 1 < length else values


def get_mfl_id(
    id_col=None,
    player_name=None,
    first=None,
    last=None,
    pos=None,
    team=None,
    crosswalk=None,
):
    """MFL ids for one scraped table, or None where nothing matched.

    `id_col` is the site's own ids, useful only with a `crosswalk` from that
    site's id space to MFL's; without one it is ignored and every player goes
    through the name matching, exactly as R's does for the players its
    crosswalk misses.
    """
    columns = [id_col, player_name, first, last, pos, team]
    length = max([_length(c) for c in columns] + [1])
    ids = [None] * length
    if crosswalk and id_col is not None:
        ids = [crosswalk.get(str(i)) for i in _vector(id_col, length)]
        if all(i is not None for i in ids):
            return ids
    info = _columns(player_name, first, last, pos, team, length)
    if not info:
        return ids
    reference = _reference()
    _defenses(ids, info, reference)
    for combo in COMBOS:
        if all(part in info for part in combo):
            _pass(ids, info, _unique(reference, combo), combo)
    return ids


def _defenses(ids, info, reference):
    """A defense is its team, so it is matched on the team and nothing else."""
    if "pos" not in info or "team" not in info:
        return
    teams = _teams(reference)
    for i, position in enumerate(info["pos"]):
        if position == "dst" and ids[i] is None:
            ids[i] = teams.get(info["team"][i])


def _pass(ids, info, lookup, combo):
    """One pass of matching, filling in only the players still unmatched."""
    for i, found in enumerate(ids):
        if found is None:
            key = "".join([info[part][i] or "" for part in combo])
            ids[i] = lookup.get(key)


def _length(value):
    if value is None:
        return 1
    if isinstance(value, (str, int, float)):
        return 1
    return len(list(value))


def scrape_year(today=None):
    """get_scrape_year: the season we are in, January to March excepted."""
    today = today or date.today()
    if today.month < SEASON_STARTS:
        return today.year - 1
    return today.year


def scrape_week(season=None, today=None):
    """get_scrape_week: how many weeks have started, so 0 before the season.

    R keeps the week's start dates in the package's binary data; they are a
    day after the previous week's last kickoff, which is how ffanalytics
    builds them in the first place, from MFL's schedule. So this asks MFL the
    same question, once per season per run -- ten scrapes in a row should not
    be ten schedule downloads. If the schedule cannot be had, the answer is 0
    -- the season-long projections -- which is the one this project wants.
    """
    today = today or date.today()
    season = season or scrape_year(today)
    if (season, today) in _WEEKS:
        return _WEEKS[(season, today)]
    _WEEKS[(season, today)] = week = _scrape_week(season, today)
    return week


def _scrape_week(season, today):
    try:
        raw = json.loads(fetch(SCHEDULE % season))
        weeks = raw["fullNflSchedule"]["nflSchedule"]
    except (OSError, ValueError, KeyError):
        return 0
    starts, previous = [], None
    for week in weeks:
        games = week.get("matchup")
        if not games:
            continue
        if isinstance(games, dict):
            games = [games]
        kickoffs = [int(g["kickoff"]) for g in games]
        first = datetime.fromtimestamp(min(kickoffs)).date()
        last = datetime.fromtimestamp(max(kickoffs)).date()
        starts.append(previous or first - timedelta(days=7))
        previous = last + timedelta(days=1)
    return sum(1 for start in starts if today >= start)
