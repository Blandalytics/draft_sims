"""Projections: the projection sites scraped and reconciled, not kept as CSVs.

The numbers are ffanalytics', and they used to be fetched by driving the
ffanalytics R package itself -- an R program handed to Rscript, handing two
tables back on its standard output. They are now built here: source_scrapes
does what its `scrape_data()` does and calc_projections what its
`projections_table()` does, so nothing in this project needs R installed.
What comes back is unchanged:

    points   one row per player -- projected points, their spread across the
             sources, VOR against the baselines, rank, tier, and the name,
             team and position that go with the id
    stats    the component stats behind those points, each with the standard
             deviation across sources, which is what vor_draft_sim samples
             from to give every simulated draft its own season

Both come out of the same call, `projections_table()`, the second with
`raw_stats=True`. That is worth knowing because it is not obvious: the
stat-level table with its per-source standard deviations is not a separate
function, it is the same aggregation returned before it is scored.

Scoring comes from scoring.py -- Yahoo's default, 0.5 PPR, unless the league
says otherwise -- and is handed to the aggregation, so the run carries its own
rules rather than depending on a saved object. The same object scores the
sampled stats in vor_draft_sim, which is the point of it being one object: the
rules also decide which stat columns come back at all, since a category worth
zero points is never aggregated.

A scrape takes a few minutes and hits up to nine sites per position, so the
result is cached under DRAFTSIM_CACHE (or ~/.cache/draftsim) as JSON, keyed by
season, week and the scoring, and is looked for in this order:

    the local cache, if this machine has already built or fetched it
    the copy published at PUBLISHED, which is that same JSON committed to the
      project: no wait at all, and the sites need not be reachable. Only the
      default scoring is published
    a fresh scrape
    failing all of that -- a scoring nobody has published, and sites that
      cannot be reached: the published component stats, scored here under the
      league's own rules. See rescore(), which says what that costs

`--refresh-projections` skips both caches and insists on the scrape, which is
the only way to get numbers newer than what is published. Whatever arrives is
kept locally, so twelve worker processes cost one fetch between them, and a
scrape that fails falls back on any cache rather than leaving you nothing.

One column ffanalytics does not have is `pid`, the id the ADP board uses --
its own ids come from a different registry. That join is made here instead, on
normalised name and position against the scraped board, and only where the
match is unique. Checked against a hand-made mapping of 477 players it agreed
on 417 and contradicted none; the rest are players with no ADP at all, deep
enough that nobody drafted them in the window, and they are simply left
without a market, exactly as combined_draft already expects.
"""

import argparse
import contextlib
import csv
import io
import json
import os
import re
import sys
import unicodedata
import urllib.request
from collections import namedtuple
from datetime import date

from . import calc_projections, source_scrapes
from .adp import add_adp_args, cache_dir
from .adp import path_from_args as adp_path_from_args
from .scoring import Scoring, add_scoring_args
from .scoring import from_args as scoring_from_args

POSITIONS = ("QB", "RB", "WR", "TE", "K", "DST")
SEASON_STARTS = 3  # from March, "this season" means this calendar year
TIMEOUT = 60  # fetching the published cache; the scrapers keep their own

# The sites worth asking. ffanalytics lists three more -- NFL.com and
# NumberFire, which both now redirect away from the pages it reads, and
# FantasyData, which is behind a paywall -- and asking them costs a request
# per position for nothing. NumberFire's projections are FanDuel's now.
#
# Not all nine are asked every time: FleaFlicker and FanDuel publish the
# coming week rather than the season, so a draft board is built from the
# other seven, and a weekly one leaves out WalterFootball and RTSports
# instead. source_scrapes.AVAILABLE is where that is decided.
SOURCES = (
    "cbs",
    "espn",
    "fanduel",
    "fantasypros",
    "fantasysharks",
    "fftoday",
    "fleaflicker",
    "rtsports",
    "walterfootball",
)

# The ranks VOR is measured from. These are the baselines this project has
# always passed ffanalytics, not its defaults, and they are not the
# replacement levels draftsim drafts against either -- league.py derives
# those from the league's own shape. They set the `rank` column's ordering.
BASELINE = {
    "QB": 10,
    "RB": 20,
    "WR": 20,
    "TE": 10,
    "K": 3,
    "DST": 3,
    "DL": 10,
    "LB": 10,
    "DB": 10,
}

# Where this project publishes the cache it built. Same JSON the local cache
# holds, under the same filename, so a borrowed laptop or a Colab notebook
# has projections to draft on without waiting three minutes for them.
PUBLISHED = (
    "https://raw.githubusercontent.com/Blandalytics/draft_sims/main/data/"
)
# how the ADP board spells the two positions that are not skill positions,
# and the columns of the board itself, which is read positionally
ADP_POS = {"DST": "TDSP", "K": "TK"}
TEAM_UNITS = frozenset(ADP_POS.values())
PID, PLAYER, TEAM, POS = 1, 2, 3, 4

# the board's team codes against ffanalytics', where the two differ
ADP_TEAM = {
    "ARI": "ARZ",
    "GBP": "GB",
    "JAC": "JAX",
    "KCC": "KC",
    "LAR": "LA",
    "LVR": "LV",
    "NEP": "NE",
    "NOS": "NO",
    "SFO": "SF",
    "TBB": "TB",
}
SUFFIX = re.compile(r"\b(jr|sr|ii|iii|iv|v)\b\.?", re.I)
# a cache file's name, and the scoring key in it, which the default leaves off
CACHE_NAME = re.compile(r"^projections_(\d+)_w(\d+)(?:_([0-9a-f]{8}))?$")

Projections = namedtuple("Projections", "points stats season week scoring")


def season_now(today=None):
    """The season ffanalytics would call current."""
    d = today or date.today()
    return d.year if d.month >= SEASON_STARTS else d.year - 1


def scrape(season, week=0, scoring=None, quiet=False, refresh=False):
    """Scrape the sites and reconcile them into the two tables.

    This is the work the R program used to do: one call to ffanalytics'
    `scrape_data()` and two to its `projections_table()`, which are now
    source_scrapes.scrape and calc_projections.projections_table. The
    scrapers narrate themselves page by page, which is worth watching from a
    terminal and worth swallowing when the draft tool is drawing over it, so
    `quiet` keeps their chatter and prints only the line that matters.

    `refresh` reaches the scrapers too: they keep an hour of their own, and
    a caller who asked for fresher numbers than the projections cache holds
    did not mean fresher by up to an hour.
    """
    scoring = scoring or Scoring()
    if not quiet:
        print(
            "projections: scraping %d week %d, a few minutes..."
            % (season, week),
            flush=True,
        )
    hush = (
        contextlib.redirect_stdout(io.StringIO())
        if quiet
        else contextlib.nullcontext()
    )
    with hush:
        scraped = source_scrapes.scrape(
            SOURCES, POSITIONS, season, week, refresh
        )
        points = _reconcile(scraped, scoring, season, week, False)
        stats = _reconcile(scraped, scoring, season, week, True)
    if not points or not stats:
        raise RuntimeError(
            "no projections came back: every site failed or had nothing"
        )
    if not quiet:
        print(
            "projections: %d players, %d stat rows"
            % (len(points), len(stats))
        )
    return points, stats


def _reconcile(scraped, scoring, season, week, raw_stats):
    """One of the two tables, with the players' names and teams attached."""
    return calc_projections.add_player_info(
        calc_projections.projections_table(
            scraped,
            scoring,
            season,
            week,
            vor_baseline=BASELINE,
            raw_stats=raw_stats,
        )
    )


def cache_path(season, week, scoring=None):
    """Where this season's projections live, under these scoring rules.

    The default scoring adds nothing to the name, so the file this project
    publishes keeps the name it has always had; every other rule set caches
    beside it under its own key and cannot be handed a board scored by
    somebody else's rules.
    """
    key = (scoring or Scoring()).key
    tail = ("_" + key) if key else ""
    return cache_dir() / ("projections_%d_w%d%s.json" % (season, week, tail))


def _name_key(path):
    """The scoring key in a cache file's name, or None if it is not one."""
    m = CACHE_NAME.match(path.stem)
    return (m.group(3) or "") if m else None


def newest_cached(key=""):
    """The most recent cache scored under `key`, or None."""
    found = sorted(
        (
            p
            for p in cache_dir().glob("projections_*.json")
            if _name_key(p) == key
        ),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return found[0] if found else None


def norm(name):
    """A name reduced to what two spellings of one player agree on."""
    s = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z]", "", SUFFIX.sub("", s.lower()))


def _only(index, key):
    """The id at `key`, if exactly one player claims it."""
    hit = index.get(key)
    return hit[0] if hit and len(hit) == 1 else ""


def attach_ids(points, adp_path):
    """Fill in `pid`, the ADP board's id, on three passes.

    The board is read positionally rather than by header: it carries two
    columns called Team, the NFL one and the league one, so a dict keyed on
    the header quietly keeps the wrong one.

    Each pass requires a unique match:

        the full name and position, which settles the great majority
        the team, for kickers and defenses -- the board lists both as team
          units, "Dallas Cowboys" at position TK being whoever kicks for
          Dallas, so there is no name on that side to match against
        the surname with the team, which catches the players the two sources
          spell differently: Kenny against Kenneth Gainwell, Chig against
          Chigoziem Okonkwo

    A player still unmatched keeps an empty pid, has no market, and is priced
    on projections alone -- exactly what combined_draft already does with
    anyone ADP has no opinion about.
    """
    by_name, by_team, by_last = {}, {}, {}
    with open(adp_path, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.reader(f, delimiter="	"))[1:]
    for r in rows:
        pid, player, team, pos = r[PID], r[PLAYER], r[TEAM], r[POS]
        by_name.setdefault((norm(player), pos), []).append(pid)
        by_team.setdefault((team, pos), []).append(pid)
        parts = player.split()
        if parts:
            by_last.setdefault((norm(parts[-1]), team, pos), []).append(pid)

    matched = 0
    for p in points:
        pos = ADP_POS.get(p["position"], p["position"])
        team = ADP_TEAM.get(p["team"], p["team"])
        name = p["first_name"] + " " + p["last_name"]
        if pos in TEAM_UNITS:
            pid = _only(by_team, (team, pos))
        else:
            pid = _only(by_name, (norm(name), pos)) or _only(
                by_last, (norm(p["last_name"]), team, pos)
            )
        p["pid"] = pid
        matched += bool(pid)
    return matched


def published_url(season, week):
    return PUBLISHED + cache_path(season, week).name


def fetch_published(season, week, quiet=False):
    """The cache this project publishes, or None if it is not there.

    The numbers a scrape would have produced, already scraped: fetched over
    HTTP and then kept locally like any other cache. It costs one request
    rather than three minutes and nine sites, which is why it is tried
    before the scrape rather than after it.
    """
    url = published_url(season, week)
    try:
        with urllib.request.urlopen(url, timeout=TIMEOUT) as r:
            blob = json.loads(r.read().decode("utf-8"))
    except Exception as exc:
        if not quiet:
            print(
                "projections: nothing published for %d week %d (%s)"
                % (season, week, exc),
                file=sys.stderr,
            )
        return None
    if not quiet:
        print("projections: %d players from %s" % (len(blob["points"]), url))
    return blob


def _write_cache(blob, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(blob), encoding="utf-8")
    os.replace(tmp, path)


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def rescore(stats, scoring):
    """A points table scored here, off the component stats.

    The way to score a board by rules nobody has published is to scrape it
    under those rules, and that is what normally happens. This is the other
    case -- custom scoring with the sites unreachable -- and it works
    because the component stats are scoring's raw material: what a player is
    projected to do does not depend on what the league pays for it.

    What it costs is the reconciliation. ffanalytics returns each avg_type
    twice over, once as reconciled stats and once as reconciled points, and
    for `average` and `weighted` the two agree exactly -- scoring the
    reconciled stats gives back the published points to the last decimal,
    which is what makes this sound. For `robust` they do not: that one
    reconciles each source's points rather than each source's stats, and
    the difference reaches 18 points on a quarterback. So a rescored
    `robust` row is the robust stats scored, not the robust points --
    a defensible number, and not the same number. --refresh-projections
    gives the scrape instead, once the sites can be reached.

    `sd_pts` follows the components too, treating them as independent,
    which is what load_board would do to it anyway.
    """
    out = []
    for r in stats:
        pts = var = 0.0
        for c, w in scoring.values.items():
            if c in r:
                pts += _num(r[c]) * w
                var += (_num(r.get(c + "_sd")) * w) ** 2
        out.append(
            {
                "id": r["id"],
                "avg_type": r["avg_type"],
                "points": pts,
                "sd_pts": var**0.5,
                "first_name": r["first_name"],
                "last_name": r["last_name"],
                "team": r["team"],
                "position": r.get("position.x") or r.get("position", ""),
            }
        )
    return out


def _rescored(season, week, quiet, scoring):
    """The published stats, scored by these rules. None if there are none."""
    base = None
    default = cache_path(season, week)
    if default.exists():
        base = json.loads(default.read_text(encoding="utf-8"))
    else:
        base = fetch_published(season, week, quiet)
    if base is None:
        return None
    missing = [
        c
        for c in scoring.diff()
        if scoring.values.get(c) and c not in base["stats"][0]
    ]
    if missing:
        print(
            "projections: no source projects %s, so scoring it changes "
            "nothing here" % ", ".join(missing),
            file=sys.stderr,
        )
    if not quiet:
        print(
            "projections: no scrape to be had, so scoring the published "
            "component stats under your rules -- see projections.rescore()"
        )
    return {
        "season": season,
        "week": week,
        "points": rescore(base["stats"], scoring),
        "stats": base["stats"],
        "scoring": scoring.rules,
        "rescored": True,
    }


def _scraped(season, week, quiet, scoring, refresh=False):
    """A fresh scrape, or None with the reason on stderr."""
    try:
        points, stats = scrape(season, week, scoring, quiet, refresh)
    except Exception as exc:
        print("projections: %s" % exc, file=sys.stderr)
        return None
    return {
        "season": season,
        "week": week,
        "points": points,
        "stats": stats,
        "scoring": scoring.rules,
    }


def _stale(scoring):
    """Any cache scored by these rules -- better than nothing at all."""
    found = newest_cached(scoring.key)
    if found is None:
        return None
    print("projections: falling back on %s" % found, file=sys.stderr)
    return json.loads(found.read_text(encoding="utf-8"))


def _build(season, week, path, quiet, scoring, refresh=False):
    """Build the projections, by whatever means the network allows.

    A scrape, which is the only way to a board these rules have never been
    applied to. Failing that -- the sites unreachable, or one of them
    changed under us -- the published component stats scored here, which the
    default scoring has no use for, since what is published is already
    scored that way and scored better. Failing that, any cache at all.

    The rescored board is deliberately not written to the cache: it is a
    compromise made because a scrape did not happen, and it should not
    outlive the run that needed it.
    """
    blob = _scraped(season, week, quiet, scoring, refresh)
    if blob is not None:
        _write_cache(blob, path)
        return blob
    if not scoring.is_default():
        blob = _rescored(season, week, quiet, scoring)
        if blob is not None:
            return blob
    blob = _stale(scoring)
    if blob is None:
        raise SystemExit(
            "no projections for %d week %d under %s: nothing cached, "
            "nothing published, and the sites could not be reached "
            "(see the README)." % (season, week, scoring.summary())
        )
    return blob


def _cached(path, season, week, scoring, quiet):
    """The local cache, or the published copy, or None."""
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    if not scoring.is_default():
        return None  # only the default rules are published
    blob = fetch_published(season, week, quiet)
    if blob is not None:
        _write_cache(blob, path)
    return blob


def load(
    season=None,
    week=0,
    adp_path=None,
    refresh=False,
    quiet=False,
    scoring=None,
):
    """The two tables, with pids attached, from the nearest source that has
    them.

    In order: the local cache, then the copy this project publishes, then a
    fresh scrape. The middle step is what makes the tool open at once on a
    machine that has never run it -- and it is cached locally on arrival, so
    it is fetched once however many worker processes want it.

    `scoring` is the league's, and everything here is keyed on it: nothing
    but the default rules is published, so a league with its own scoring
    scrapes its own board, or has the published components scored for it.

    `refresh` skips both caches and insists on a scrape, which is the only
    way to get numbers newer than what is published.
    """
    scoring = scoring or Scoring()
    season = season or season_now()
    path = cache_path(season, week, scoring)
    blob = None if refresh else _cached(path, season, week, scoring, quiet)
    if blob is None:
        blob = _build(season, week, path, quiet, scoring, refresh)
    if adp_path is not None:
        attach_ids(blob["points"], adp_path)
    return Projections(
        blob["points"], blob["stats"], blob["season"], blob["week"], scoring
    )


def add_projection_args(ap):
    g = ap.add_argument_group(
        "projections",
        "scraped and reconciled as ffanalytics does, cached under %s"
        % cache_dir(),
    )
    g.add_argument(
        "--projections-season",
        type=int,
        default=None,
        metavar="YEAR",
        help="default: the current season",
    )
    g.add_argument(
        "--projections-week",
        type=int,
        default=0,
        help="0 is the preseason projection (default)",
    )
    g.add_argument(
        "--refresh-projections",
        action="store_true",
        help="re-scrape even if this season is cached",
    )
    return ap


def from_args(a, adp_path=None, quiet=False, scoring=None):
    """The projections those flags describe.

    `scoring` is the league's -- pass league_from_args(a).scoring, which is
    where the --score flags land. Without one it reads them itself, for a
    caller that has no league to speak of.
    """
    return load(
        a.projections_season,
        a.projections_week,
        adp_path,
        a.refresh_projections,
        quiet,
        scoring or scoring_from_args(a),
    )


def main():
    ap = add_scoring_args(
        add_projection_args(
            add_adp_args(
                argparse.ArgumentParser(
                    description=__doc__,
                    formatter_class=argparse.RawDescriptionHelpFormatter,
                )
            )
        )
    )
    a = ap.parse_args()
    board = adp_path_from_args(a)
    p = from_args(a, board)
    n = sum(1 for r in p.points if r["pid"])
    print("scoring: %s" % p.scoring.summary())
    print(
        "season %d week %d: %d players, %d stat rows, %d joined to ADP"
        % (p.season, p.week, len(p.points), len(p.stats), n)
    )
    print(cache_path(p.season, p.week, p.scoring))


if __name__ == "__main__":
    main()
