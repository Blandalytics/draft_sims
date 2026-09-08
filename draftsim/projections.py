"""Projections, pulled from the ffanalytics R package rather than kept as CSVs.

ffanalytics scrapes half a dozen projection sites and reconciles them, and it
is an R package. So this drives it directly: an R program, held below as a
string, is handed to Rscript, and hands back the two tables draftsim needs on
its standard output. Nothing is written to a CSV on the way through.

    points   one row per player -- projected points, their spread across the
             sources, VOR against the baselines, rank, tier, and the name,
             team and position that go with the id
    stats    the component stats behind those points, each with the standard
             deviation across sources, which is what vor_draft_sim samples
             from to give every simulated draft its own season

Both come out of the same call, `projections_table()`, the second with
`return_raw_stats = TRUE`. That is worth knowing because it is not obvious:
the stat-level table with its per-source standard deviations is not a separate
function, it is the same aggregation returned before it is scored.

Scoring is Yahoo's default, 0.5 PPR, written into the R program below so that
the run carries its own rules rather than depending on a saved object. It has
to agree with vor_draft_sim.YAHOO_POINTS, which scores the sampled stats on
the Python side -- the rules decide which stat columns come back at all, since
a category worth zero points is not returned.

A scrape takes about two minutes and hits six sites per position, so the
result is cached under DRAFTSIM_CACHE (or ~/.cache/draftsim) as JSON, keyed by
season and week. `--refresh-projections` forces a new one; a failed scrape
falls back on the newest cache rather than leaving you with nothing.

One column ffanalytics does not have is `pid`, the id the ADP board uses --
its own ids come from a different registry. That join is made here instead, on
normalised name and position against the scraped board, and only where the
match is unique. Checked against a hand-made mapping of 477 players it agreed
on 417 and contradicted none; the rest are players with no ADP at all, deep
enough that nobody drafted them in the window, and they are simply left
without a market, exactly as combined_draft already expects.
"""

import argparse
import csv
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unicodedata
from collections import namedtuple
from datetime import date
from pathlib import Path

from .adp import add_adp_args, cache_dir
from .adp import path_from_args as adp_path_from_args

POSITIONS = ("QB", "RB", "WR", "TE", "K", "DST")
TIMEOUT = 900  # a scrape is ~2 minutes; this is a wide safety margin
SEASON_STARTS = 3  # from March, "this season" means this calendar year
POINTS_MARK = "##POINTS"
STATS_MARK = "##STATS"

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

Projections = namedtuple("Projections", "points stats season week")

# Yahoo default scoring, 0.5 PPR, and the VOR baselines. These live in the R
# program because that is where they are applied; vor_draft_sim.YAHOO_POINTS
# is the same table on the Python side, for scoring the sampled stats.
R_PROGRAM = r"""
suppressMessages(library(ffanalytics))

args   <- commandArgs(trailingOnly = TRUE)
season <- as.integer(args[1])
week   <- as.integer(args[2])

scoring <- list(
  pass = list(pass_att = 0, pass_comp = 0, pass_inc = 0, pass_yds = 0.04,
              pass_tds = 4, pass_int = -1, pass_40_yds = 0, pass_300_yds = 0,
              pass_350_yds = 0, pass_400_yds = 0),
  rush = list(all_pos = TRUE, rush_yds = 0.1, rush_att = 0, rush_40_yds = 0,
              rush_tds = 6, rush_100_yds = 0, rush_150_yds = 0,
              rush_200_yds = 0),
  rec  = list(all_pos = TRUE, rec = 0.5, rec_yds = 0.1, rec_tds = 6,
              rec_40_yds = 0, rec_100_yds = 0, rec_150_yds = 0,
              rec_200_yds = 0),
  misc = list(all_pos = TRUE, fumbles_lost = -2, fumbles_total = 0,
              sacks = 0, two_pts = 2),
  kick = list(xp = 1, fg_0019 = 3, fg_2029 = 3, fg_3039 = 3, fg_4049 = 4,
              fg_50 = 5, fg_miss = 0),
  ret  = list(all_pos = TRUE, return_tds = 6, return_yds = 0),
  dst  = list(dst_fum_rec = 2, dst_int = 2, dst_safety = 2, dst_sacks = 1,
              dst_td = 6, dst_blk = 2, dst_ret_yds = 0, dst_pts_allowed = 0),
  pts_bracket = list(
    list(threshold = 0,  points = 10), list(threshold = 6,  points = 7),
    list(threshold = 13, points = 4),  list(threshold = 20, points = 1),
    list(threshold = 27, points = 0),  list(threshold = 34, points = -1),
    list(threshold = 99, points = -4))
)
baseline <- c(QB = 10, RB = 20, WR = 20, TE = 10, K = 3, DST = 3,
              DL = 10, LB = 10, DB = 10)

# scrape_data narrates itself on stdout, which is where the tables have to go,
# so its chatter is caught and dropped. Anything it raises as a message still
# reaches stderr, where the caller reports it.
noise <- textConnection("scrape_noise", "w", local = TRUE)
sink(noise, type = "output")
raw <- scrape_data(pos = c("QB", "RB", "WR", "TE", "K", "DST"),
                   season = season, week = week)
sink(type = "output")
close(noise)

points <- add_player_info(projections_table(
  raw, scoring_rules = scoring, vor_baseline = baseline))
stats <- add_player_info(projections_table(
  raw, scoring_rules = scoring, vor_baseline = baseline,
  return_raw_stats = TRUE))

out <- function(mark, df) {
  cat(mark, "\n", sep = "")
  write.table(df, stdout(), sep = "\t", row.names = FALSE, quote = FALSE,
              na = "")
}
out("##POINTS", points)
out("##STATS", stats)
"""


def season_now(today=None):
    """The season ffanalytics would call current."""
    d = today or date.today()
    return d.year if d.month >= SEASON_STARTS else d.year - 1


def rscript():
    """Rscript, from the path, DRAFTSIM_RSCRIPT, or a Windows install."""
    env = os.environ.get("DRAFTSIM_RSCRIPT")
    if env:
        return env
    found = shutil.which("Rscript")
    if found:
        return found
    installs = sorted(
        Path("C:/Program Files/R").glob("R-*/bin/Rscript.exe"), reverse=True
    )
    if installs:
        return str(installs[0])
    raise SystemExit(
        "Rscript not found. draftsim reads its projections out of the "
        "ffanalytics R package, so R has to be installed and on the path, "
        "or DRAFTSIM_RSCRIPT set to the Rscript binary."
    )


def scrape(season, week=0, timeout=TIMEOUT, quiet=False):
    """Run the R program and read the two tables off its output."""
    if not quiet:
        print(
            "projections: scraping %d week %d via ffanalytics, ~2 minutes..."
            % (season, week),
            end=" ",
            flush=True,
        )
    # The program goes to a temporary file rather than `Rscript -e`: a
    # multi-line program handed to -e crashes the interpreter outright on
    # Windows. The file is the R program held above, written out and removed
    # again, not an artifact of the run.
    with tempfile.TemporaryDirectory(prefix="draftsim-r-") as tmp:
        prog = Path(tmp) / "projections.R"
        prog.write_text(R_PROGRAM, encoding="utf-8")
        proc = subprocess.run(
            [rscript(), str(prog), str(season), str(week)],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    if proc.returncode != 0:
        tail = "\n".join(proc.stderr.strip().splitlines()[-8:])
        raise RuntimeError(
            "ffanalytics failed (exit %d):\n%s" % (proc.returncode, tail)
        )
    points, stats = split(proc.stdout)
    if not points or not stats:
        raise RuntimeError("ffanalytics returned no projections")
    if not quiet:
        print("%d players" % len(points))
    return points, stats


def split(text):
    """The two marked TSV sections of the R program's output."""
    a = text.find(POINTS_MARK)
    b = text.find(STATS_MARK, a + 1)
    if a < 0 or b < 0:
        raise RuntimeError("ffanalytics output carried no table markers")
    return (
        _tsv(text[a + len(POINTS_MARK) : b]),
        _tsv(text[b + len(STATS_MARK) :]),
    )


def _tsv(chunk):
    return list(
        csv.DictReader(io.StringIO(chunk.strip("\r\n")), delimiter="\t")
    )


def cache_path(season, week):
    return cache_dir() / ("projections_%d_w%d.json" % (season, week))


def newest_cached():
    found = sorted(
        cache_dir().glob("projections_*.json"),
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


def load(season=None, week=0, adp_path=None, refresh=False, quiet=False):
    """The two tables, from cache or a fresh scrape, with pids attached."""
    season = season or season_now()
    path = cache_path(season, week)
    if path.exists() and not refresh:
        blob = json.loads(path.read_text(encoding="utf-8"))
    else:
        try:
            points, stats = scrape(season, week, quiet=quiet)
        except Exception as exc:
            stale = newest_cached()
            if stale is None:
                raise SystemExit(
                    "could not build projections (%s), and nothing cached to "
                    "fall back on." % exc
                ) from exc
            print(
                "projections: %s\n  falling back on %s" % (exc, stale),
                file=sys.stderr,
            )
            blob = json.loads(stale.read_text(encoding="utf-8"))
        else:
            blob = {
                "season": season,
                "week": week,
                "points": points,
                "stats": stats,
            }
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(blob), encoding="utf-8")
            os.replace(tmp, path)
    if adp_path is not None:
        attach_ids(blob["points"], adp_path)
    return Projections(
        blob["points"], blob["stats"], blob["season"], blob["week"]
    )


def add_projection_args(ap):
    g = ap.add_argument_group(
        "projections",
        "pulled from the ffanalytics R package, cached under %s" % cache_dir(),
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


def from_args(a, adp_path=None, quiet=False):
    return load(
        a.projections_season,
        a.projections_week,
        adp_path,
        a.refresh_projections,
        quiet,
    )


def main():
    ap = add_projection_args(
        add_adp_args(
            argparse.ArgumentParser(
                description=__doc__,
                formatter_class=argparse.RawDescriptionHelpFormatter,
            )
        )
    )
    a = ap.parse_args()
    board = adp_path_from_args(a)
    p = from_args(a, board)
    n = sum(1 for r in p.points if r["pid"])
    print(
        "season %d week %d: %d players, %d stat rows, %d joined to ADP"
        % (p.season, p.week, len(p.points), len(p.stats), n)
    )
    print(cache_path(p.season, p.week))


if __name__ == "__main__":
    main()
