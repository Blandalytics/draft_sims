"""Nine sources into one board: ffanalytics' calc_projections.R, in Python.

source_scrapes brings back what each site projects. This is the other half of
ffanalytics -- `projections_table()` -- which turns nine disagreeing sites
into the two tables draftsim drafts on:

    points   a player's projected points, the spread across the sources, his
             floor and ceiling, his VOR against a baseline, his rank and his
             tier
    stats    the component stats behind those points, each with the standard
             deviation across the sources, which is what vor_draft_sim samples
             from to give every simulated draft its own season

Both come out of the same walk, the second by asking for the components
before they are scored rather than after -- which is why `raw_stats=True`
returns so much and reads so little like a separate function.

The walk, in the order the R takes it:

    stack the sources into one table per position, drop anyone without an
      MFL id, and give every row its source's weight

    impute what a source did not say. A site that projects passing yards but
      no attempts is not silent about attempts -- the other sites' ratio of
      attempts to yards says roughly what this site meant. Where there is no
      such ratio to borrow, the column's mean across the sources stands in.
      A player nobody projects for a column keeps nothing

    score each source's projection separately, by the league's rules, so
      that what is reconciled is nine opinions about the same player rather
      than one average of nine different-looking stat lines

    reconcile, three ways at once, because they disagree usefully:

        average   the plain mean, and the sample standard deviation
        robust    Hodges-Lehmann location (the median of every pairwise
                  average), with MAD for spread -- one wild source moves it
                  much less than it moves the mean
        weighted  ffanalytics' own source weights, with a weighted standard
                  deviation and Harrell-Davis weighted quantiles

    rank, tier and price. Tiers break where the gap to the next player
      exceeds the position's typical spread; VOR is measured against the
      baseline rank ffanalytics uses, which is not the replacement level
      draftsim drafts against -- league.py derives that from the league's own
      shape, and this baseline only decides the `rank` column's ordering

Everything here is the R's arithmetic, including the parts that surprise:
`sd` is NA for a player only one source projects, the weighted average is
NaN for a player only the zero-weighted sources have (which is why the
weighted table is shorter), and `dropoff` is the gap down to the next player
rather than up to the last.

Two things ffanalytics does that this cannot, both because they need tables
kept in the package's binary data rather than published anywhere readable:

    the week-0 points-allowed bracket, which R scores by simulating a season
      of games per defense from per-team coefficients. Here the projected
      points allowed are scored per game and multiplied out, which is the
      same idea without the noise. Under every scoring rule this project
      ships the result is multiplied by zero anyway -- see `dst_bracket()`

    imputing bonus columns (a 300-yard passing game, a 100-yard rushing one)
      from a regression on the yardage. Those columns are aggregated here
      only from the sources that actually report them, which matters solely
      to a league that pays yardage bonuses
"""

import math
from collections import defaultdict

import numpy as np
from scipy.stats import beta

from .mfl_ids import player_table
from .scoring import BRACKET, Scoring

AVG_TYPES = ("average", "robust", "weighted")

# ffanalytics' default_weights: what each source is worth in the weighted
# reconciliation. A zero is not an oversight -- FantasyPros publishes other
# people's projections, so counting it would count them twice.
DEFAULT_WEIGHTS = {
    "CBS": 0.145,
    "Yahoo": 0.000,
    "ESPN": 0.157,
    "NFL": 0.140,
    "FFToday": 0.151,
    "NumberFire": 0.142,
    "FantasyPros": 0.000,
    "FantasySharks": 0.142,
    "FantasyFootballNerd": 0.000,
    "WalterFootball": 0.130,
    "RTSports": 0.123,
    "FantasyData": 0.000,
    "FleaFlicker": 0.000,
    "FanDuel": 0.142,
}

# The rank VOR is measured from, per position. ffanalytics' default_baseline.
DEFAULT_BASELINE = {
    "QB": 13,
    "RB": 35,
    "WR": 36,
    "TE": 13,
    "K": 8,
    "DST": 3,
    "DL": 10,
    "LB": 10,
    "DB": 10,
}

# How big a gap has to be, in units of the position's median spread, before
# it starts a new tier. ffanalytics' default_threshold.
DEFAULT_THRESHOLD = {
    "QB": 1,
    "RB": 1,
    "WR": 1,
    "TE": 1,
    "K": 1,
    "DST": 0.1,
    "DL": 1,
    "DB": 1,
    "LB": 1,
}

# What a source that gives one column but not another implies about the one
# it left out: the ratio between them, averaged over the sources that give
# both. ffanalytics' impute_fun_list, which is keyed by the column to fill
# and names the column to fill it from.
#
# `rec` is missing from this list on purpose, because it is missing from the
# R's: the entry meant for it is named `rec_tgt`, which is already taken, so
# `rec_tgt` fills from `rec` and `rec` itself falls through to its own mean.
IMPUTE_FROM = {
    "pass_att": "pass_yds",
    "pass_comp": "pass_yds",
    "pass_tds": "pass_comp",
    "pass_int": "pass_att",
    "rush_att": "rush_yds",
    "rush_tds": "rush_yds",
    "rec_tgt": "rec",
    "rec_tds": "rec_yds",
    "xp_att": "xp",
    "fg_att": "fg",
    "fg_0019": "fg",
    "fg_2029": "fg",
    "fg_3039": "fg",
    "fg_4049": "fg",
    "fg_50": "fg",
    "fg_miss": "fg",
}

# The columns a kicker's field goals can be totalled from, and the ones a
# miss can be counted from, when a source gives the parts but not the total.
FG_PARTS = ("fg_0019", "fg_2029", "fg_3039", "fg_4049", "fg_50", "fg_0039")
FG_MISSES = (
    "fg_miss_0019",
    "fg_miss_2029",
    "fg_miss_3039",
    "fg_miss_4049",
    "fg_miss_50",
)

# The player table's columns that travel with a projection.
PLAYER_INFO = ("first_name", "last_name", "team", "position", "age", "exp")


# ------------------------------------------------------------- estimators


def _present(values, weights):
    """The observations that exist, with the weights that go with them."""
    if weights is None:
        weights = [1.0] * len(values)
    pairs = [
        (v, w) for v, w in zip(values, weights, strict=True) if v is not None
    ]
    return [v for v, _ in pairs], [w for _, w in pairs]


def mean_(values, _weights=None):
    """mean.default(x, na.rm = TRUE)."""
    present, _ = _present(values, None)
    return sum(present) / len(present) if present else None


def sd_(values, _weights=None):
    """sd(x, na.rm = TRUE): NA for a player only one source projects."""
    present, _ = _present(values, None)
    if len(present) < 2:
        return None
    return float(np.std(present, ddof=1))


def quantile_(values, probs, _weights=None):
    """quantile(x, probs, na.rm = TRUE), R's default type 7."""
    present, _ = _present(values, None)
    if not present:
        return [None] * len(probs)
    return [float(q) for q in np.quantile(present, probs, method="linear")]


def wilcox_loc(values, _weights=None):
    """The Hodges-Lehmann location: the median of every pairwise average.

    Every value is also paired with itself, which is what the R's
    `c(vec, combn(vec, 2, mean))` amounts to, and with two observations or
    fewer it is just the mean.
    """
    if len(values) <= 2:
        return mean_(values)
    present, _ = _present(values, None)
    pool = list(present)
    for i, left in enumerate(present):
        pool.extend([(left + right) / 2 for right in present[i + 1 :]])
    return float(np.median(pool)) if pool else None


def mad_(values, _weights=None):
    """mad(x) * 1.4826, and NA wherever the R's would be NA.

    The R's `mad2` takes its centre from the vector before the NAs are
    dropped, so one missing observation makes the whole spread NA. That is
    left as it is: a source that did not project a column is not evidence
    about how much the others disagree.
    """
    if len(values) <= 1 or any(v is None for v in values):
        return None
    centre = np.median(values)
    return float(np.median(np.abs(np.asarray(values) - centre)) * 1.4826)


def weighted_mean(values, weights):
    """weighted.mean(x, w, na.rm = TRUE): NaN when every weight is zero.

    A source ffanalytics has no weight for makes the whole average NA, as it
    does in R -- an unweighted opinion is not the same as an unimportant one,
    and the answer is to give the source a weight rather than to guess.
    """
    present, kept = _present(values, weights)
    if not present:
        return None
    if any(w is None for w in kept):
        return math.nan
    total = sum(kept)
    if total == 0:
        return math.nan  # dropped downstream, as R drops a non-finite mean
    return sum(v * w for v, w in zip(present, kept, strict=True)) / total


def weighted_sd(values, weights):
    """ffanalytics' weighted.sd: the reliability-weighted sample spread."""
    present, kept = _numeric(values, weights)
    if len(present) <= 1:
        return None
    total, squares = sum(kept), sum(w * w for w in kept)
    if total**2 == squares:
        return None
    centre = sum(v * w for v, w in zip(present, kept, strict=True)) / total
    spread = sum(
        w * (v - centre) ** 2 for v, w in zip(present, kept, strict=True)
    )
    return math.sqrt((total / (total**2 - squares)) * spread)


def whd_quantile(values, probs, weights):
    """The weighted Harrell-Davis quantile estimator.

    From Akinshin (2023), "Weighted quantile estimators", and it is what
    ffanalytics uses for the weighted floor and ceiling: every observation
    contributes to every quantile, in proportion to how much of the beta
    distribution around that quantile it covers.
    """
    present, kept = _numeric(values, weights)
    if len(present) <= 1:
        return [None] * len(probs)
    total, squares = sum(kept), sum(w * w for w in kept)
    effective = total**2 / squares  # Kish's effective sample size
    order = np.argsort(np.asarray(present), kind="stable")
    ordered = np.asarray(present)[order]
    share = np.asarray(kept)[order] / total
    edges = np.concatenate(([0.0], np.cumsum(share)))
    out = []
    for prob in probs:
        cdf = beta.cdf(
            edges, (effective + 1) * prob, (effective + 1) * (1 - prob)
        )
        out.append(float(np.sum(np.diff(cdf) * ordered)))
    return out


def _numeric(values, weights):
    """The observations with a weight worth having, as the R's do."""
    pairs = [
        (v, w)
        for v, w in zip(values, weights, strict=True)
        if v is not None and w is not None and w > 0
    ]
    return [v for v, _ in pairs], [w for _, w in pairs]


ESTIMATORS = {
    "average": (mean_, sd_, quantile_),
    "robust": (wilcox_loc, mad_, quantile_),
    "weighted": (weighted_mean, weighted_sd, whd_quantile),
}


# ------------------------------------------------------------ the sources


def by_position(scraped, weights=None):
    """The sources stacked into one table per position, weights attached.

    Rows with no MFL id are dropped -- there is nothing to reconcile them
    with -- and every table is made rectangular, so that a column one source
    does not have reads as missing for it rather than as absent.
    """
    weights = DEFAULT_WEIGHTS if weights is None else weights
    stacked = defaultdict(list)
    for tables in scraped.values():
        for position, rows in tables.items():
            for row in rows:
                if row.get("id"):
                    stacked[position].append(
                        {**row, "weights": weights.get(row.get("data_src"))}
                    )
    return {pos: _rectangular(rows) for pos, rows in stacked.items()}


def _rectangular(rows):
    """Every row carrying every column, missing ones as None."""
    names = list(dict.fromkeys([n for row in rows for n in row]))
    return [{n: row.get(n) for n in names} for row in rows]


def _by_id(rows):
    """The rows for each player, in the order the players first appear."""
    groups = defaultdict(list)
    for row in rows:
        groups[row["id"]].append(row)
    return groups


# ------------------------------------------------------------- imputation


def impute(rows, position, scoring):
    """Fill in what a source left out, from what the others said.

    Only the columns the league scores are worth filling in, and only where
    something is actually missing.
    """
    if not rows:
        return rows
    if position == "K":
        _kicker_totals(rows)
    names = list(rows[0])
    wanted = [n for n in names if scoring.values.get(n)]
    if position == "DST" and "dst_pts_allowed" in names:
        wanted = list(dict.fromkeys(wanted + ["dst_pts_allowed"]))
    groups = _by_id(rows)
    for column in wanted:
        if not any(row[column] is None for row in rows):
            continue
        for group in groups.values():
            _impute_column(group, column)
    return rows


def _impute_column(group, column):
    values = [row[column] for row in group]
    reference = IMPUTE_FROM.get(column)
    if reference and reference in group[0]:
        filled = derive_from_rate(values, [row[reference] for row in group])
    else:
        filled = derive_from_mean(values)
    for row, value in zip(group, filled, strict=True):
        row[column] = value


def derive_from_rate(need, reference):
    """What this source meant, at the rate the other sources imply.

    The ratio between the two columns is averaged over the sources that give
    both, then applied to the reference this source did give. A reference of
    zero makes the ratio meaningless, so the column's own mean is used
    instead, and a column nobody gives stays missing.
    """
    known = [i for i, v in enumerate(need) if v is not None]
    if not known:
        return list(need)
    if any(reference[i] == 0 for i in known if reference[i] is not None):
        return derive_from_mean(need)
    ratios = [
        need[i] / reference[i] for i in known if reference[i] is not None
    ]
    rate = sum(ratios) / len(known)
    return [
        v
        if v is not None
        else (None if reference[i] is None else rate * reference[i])
        for i, v in enumerate(need)
    ]


def derive_from_mean(need):
    """The column's mean over the sources that gave it."""
    present = [v for v in need if v is not None]
    if not present:
        return list(need)
    mean = sum(present) / len(present)
    return [mean if v is None else v for v in need]


def _kicker_totals(rows):
    """A kicker's totals, where a source gives only the parts.

    Field goals made, attempted and missed are each derivable from the
    others, and the sources disagree about which of them to publish.
    """
    names = list(rows[0])
    for row in rows:
        if "fg_miss" not in names and all(c in names for c in FG_MISSES):
            row["fg_miss"] = _total(row, FG_MISSES)
        if row.get("fg") is None:
            row["fg"] = _total(row, [c for c in FG_PARTS if c in names])
        if "xp_att" not in names:
            row["xp_att"] = _sum_or_none(row.get("xp"), row.get("xp_miss"))
        if row.get("fg_att") is None:
            row["fg_att"] = _fg_att(row)
    for name in ("fg_miss", "xp_att", "fg_att"):
        for row in rows:
            row.setdefault(name, None)


def _fg_att(row):
    """Attempts, from the misses or from the percentage, if either is there."""
    from_miss = _sum_or_none(row.get("fg"), row.get("fg_miss"))
    if from_miss is not None:
        return from_miss
    made, pct = row.get("fg"), row.get("fg_pct")
    if made is not None and isinstance(pct, (int, float)) and pct:
        return made / (pct * 0.01)
    return None


def _total(row, columns):
    values = [row.get(c) for c in columns]
    present = [v for v in values if v is not None]
    return sum(present) if present else None


def _sum_or_none(left, right):
    if left is None or right is None:
        return None
    return left + right


# ---------------------------------------------------------------- scoring


def source_points(tables, scoring, season, week):
    """What each source's projection is worth, one row at a time.

    Scoring every source separately is the whole point: what gets reconciled
    is nine opinions about a player, not one average of nine stat lines.
    """
    for position, rows in tables.items():
        if position == "DST":
            dst_bracket(rows, scoring, season, week)
        for row in rows:
            row["raw_points"] = sum(
                row[column] * value
                for column, value in scoring.values.items()
                if row.get(column) is not None
            )
    return tables


def dst_bracket(rows, scoring, season, week):
    """Turn a defense's projected points allowed into bracket points.

    A league pays a defense on a sliding scale -- ten points for a shutout,
    less as the scoreboard moves -- so what a source projects as points
    allowed has to be scored against that scale before it can be added up.

    Two things about this are worth knowing. Over a season the scale is per
    game, so the projection is scored per game and multiplied out; R instead
    simulates seventeen games around the projection, which needs a per-team
    variance table that lives in its binary package data. And whatever comes
    out is then multiplied by the rules' own `dst_pts_allowed`, which is
    zero in Yahoo's scoring and in ffanalytics' -- so on any rule set this
    project ships, the bracket contributes nothing at all. It is here so
    that a league that does pay for it is scored correctly.
    """
    bracket = scoring.rules.get(BRACKET) or []
    if not bracket:
        return
    games = 17 if season is None or season >= 2021 else 16
    for row in rows:
        allowed = row.get("dst_pts_allowed")
        if allowed is None:
            continue
        if week == 0:
            row["dst_pts_allowed"] = _bracket(allowed / games, bracket) * games
        else:
            row["dst_pts_allowed"] = _bracket(allowed, bracket)


def _bracket(allowed, bracket):
    """The first band the points allowed fall inside."""
    for threshold, points in bracket:
        if allowed <= threshold:
            return points
    return bracket[0][1]  # R's which.max on no match, kept as it is


# ------------------------------------------------------------ aggregation


def _points_rows(rows, position, avg_type):
    """One row per player: his points, spread, floor and ceiling."""
    average, spread, quantile = ESTIMATORS[avg_type]
    out = []
    groups = _by_id(rows)
    for player_id in sorted(groups):
        group = groups[player_id]
        values = [row["raw_points"] for row in group]
        weights = [row["weights"] for row in group]
        low, high = quantile(values, [0.05, 0.95], weights)
        out.append(
            {
                "id": player_id,
                "pos": position,
                "points": average(values, weights),
                "sd_pts": spread(values, weights),
                "floor": low,
                "ceiling": high,
            }
        )
    return [r for r in out if _finite(r["points"]) and r["points"] > 0]


def _finite(value):
    return isinstance(value, (int, float)) and math.isfinite(value)


def _rank_position(rows, threshold):
    """Rank a position, mark its dropoffs, and cut it into tiers.

    `dropoff` is the gap down to the next player, so the last man's is zero,
    and a tier ends where the gaps since the top of the table add up to more
    than the position's typical disagreement.
    """
    rows.sort(key=lambda r: r["points"])
    for i, row in enumerate(rows):
        row["dropoff"] = 0 if i == 0 else row["points"] - rows[i - 1]["points"]
    ranks = dense_rank([-r["points"] for r in rows])
    for row, rank in zip(rows, ranks, strict=True):
        row["pos_rank"] = rank
    rows.reverse()
    spreads = [r["sd_pts"] for r in rows if r["sd_pts"] is not None]
    width = float(np.median(spreads)) * threshold if spreads else 0.0
    running, first = 0.0, rows[0]["dropoff"] if rows else 0.0
    for row in rows:
        running += row["dropoff"]
        row["tier"] = (
            1 + math.trunc((running - first) / width) if width else 1
        )
    for row, tier in zip(rows, dense_rank([r["tier"] for r in rows]),
                         strict=True):
        row["tier"] = tier
    return rows


def dense_rank(values):
    """dplyr::dense_rank: ties share a rank and no rank is skipped."""
    order = {v: i + 1 for i, v in enumerate(sorted({v for v in values
                                                    if v is not None}))}
    return [None if v is None else order[v] for v in values]


def add_vor(rows, baseline):
    """Points above the baseline player, and the ranks that follow.

    Position by position, the baseline is the player at a fixed rank -- the
    35th running back, the 13th quarterback -- and everyone is priced by how
    far above him they are. The overall ranks are then taken on that
    difference rather than on points, which is what lets a quarterback and a
    running back be compared at all.
    """
    for position_rows in _split(rows, "pos").values():
        _vor_column(position_rows, baseline, "points", "pos_rank")
        floors = dense_rank([_negate(r["floor"]) for r in position_rows])
        ceilings = dense_rank([_negate(r["ceiling"]) for r in position_rows])
        _vor_column(position_rows, baseline, "floor", floors)
        _vor_column(position_rows, baseline, "ceiling", ceilings)
    for column, into in (
        ("points_vor", "rank"),
        ("floor_vor", "floor_rank"),
        ("ceiling_vor", "ceiling_rank"),
    ):
        ranks = dense_rank([_negate(r[column]) for r in rows])
        for row, rank in zip(rows, ranks, strict=True):
            row[into] = rank
    return rows


def _negate(value):
    return None if value is None else -value


def _vor_column(rows, baseline, column, ranks):
    """One column priced against the player at the baseline rank."""
    if isinstance(ranks, str):
        ranks = [r[ranks] for r in rows]
    wanted = baseline.get(rows[0]["pos"])
    reference = rows[0][column]
    for row, rank in zip(rows, ranks, strict=True):
        if rank == wanted:
            reference = row[column]
            break
    for row in rows:
        known = row[column] is not None and reference is not None
        row[column + "_vor"] = row[column] - reference if known else None


def _split(rows, key):
    out = defaultdict(list)
    for row in rows:
        out[row[key]].append(row)
    return out


def _stats_rows(rows, columns, avg_type):
    """One row per player: each component stat, and its spread."""
    average, spread, _ = ESTIMATORS[avg_type]
    out = {}
    for player_id, group in _by_id(rows).items():
        if len(group) < 2:
            continue  # one source is not a reconciliation
        weights = [row["weights"] for row in group]
        row = {"avg_type": avg_type}
        for column in columns:
            values = [member[column] for member in group]
            row[column] = average(values, weights)
            row[column + "_sd"] = spread(values, weights)
        out[player_id] = row
    return out


# ------------------------------------------------------------- the tables


def projections_table(
    scraped,
    scoring=None,
    season=None,
    week=0,
    src_weights=None,
    vor_baseline=None,
    tier_thresholds=None,
    avg_types=AVG_TYPES,
    raw_stats=False,
):
    """The projections table, as ffanalytics builds it.

    `scraped` is what source_scrapes.scrape returns: a table per position
    per source. What comes back is one long table with a row per player per
    avg_type -- points and ranks by default, the component stats behind them
    with `raw_stats=True`.
    """
    scoring = scoring or Scoring()
    tables = by_position(scraped, src_weights)
    for position, rows in tables.items():
        impute(rows, position, scoring)
    if raw_stats:
        return _real(_stats_table(tables, scoring, avg_types))
    tables = source_points(tables, scoring, season, week)
    return _real(
        _points_table(
            tables,
            avg_types,
            vor_baseline or DEFAULT_BASELINE,
            tier_thresholds or DEFAULT_THRESHOLD,
        )
    )


def _real(rows):
    """A number that is not a number is missing, and says so as `None`.

    R writes its NAs out as blanks; a NaN left in a table would be written
    as JSON that only Python reads back.
    """
    return [
        {
            name: None
            if isinstance(v, float) and not math.isfinite(v)
            else v
            for name, v in row.items()
        }
        for row in rows
    ]


def _points_table(tables, avg_types, baseline, thresholds):
    out = []
    for avg_type in avg_types:
        scored = []
        for position, rows in tables.items():
            found = _points_rows(rows, position, avg_type)
            if found:
                scored.extend(
                    _rank_position(found, thresholds.get(position, 1))
                )
        for row in add_vor(scored, baseline):
            out.append({"avg_type": avg_type, **row})
    return [{name: row.get(name) for name in POINTS_COLUMNS} for row in out]


POINTS_COLUMNS = (
    "avg_type",
    "id",
    "pos",
    "points",
    "sd_pts",
    "dropoff",
    "floor",
    "ceiling",
    "points_vor",
    "floor_vor",
    "ceiling_vor",
    "rank",
    "floor_rank",
    "ceiling_rank",
    "pos_rank",
    "tier",
)


def _stats_table(tables, scoring, avg_types):
    """The component stats, one row per player per avg_type.

    Only the columns the league scores are aggregated -- a category worth
    nothing is not worth reconciling -- and only the players more than one
    source projects, since a single source has no spread to report.
    """
    out = []
    for position, rows in tables.items():
        if not rows:
            continue
        columns = [n for n in rows[0] if scoring.values.get(n)]
        found = {t: _stats_rows(rows, columns, t) for t in avg_types}
        reconciled = [
            r["id"]
            for r in rows
            if any(r["id"] in table for table in found.values())
        ]
        for player_id in dict.fromkeys(reconciled):
            for avg_type in avg_types:
                row = found[avg_type].get(player_id)
                if row is not None:
                    out.append(
                        {"position": position, "id": player_id, **row}
                    )
    return _rectangular(out)


def add_player_info(rows):
    """Who these ids are: name, team, position, age and years played.

    Where the projection already carries a `position` -- the stats table
    does, naming the table it came from -- the player table's own goes in
    beside it as `position.y`, which is what R's join does and what
    vor_draft_sim reads.
    """
    players = {row["id"]: row for row in player_table()}
    out = []
    for row in rows:
        found = players.get(row["id"], {})
        clash = "position" in row
        info = {}
        for name in PLAYER_INFO:
            value = found.get(name, "")
            if name in ("age", "exp"):
                value = _number(value)
            info["position.y" if clash and name == "position" else name] = (
                value
            )
        if clash:
            row = {
                ("position.x" if name == "position" else name): value
                for name, value in row.items()
            }
        out.append({**row, **info})
    return out


def _number(text):
    try:
        return int(text)
    except (TypeError, ValueError):
        return None
