"""Simulate snake drafts off the ADP mean/SD features.

Each draft draws one simulated slot per player from a skew normal fitted to
that player, clipped up to his observed Min Pick so nobody ever simulates
earlier than he has actually been taken. That gives each draft its own noisy
board. Teams then take the best available player -- lowest simulated slot --
that their roster can still legally hold.

The skew normal is fitted once at load, three features to three parameters:
    shape alpha    <- where ADP sits in [Min Pick, Max Pick]
    scale omega    <- the SD estimate (Max-Min)/(0.61*ln(n)+2.235)
    location xi    <- ADP
so each player's simulated slot has mean exactly ADP and sd exactly that SD
estimate, with the skew implied by his observed range. See skewnorm_fit.py.

The league -- team count, roster size, starting lineup, flex -- comes from
league.py and defaults to 12 teams, 15 spots each (180 picks), 1 QB, 2 RB,
3 WR, 1 TE, 1 FLEX (RB/WR/TE), 1 DST, 1 K, 5 bench. It must match
vor_draft_sim.py's, since combined_draft.py blends the two boards and drafts
the result; the shared league flags (--teams, --roster, --qb ...) keep them in
step. See league.py for the roster rules and draft-order policy that follow.
"""

import csv
import math

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.stats import skewnorm

from . import adp
from .league import (
    DEF,
    DEFAULT,
    DRAFTABLE,
    K,
    add_league_args,
    league_from_args,
)

N_DRAFTS = 10000
MIN_PICKS = 20
SRC_DEF = "TDSP"  # DEF as spelled in ADP.tsv
SRC_K = "TK"  # K as spelled in ADP.tsv

RANK, PID, PLAYER, TEAM, POS, ADP_C, MINP, MAXP, _DIFF, NPICKS = range(10)

A_LO, A_HI = -60.0, 60.0  # bisection bracket for the skew parameter


def mean_position(a, u_lo, u_hi):
    """Where a standard skew normal's mean falls between two quantiles."""
    lo, hi = skewnorm.ppf(u_lo, a), skewnorm.ppf(u_hi, a)
    return (skewnorm.mean(a) - lo) / (hi - lo)


def solve_shape(p_obs, u_lo, u_hi, tol=1e-6):
    """Bisect for alpha. mean_position DECREASES in alpha, hence the flip.

    Players whose ADP sits more lopsidedly inside their range than any skew
    normal allows (|skewness| < 0.995) clamp at the bracket ends; their mean
    and sd are still exact, only the tail shape is capped.
    """
    lo, hi = A_LO, A_HI
    if p_obs >= mean_position(lo, u_lo, u_hi):
        return lo, True
    if p_obs <= mean_position(hi, u_lo, u_hi):
        return hi, True
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if mean_position(mid, u_lo, u_hi) > p_obs:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi), False


def fit_skewnorm(mu, lo_p, hi_p, n, sd):
    """(alpha, omega, xi, clamped) matching ADP, the SD and the skew."""
    p_obs = (mu - lo_p) / (hi_p - lo_p)
    alpha, clamped = solve_shape(p_obs, 1.0 / (n + 1.0), n / (n + 1.0))
    delta = alpha / math.sqrt(1.0 + alpha * alpha)
    omega = sd / math.sqrt(1.0 - 2.0 * delta * delta / math.pi)
    xi = mu - omega * delta * math.sqrt(2.0 / math.pi)
    return alpha, omega, xi, clamped


def draw_arrays(players):
    """Pack the per-player sampling constants into numpy arrays.

    Sampling uses the standard construction
        Z = delta*|U0| + sqrt(1-delta^2)*U1,   U0, U1 ~ N(0,1) iid
        X = xi + omega*Z
    which is exact for the skew normal and vectorises, so a whole board is
    two normal draws rather than 378 scipy calls.
    """
    delta = np.array([p["alpha"] for p in players], dtype=float)
    delta = delta / np.sqrt(1.0 + delta * delta)
    return {
        "xi": np.array([p["xi"] for p in players], dtype=float),
        "omega": np.array([p["omega"] for p in players], dtype=float),
        "delta": delta,
        "co_delta": np.sqrt(1.0 - delta * delta),
        "min_pick": np.array([p["min_pick"] for p in players], dtype=float),
    }


def load_players(path=None):
    """The ADP pool, off a board scraped by adp.py unless one is given."""
    path = path or adp.table()
    out = []
    with open(path, newline="", encoding="utf-8-sig") as f:
        rdr = csv.reader(f, delimiter="\t")
        next(rdr)
        for r in rdr:
            try:
                mu = float(r[ADP_C])
                lo, hi = float(r[MINP]), float(r[MAXP])
                n = float(r[NPICKS])
            except (TypeError, ValueError, IndexError):
                continue
            pos = {SRC_DEF: DEF, SRC_K: K}.get(r[POS], r[POS])
            if n < MIN_PICKS or pos not in DRAFTABLE:
                continue
            sigma = (hi - lo) / (0.61 * math.log(n) + 2.235)
            if sigma <= 0:
                continue
            alpha, omega, xi, clamped = fit_skewnorm(mu, lo, hi, n, sigma)
            out.append(
                {
                    "pid": r[PID],
                    "name": r[PLAYER],
                    "nfl": r[TEAM],
                    "pos": pos,
                    "adp": mu,
                    "sd": sigma,
                    "min_pick": lo,
                    "alpha": alpha,
                    "omega": omega,
                    "xi": xi,
                    "clamped": clamped,
                }
            )
    return out


def draw_slots(arrays, rng):
    """One draft's simulated draft slot for every player.

    Split out of run_draft so combined_draft.py can draw an ADP board per
    draft the same way this simulator does, without duplicating the skew
    normal construction.
    """
    m = len(arrays["xi"])
    z = arrays["delta"] * np.abs(rng.standard_normal(m)) + arrays[
        "co_delta"
    ] * rng.standard_normal(m)
    sims = arrays["xi"] + arrays["omega"] * z
    np.maximum(sims, arrays["min_pick"], out=sims)  # never before Min Pick
    return sims


def run_draft(players, arrays, seed, lg):
    sims = draw_slots(arrays, np.random.default_rng(seed))
    board = np.argsort(sims, kind="stable")

    # The board as a plain list of (player, simulated slot), best first, from
    # which a pick is deleted as it is made. Scanning a list that only holds
    # undrafted players beats rescanning the whole board behind a "already
    # gone?" test: by the last round that test was rejecting ~190 entries
    # before reaching a live one. Rounding the slots here also keeps float()
    # and round() out of the pick loop, and .tolist() hands back Python ints
    # and floats rather than numpy scalars.
    alive = [
        (players[i], r)
        for i, r in zip(
            board.tolist(), np.round(sims, 2)[board].tolist(), strict=True
        )
    ]

    counts = [dict.fromkeys(DRAFTABLE, 0) for _ in range(lg.n_teams)]
    taken = [0] * lg.n_teams
    picks = []

    for overall, (rd, team) in enumerate(lg.snake_order(), start=1):
        c = counts[team]
        t = taken[team]
        for j, (p, sim) in enumerate(alive):
            pos = p["pos"]
            if not lg.legal(c, pos, t):
                continue
            del alive[j]
            c[pos] += 1
            taken[team] = t + 1
            picks.append(
                {
                    "round": rd,
                    "overall": overall,
                    "team": team + 1,
                    "pid": p["pid"],
                    "name": p["name"],
                    "nfl": p["nfl"],
                    "pos": pos,
                    "adp": p["adp"],
                    "sim": sim,
                }
            )
            break
        else:
            raise RuntimeError(f"no legal player at overall pick {overall}")
    return picks, counts


# Stream picks to Parquet in row-group batches -- at 10k drafts the full log
# is ~1.7M rows, too big to hold in memory just to write it once at the end.
SCHEMA = pa.schema(
    [
        ("draft", pa.int32()),
        ("round", pa.int16()),
        ("overall", pa.int16()),
        ("team", pa.int16()),
        ("pid", pa.string()),
        ("name", pa.string()),
        ("nfl", pa.string()),
        ("pos", pa.string()),
        ("adp", pa.float32()),
        ("sim", pa.float32()),
    ]
)
FIELDS = (
    "round",
    "overall",
    "team",
    "pid",
    "name",
    "nfl",
    "pos",
    "adp",
    "sim",
)
BATCH_DRAFTS = 250  # drafts buffered per row group
SUMMARY_COLS = [
    "Player_ID",
    "Player",
    "Team",
    "Pos",
    "ADP",
    "Times_Drafted",
    "Mean_Pick",
    "Pct_Drafted",
    "Min_Pick",
    "Max_Pick",
]


def collect(buf, agg, meta, d, picks):
    """Add one draft's picks to the parquet buffer and the totals."""
    for pk in picks:
        buf["draft"].append(d)
        for k in FIELDS:
            buf[k].append(pk[k])
        a = agg.get(pk["pid"])
        o = pk["overall"]
        if a is None:
            agg[pk["pid"]] = [1, o, o, o]
            meta[pk["pid"]] = pk
        else:
            a[0] += 1
            a[1] += o
            a[2] = min(a[2], o)
            a[3] = max(a[3], o)


def simulate(players, arrays, lg, path="draft_results.parquet"):
    """Run every draft, streaming the picks out. Returns the totals."""
    agg, meta = {}, {}  # pid -> [n, sum(overall), min, max] / one pick
    buf = {f.name: [] for f in SCHEMA}
    with pq.ParquetWriter(path, SCHEMA, compression="zstd") as writer:
        for d in range(1, N_DRAFTS + 1):
            picks, counts = run_draft(players, arrays, d, lg)
            lg.check(counts)
            collect(buf, agg, meta, d, picks)
            if d % BATCH_DRAFTS == 0 or d == N_DRAFTS:
                writer.write_table(pa.table(buf, schema=SCHEMA))
                buf = {f.name: [] for f in SCHEMA}
    return agg, meta


def summarise(agg, meta, path):
    """Where each player went over the whole run, earliest first."""
    summary = []
    for pid, (n, tot, lo, hi) in agg.items():
        m = meta[pid]
        summary.append(
            (
                pid,
                m["name"],
                m["nfl"],
                m["pos"],
                m["adp"],
                n,
                round(tot / n, 2),
                round(n / N_DRAFTS, 4),
                lo,
                hi,
            )
        )
    summary.sort(key=lambda r: r[6])
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(SUMMARY_COLS)
        w.writerows(summary)
    return summary


def main(lg=DEFAULT, summary_out=None, adp_path=None):
    players = load_players(adp_path)
    arrays = draw_arrays(players)
    agg, meta = simulate(players, arrays, lg)
    SUMMARY_OUT = summary_out or "draft_player_summary.csv"
    summary = summarise(agg, meta, SUMMARY_OUT)

    n_clamped = sum(p["clamped"] for p in players)
    print(f"pool: {len(players)} players ({n_clamped} with clamped skew)")
    picks_per = lg.n_teams * lg.roster
    print(
        f"{lg.name}: {N_DRAFTS} drafts x {picks_per} picks = "
        f"{N_DRAFTS * picks_per} rows -> draft_results.parquet"
    )
    print(f"all rosters legal; {len(summary)} unique players -> {SUMMARY_OUT}")


if __name__ == "__main__":
    import argparse

    ap = adp.add_adp_args(
        add_league_args(
            argparse.ArgumentParser(description=__doc__.split("\n")[0])
        )
    )
    ap.add_argument("--summary-out", default=None)
    _a = ap.parse_args()
    main(league_from_args(_a), _a.summary_out, adp.path_from_args(_a))
