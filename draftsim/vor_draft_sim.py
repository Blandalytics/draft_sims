"""Simulate snake drafts off sampled projected points, picking by VOR.

Each draft draws one simulated season for every player by sampling each
scored component stat independently,
    stat ~ Normal(stat, stat_sd), clipped at 0,
from the `average` avg_type lines in projections_stats.csv, then scoring the
draw with Yahoo default values (0.5 PPR). So a player's points come from his
own passing/rushing/receiving draws rather than from one aggregate number,
and each draft gets its own board. Replacement levels are then recomputed from
that draft's sampled points -- VOR is only meaningful against the sampled
board, not against the static projection -- and teams take the highest-VOR
player their roster can still legally hold.

League comes from league.py and defaults to 12 teams, snake order, 15 roster
spots each (180 picks):
    1 QB, 2 RB, 3 WR, 1 TE, 1 FLEX (RB/WR/TE), 1 K, 1 DST, 5 bench
    QB and TE capped at starters + 1; exactly one K and one DST per roster,
    so the bench is effectively QB/RB/WR/TE

Draft-order policy, on top of pure roster legality:
  * at most 1 bench player before the starting RB (2) and WR (2) slots are full
  * no bench QB or TE at all until those RB/WR starters are full
  * QB, TE, K and DST starters may be deferred as long as the roster can still
    be completed, so a team may fill bench spots ahead of them

Replacement level = N teams x that position's starters, recomputed per draft
from the sampled points -- at the default league:
    QB  12th   (1 x 12 teams)      RB  24th   (2 x 12)
    WR  36th   (3 x 12)            TE  12th   (1 x 12)
    K    3rd   DST  3rd            (pinned: both are streamed, and a
                                    roster-derived baseline inflates them)
    FLEX 12th best of the RB/WR/TE left over once each position's starting
         allotment is set aside (1 flex x 12 teams)

A player is valued against the baseline of the slot he would actually occupy.
An RB/WR/TE is worth his positional VOR only while his own starting slot is
still open; once those starters are filled the team pivots to the flex
baseline for him, on the flex spot and on the bench alike, since that is the
slot he is really competing for. A QB is always valued off the QB baseline.
"""

import csv
import os
from collections import Counter, defaultdict

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from . import PROJECTION_STATS, PROJECTIONS
from .league import (
    BASELINE_OVERRIDE,
    DEFAULT,
    DRAFTABLE,
    FLEX_POS,
    MAX_BENCH_BEFORE_RBWR,
    add_league_args,
    league_from_args,
)

SRC = PROJECTIONS
STATS_SRC = PROJECTION_STATS
STATS_AVG_TYPE = "average"

# Yahoo default scoring, applied to the sampled component stats.
# help.yahoo.com/kb/default-league-settings-fantasy-football-sln6489.html
YAHOO_POINTS = {
    "pass_yds": 0.04,
    "pass_tds": 4,
    "pass_int": -1,
    "rush_yds": 0.1,
    "rush_tds": 6,
    "rec": 0.5,
    "rec_yds": 0.1,
    "rec_tds": 6,
    "fumbles_lost": -2,
    "fg_0019": 3,
    "fg_2029": 3,
    "fg_3039": 3,
    "fg_4049": 4,
    "fg_50": 5,
    "xp": 1,
    "dst_int": 2,
    "dst_fum_rec": 2,
    "dst_sacks": 1,
    "dst_safety": 2,
    "dst_td": 6,
}
OUT_PICKS = "vor_draft_results.parquet"
OUT_SUMM = "vor_draft_summary.csv"

SLOTS = ("START", "FLEX", "BENCH")
SLOT_CODE = {s: i for i, s in enumerate(SLOTS)}
BASES = ("pos", "flex")

N_DRAFTS = 10000
SEED = 20260905
# The league -- team count, roster size, starting lineup, flex -- comes from
# league.py, so this simulator and draft_sim.py cannot disagree about the
# league combined_draft.py blends their boards for. Overridable per run; see
# add_league_args.


def load_players():
    players = []
    with open(SRC, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            pos = r["position"]
            if pos not in DRAFTABLE:
                continue
            try:
                pts, sd = float(r["points"]), float(r["sd_pts"])
            except (ValueError, KeyError):
                continue
            players.append(
                {
                    "id": r["id"],
                    "pid": r.get("pid", ""),  # ADP.tsv player id
                    "name": (r["first_name"] + " " + r["last_name"]).strip(),
                    "pos": pos,
                    "team": r["team"],
                    "points": pts,
                    "sd_pts": sd,
                    # The projection exactly as published. load_board
                    # overwrites `points` with what this file's own
                    # components imply, which is what a VOR board must
                    # be priced on; anything reporting a projection
                    # rather than ranking on it wants the source's
                    # number -- see pick_sim.Engine.
                    "src_points": pts,
                    "adp": r.get("adp") or "",
                }
            )
    return players


def load_stats():
    """Component stat means and SDs, keyed by player id.

    Only the columns Yahoo actually scores are kept, and only those present in
    the file. A blank stat is 0; a blank SD is 0, i.e. that component is
    treated as certain rather than dropped.
    """
    with open(STATS_SRC, encoding="utf-8-sig") as f:
        rows = [
            r for r in csv.DictReader(f) if r["avg_type"] == STATS_AVG_TYPE
        ]
    cols = [c for c in YAHOO_POINTS if c in rows[0] and c + "_sd" in rows[0]]

    def num(v):
        try:
            return float(v)
        except (TypeError, ValueError):
            return 0.0

    out = {}
    for r in rows:
        out[r["id"]] = (
            [num(r[c]) for c in cols],
            [num(r[c + "_sd"]) for c in cols],
        )
    return cols, out


def replacement_levels(sampled, pos_of, lg=DEFAULT):
    """Per-position and flex replacement points, from one draft's sample.

    Baselines are league-derived -- N teams x that position's starters -- so
    they only mean anything in the league the board will actually be drafted
    in. combined_draft.py passes its own league in for exactly that reason.
    """
    by_pos = defaultdict(list)
    for i, p in enumerate(sampled):
        by_pos[pos_of[i]].append(p)
    for v in by_pos.values():
        v.sort(reverse=True)

    repl = {}
    for pos, n_start in lg.all_starters.items():
        k = BASELINE_OVERRIDE.get(pos, n_start * lg.n_teams)
        v = by_pos[pos]
        repl[pos] = v[k - 1] if len(v) >= k else (v[-1] if v else 0.0)

    # flex pool: everyone beyond his own position's starting allotment
    pool = []
    for pos in FLEX_POS:
        pool += by_pos[pos][lg.all_starters[pos] * lg.n_teams :]
    pool.sort(reverse=True)
    k = lg.n_flex * lg.n_teams
    repl["FLEX"] = (
        pool[k - 1] if len(pool) >= k else (pool[-1] if pool else 0.0)
    )
    return repl


class Roster:
    """Roster state, with the derived slot counts cached.

    `flex_used` and `unmet` only change when a player is added, so they are
    recomputed once per pick in add() rather than once per candidate.
    """

    __slots__ = ("lg", "st", "count", "n", "flex_used", "unmet", "bench")

    def __init__(self, lg):
        self.lg = lg
        self.st = lg.all_starters
        self.count = Counter()
        self.n = 0
        self.flex_used = 0
        self.unmet = sum(self.st.values()) + lg.n_flex
        self.bench = 0

    def rbwr_full(self):
        """Are the starting RB and WR slots both filled?"""
        return (
            self.count["RB"] >= self.st["RB"]
            and self.count["WR"] >= self.st["WR"]
        )

    def _refresh(self):
        extra = sum(max(0, self.count[p] - self.st[p]) for p in FLEX_POS)
        self.flex_used = min(extra, self.lg.n_flex)
        need = sum(max(0, self.st[p] - self.count[p]) for p in self.st)
        self.unmet = need + (self.lg.n_flex - self.flex_used)

    def slot_for(self, pos):
        """Which slot a player of `pos` would fill: START, FLEX or BENCH."""
        if self.count[pos] < self.st[pos]:
            return "START"
        if pos in FLEX_POS and self.flex_used < self.lg.n_flex:
            return "FLEX"
        return "BENCH"

    def helps(self, pos):
        """Would a player of `pos` reduce the mandatory-slot deficit?"""
        if self.count[pos] < self.st[pos]:
            return True
        return pos in FLEX_POS and self.flex_used < self.lg.n_flex

    def legal(self, pos):
        if self.n >= self.lg.roster:
            return False
        if self.count[pos] >= self.lg.caps.get(pos, self.lg.roster):
            return False
        # Draft-order policy while the starting RB and WR slots are still open:
        # no bench QB or TE at all, and at most one bench player of any kind.
        # A starter or the flex is always allowed.
        if not self.rbwr_full() and self.slot_for(pos) == "BENCH":
            if pos in ("QB", "TE"):
                return False
            if self.bench >= MAX_BENCH_BEFORE_RBWR:
                return False
        # reserve the closing picks for slots that must still be filled
        return not (
            (self.lg.roster - self.n) <= self.unmet and not self.helps(pos)
        )

    def add(self, pos):
        if self.slot_for(pos) == "BENCH":  # classify before counts change
            self.bench += 1
        self.count[pos] += 1
        self.n += 1
        self._refresh()


def load_board():
    """The player pool, its component stat matrices and the scoring weights.

    Split out of run() so combined_draft.py can draw a VOR board per draft the
    same way this simulator does. Each player's `points`/`sd_pts` are
    overwritten with the mean and SD implied by his components.
    """
    players = load_players()
    stat_cols, stats = load_stats()

    dropped = [p for p in players if p["id"] not in stats]
    players = [p for p in players if p["id"] in stats]
    pos_of = [p["pos"] for p in players]

    # component stat matrices: one row per player, one column per scored stat
    MU = np.array([stats[p["id"]][0] for p in players], dtype=np.float64)
    SD = np.array([stats[p["id"]][1] for p in players], dtype=np.float64)
    W = np.array([YAHOO_POINTS[c] for c in stat_cols], dtype=np.float64)

    # the projection each player's sampled points vary around, and the SD that
    # follows from the components (treating the components as independent)
    proj_pts = MU @ W
    proj_sd = np.sqrt((SD**2) @ (W**2))
    for p, mval, sval in zip(players, proj_pts, proj_sd, strict=True):
        p["points"], p["sd_pts"] = float(mval), float(sval)
    return players, pos_of, MU, SD, W, stat_cols, dropped


def draw_points(rng, MU, SD, W):
    """One draft's sampled season points for every player.

    Stats are counts and yardages, so a draw is clipped at 0 rather than
    allowed negative -- including the ones with negative point values.
    """
    return np.clip(rng.normal(MU, SD), 0.0, None) @ W


class Picks:
    """One column per field of the pick log, filled in pick order.

    Every per-player field -- name, position, nfl team, projection -- is
    recoverable from the player index, so a pick stores the index and nothing
    else, and the columns are expanded back out once at the end.
    """

    __slots__ = (
        "draft",
        "round",
        "pick",
        "team",
        "pidx",
        "slot",
        "basis",
        "sampled",
        "vor",
        "row",
    )

    def __init__(self, total):
        self.draft = np.empty(total, dtype=np.int32)
        self.round = np.empty(total, dtype=np.int8)
        self.pick = np.empty(total, dtype=np.int16)
        self.team = np.empty(total, dtype=np.int8)
        self.pidx = np.empty(total, dtype=np.int32)
        self.slot = np.empty(total, dtype=np.int8)
        self.basis = np.empty(total, dtype=np.int8)
        self.sampled = np.empty(total, dtype=np.float32)
        self.vor = np.empty(total, dtype=np.float32)
        self.row = 0

    def add(self, d, rnd, overall, team, i, slot, basis, pts, vor):
        r = self.row
        self.draft[r] = d
        self.round[r] = rnd
        self.pick[r] = overall
        self.team[r] = team + 1
        self.pidx[r] = i
        self.slot[r] = SLOT_CODE[slot]
        self.basis[r] = 1 if basis == "flex" else 0
        self.sampled[r] = pts
        self.vor[r] = vor
        self.row = r + 1


def best_pick(board, head, taken, R, vor_pos, vor_flex):
    """The highest-VOR player this roster may add, and how he is priced.

    Within a position both baselines are the sampled points minus a constant,
    so the two rank a position's players identically and one descending sort
    serves both. Legality depends only on position, so the leader at each is
    the only candidate worth pricing -- at most six per pick rather than the
    whole pool. Ties break on player index, matching a scan down the pool.
    """
    best_i, best_v, best_slot, best_basis = -1, -np.inf, None, None
    for pos, idx in board.items():
        h = head[pos]
        while h < len(idx) and taken[idx[h]]:
            h += 1
        head[pos] = h
        if h >= len(idx) or not R.legal(pos):
            continue
        i = idx[h]
        slot = R.slot_for(pos)
        # once his own starters are filled, an RB/WR/TE is competing for the
        # flex, so value him off the flex baseline from there on
        flex_priced = pos in FLEX_POS and slot != "START"
        v = vor_flex[i] if flex_priced else vor_pos[i]
        if v > best_v or (v == best_v and i < best_i):
            best_i, best_v, best_slot = i, v, slot
            best_basis = "flex" if flex_priced else "pos"
    return best_i, best_v, best_slot, best_basis


def one_draft(d, out, order, rng, pos_of, MU, SD, W, lg):
    """Draft the whole board once off a freshly sampled season."""
    n = len(pos_of)
    pts = draw_points(rng, MU, SD, W).tolist()
    repl = replacement_levels(pts, pos_of, lg)
    rosters = [Roster(lg) for _ in range(lg.n_teams)]
    taken = np.zeros(n, dtype=bool)

    # this draft's VOR for each player, against each baseline he could use
    vor_pos = [pts[i] - repl[pos_of[i]] for i in range(n)]
    vor_flex = [pts[i] - repl["FLEX"] for i in range(n)]
    board = {
        p: sorted(
            (i for i in range(n) if pos_of[i] == p), key=lambda i: (-pts[i], i)
        )
        for p in DRAFTABLE
    }
    head = dict.fromkeys(DRAFTABLE, 0)

    for overall, (rnd, team) in enumerate(order, start=1):
        R = rosters[team]
        i, v, slot, basis = best_pick(board, head, taken, R, vor_pos, vor_flex)
        if i < 0:
            raise RuntimeError(
                "draft %d pick %d: no legal player" % (d, overall)
            )
        taken[i] = True
        R.add(pos_of[i])
        out.add(d, rnd, overall, team, i, slot, basis, pts[i], v)
    return repl


def report_draft(d, repl):
    """Per-draft detail is unreadable past a couple of dozen drafts."""
    if N_DRAFTS <= 20:
        print(
            "  draft %2d: replacement QB %.1f | RB %.1f | WR %.1f | "
            "TE %.1f | FLEX %.1f"
            % (d, repl["QB"], repl["RB"], repl["WR"], repl["TE"], repl["FLEX"])
        )
    elif d % max(1, N_DRAFTS // 10) == 0:
        print("  %5d / %d drafts (%.0f%%)" % (d, N_DRAFTS, 100 * d / N_DRAFTS))


def pick_table(out, players, pos_of):
    """The pick log, with every per-player column dictionary-encoded.

    A player with no ADP.tsv match gets a null pid, and Parquet cannot store a
    null inside a dictionary, so it is encoded in the indices instead.
    """
    pids = pa.array([p["pid"] or "" for p in players])
    no_pid = np.array([not p["pid"] for p in players])
    proj = np.round(
        np.array([p["points"] for p in players], dtype=np.float32), 2
    )
    sdev = np.round(
        np.array([p["sd_pts"] for p in players], dtype=np.float32), 2
    )
    adps = np.array(
        [float(p["adp"]) if p["adp"] else np.nan for p in players],
        dtype=np.float32,
    )

    def dict_col(codes, values, mask=None):
        return pa.DictionaryArray.from_arrays(
            pa.array(codes, type=pa.int32(), mask=mask), values
        )

    idx = out.pidx
    return pa.table(
        {
            "draft": out.draft,
            "round": out.round,
            "pick": out.pick,
            "team": out.team,
            "pid": dict_col(idx, pids, mask=no_pid[idx]),
            "player": dict_col(idx, pa.array([p["name"] for p in players])),
            "pos": dict_col(idx, pa.array(pos_of)),
            "nfl_team": dict_col(idx, pa.array([p["team"] for p in players])),
            "slot": dict_col(out.slot, pa.array(SLOTS)),
            "vor_basis": dict_col(out.basis, pa.array(BASES)),
            "sampled_points": np.round(out.sampled, 2),
            "proj_points": proj[idx],
            "sd_pts": sdev[idx],
            "vor": np.round(out.vor, 2),
            "adp": adps[idx],
        }
    )


def summarise(out, players, path):
    """Where each player went, aggregated straight off the pick columns."""
    n = len(players)
    times = np.bincount(out.pidx, minlength=n)
    sum_pick = np.bincount(
        out.pidx, weights=out.pick.astype(np.float64), minlength=n
    )
    min_pick = np.full(n, np.iinfo(np.int32).max, dtype=np.int32)
    max_pick = np.zeros(n, dtype=np.int32)
    np.minimum.at(min_pick, out.pidx, out.pick)
    np.maximum.at(max_pick, out.pidx, out.pick)

    summ = [
        {
            "pid": p["pid"],
            "player": p["name"],
            "pos": p["pos"],
            "nfl_team": p["team"],
            "proj_points": round(p["points"], 2),
            "sd_pts": round(p["sd_pts"], 2),
            "adp": p["adp"],
            "times_drafted": int(times[i]),
            "pct_drafted": round(times[i] / N_DRAFTS, 3),
            "mean_pick": round(sum_pick[i] / times[i], 2),
            "min_pick": int(min_pick[i]),
            "max_pick": int(max_pick[i]),
        }
        for i, p in enumerate(players)
        if times[i]
    ]
    summ.sort(key=lambda r: r["mean_pick"])
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summ[0].keys()))
        w.writeheader()
        w.writerows(summ)
    return summ


def describe(players, pos_of, stat_cols, dropped, lg):
    print("pool: %d players -> %s" % (len(players), dict(Counter(pos_of))))
    print(
        "sampling %d scored stats per player: %s"
        % (len(stat_cols), ", ".join(stat_cols))
    )
    if dropped:
        print(
            "dropped %d players with no %s stat line"
            % (len(dropped), STATS_AVG_TYPE)
        )
    print(
        "%s: %d teams x %d spots = %d picks per draft, %d drafts\n"
        % (lg.name, lg.n_teams, lg.roster, lg.n_teams * lg.roster, N_DRAFTS)
    )


def run(lg=DEFAULT):
    players, pos_of, MU, SD, W, stat_cols, dropped = load_board()
    describe(players, pos_of, stat_cols, dropped, lg)

    rng = np.random.default_rng(SEED)
    order = list(lg.snake_order())
    out = Picks(N_DRAFTS * len(order))
    for d in range(1, N_DRAFTS + 1):
        repl = one_draft(d, out, order, rng, pos_of, MU, SD, W, lg)
        report_draft(d, repl)

    table = pick_table(out, players, pos_of)
    pq.write_table(table, OUT_PICKS, compression="zstd")
    summ = summarise(out, players, OUT_SUMM)
    print(
        "\nwrote %s (%d picks, %.1f MB)"
        % (OUT_PICKS, table.num_rows, os.path.getsize(OUT_PICKS) / 1048576)
    )
    print(
        "wrote %s (%d players drafted at least once)" % (OUT_SUMM, len(summ))
    )
    return table, summ


if __name__ == "__main__":
    import argparse

    ap = add_league_args(
        argparse.ArgumentParser(description=__doc__.split("\n")[0])
    )
    run(league_from_args(ap.parse_args()))
