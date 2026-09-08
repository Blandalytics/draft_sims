"""Snake drafts off a combined VOR/ADP board that is redrawn every draft.

Both source simulators randomise their board once per draft, and this one
inherits that rather than jittering a fixed ranking:

  * draft_sim.draw_slots draws each player a simulated draft slot from his
    fitted skew normal, clipped at his observed Min Pick. Sorting those slots
    ascending gives that draft's ADP rank.
  * vor_draft_sim.draw_points samples every scored component stat for every
    player and scores the draw. Replacement levels are recomputed from that
    same sample -- VOR is only meaningful against the sampled board -- and
    sorting the resulting positional VOR descending gives that draft's VOR
    rank.

Every team then blends those two ranks on its own terms. Each draft, each
team draws a VOR weight

    w ~ Uniform(1/3, 2/3),        sim weight = 1 - w

re-rolled for every draft, and drafts down its own list,

    score = w * vor_rank + (1 - w) * sim_rank,   sorted ascending

taking the best player still on the board that its roster can legally hold.
So the twelve teams share one draft's ranks but disagree about how to read
them -- one leans on projected value, the next on where the market says a
player goes -- and a player's spot moves from draft to draft for three
compounding reasons: the market's uncertainty about where he goes, the
projections' uncertainty about what he scores, and which team is on the clock.

A player only one source ranks -- a kicker the ADP pool never lists, someone
with no projected stat line -- is scored on the other as N + 1, one slot past
that board's last player. Pass --matched-only to draft only players both
sources rank. This mirrors combine_ranks.py, which runs the same blend over
the two simulators' aggregate summaries instead of over per-draft draws.

The league comes from this script's own arguments (--teams, --roster, --qb,
--rb, --wr, --te, --flex; defaulting to 12 teams, 15 spots, 1 QB / 2 RB /
3 WR / 1 TE / 1 FLEX / 1 DST / 1 K / 5 bench) and is handed to both sources as
well as used for the draft itself. That matters because a VOR rank is priced
against league-derived replacement levels: rank the board in one league and
draft it in another and the ranks mean nothing. One league in, one league
throughout.
"""

import argparse
import csv
import os

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from . import adp, draft_sim, vor_draft_sim
from .adp import add_adp_args
from .combine_ranks import first_last
from .league import DRAFTABLE, add_league_args, league_from_args
from .projections import add_projection_args
from .projections import from_args as projections_from_args

OUT_PICKS = "combined_draft_results.parquet"
OUT_SUMM = "combined_draft_summary.csv"

N_DRAFTS = 10000
SEED = 20260906
W_LO, W_HI = 1.0 / 3.0, 2.0 / 3.0  # each team's VOR weight is drawn
# from this range
BATCH_DRAFTS = 250  # drafts buffered per parquet row group


class Board:
    """Both simulators' machinery, loaded once, joined on ADP player id.

    Player metadata and the pid -> row maps are fixed for the whole run; only
    the two rank vectors change per draft. The pool is built once so a draft
    costs two draws, two argsorts and a blend, with no reallocation.
    """

    def __init__(self, lg, matched_only=False, adp_path=None, source=None):
        self.lg = lg
        # ADP side: one fitted skew normal per player, as draft_sim fits
        # them. `adp_path` is resolved once by whoever owns the run and
        # handed down, so a pool of workers all read the one board its
        # parent scraped rather than each scraping its own.
        self.sim_players = draft_sim.load_players(adp_path)
        self.arrays = draft_sim.draw_arrays(self.sim_players)

        # VOR side: component stat matrices, as vor_draft_sim builds them
        (
            self.vor_players,
            self.pos_of,
            self.MU,
            self.SD,
            self.W,
            _cols,
            _dropped,
        ) = vor_draft_sim.load_board(source)
        self.n_vor = len(self.vor_players)
        self.n_sim = len(self.sim_players)
        # replacement lookup is per-position, so map it onto players once
        self._pos_index = [self.pos_of[i] for i in range(self.n_vor)]

        # Join on ADP player id. A projected player carrying no pid cannot
        # match anything on the ADP side, so he keys on his own projection id
        # instead -- that makes him vor-only, exactly like a player whose pid
        # simply is not in the ADP pool, rather than dropping him silently.
        sim_by_pid = {p["pid"]: i for i, p in enumerate(self.sim_players)}

        # A pid is meant to name one player, and the projections give the same
        # one to two: 21126 is Jayden Daniels in ADP.tsv, and the projections
        # hang it on both him and Jalon Daniels. Built as a dict comprehension
        # this join keeps whichever came last, which drops a top-five quarter-
        # back out of the pool without a word and hands his ADP to a stranger
        # projected for five points. So a contested pid goes to the row the
        # ADP pool names, and the loser keys on his own projection id --
        # vor-only, exactly like a row that carries no pid at all. Both
        # players stay in the pool; only the market rank is contested.
        vor_by_pid, self.pid_clashes = {}, []
        for i, p in enumerate(self.vor_players):
            key = p["pid"] or "proj:" + p["id"]
            j = vor_by_pid.get(key)
            if j is None:
                vor_by_pid[key] = i
                continue
            si = sim_by_pid.get(key)
            named = first_last(self.sim_players[si]["name"]) if si else None
            win, lose = (i, j) if p["name"] == named else (j, i)
            vor_by_pid[key] = win
            vor_by_pid["proj:" + self.vor_players[lose]["id"]] = lose
            self.pid_clashes.append(
                (
                    key,
                    self.vor_players[win]["name"],
                    self.vor_players[lose]["name"],
                )
            )
        pids = (
            set(vor_by_pid) & set(sim_by_pid)
            if matched_only
            else set(vor_by_pid) | set(sim_by_pid)
        )

        # sorted pids give a stable, deterministic pool order, which is also
        # what score ties fall back on
        self.entries, vrow, srow = [], [], []
        for pid in sorted(pids):
            vi, si = vor_by_pid.get(pid), sim_by_pid.get(pid)
            v = self.vor_players[vi] if vi is not None else None
            s = self.sim_players[si] if si is not None else None
            pos = (v or s)["pos"]
            if pos not in DRAFTABLE:
                continue
            adp = v["adp"] if v and v["adp"] else (s["adp"] if s else "")
            self.entries.append(
                {
                    "pid": "" if pid.startswith("proj:") else pid,
                    "name": v["name"] if v else first_last(s["name"]),
                    "nfl": v["team"] if v else s["nfl"],
                    "pos": pos,
                    "adp": float(adp) if adp else float("nan"),
                    "source": "both" if v and s else ("vor" if v else "sim"),
                }
            )
            vrow.append(-1 if vi is None else vi)
            srow.append(-1 if si is None else si)
        self.vor_row = np.array(vrow, dtype=np.int64)
        self.sim_row = np.array(srow, dtype=np.int64)

    def draw(self, rng):
        """One draft's VOR rank and ADP rank, per pooled player."""
        # ADP rank: ascending simulated slot, 1..N over that simulator's pool
        slots = draft_sim.draw_slots(self.arrays, rng)
        sim_rank = np.empty(self.n_sim, dtype=np.float64)
        sim_rank[np.argsort(slots, kind="stable")] = np.arange(
            1, self.n_sim + 1
        )

        # VOR rank: descending positional VOR, against this same sample's
        # recomputed replacement levels
        pts = vor_draft_sim.draw_points(rng, self.MU, self.SD, self.W)
        repl = vor_draft_sim.replacement_levels(
            pts.tolist(), self.pos_of, self.lg
        )
        vor = pts - np.array([repl[p] for p in self._pos_index])
        vor_rank = np.empty(self.n_vor, dtype=np.float64)
        vor_rank[np.argsort(-vor, kind="stable")] = np.arange(
            1, self.n_vor + 1
        )

        # unranked on one side -> one slot past that board's last player
        vr = np.where(
            self.vor_row >= 0, vor_rank[self.vor_row], self.n_vor + 1.0
        )
        sr = np.where(
            self.sim_row >= 0, sim_rank[self.sim_row], self.n_sim + 1.0
        )
        return vr, sr


def team_boards(vr, sr, weights):
    """Each team's blended scores and its board order over the shared ranks.

    One (teams x players) matrix rather than a sort per team: the twelve
    blends are the same two rank vectors under different weights.
    """
    w = weights[:, None]
    scores = w * vr[None, :] + (1.0 - w) * sr[None, :]
    # ties fall back on pool order, which is sorted by pid
    return scores, np.argsort(scores, axis=1, kind="stable").tolist()


def run_draft(entries, orders, lg):
    """Run the draft, each team taking from its own board in `orders`.

    Returns (pool index, round, overall, team, rank on that team's board) per
    pick. Player detail stays in `entries` rather than being copied per pick.

    Boards differ per team, so a single shared "undrafted only" list is no
    longer possible. Instead each team keeps a pointer into its own board that
    walks forward past players already gone; since a board is fixed for the
    draft and picks only ever remove players, that pointer never rewinds, so
    the leading run of drafted players is skipped rather than rescanned.
    """
    n = len(entries)
    gone = bytearray(n)
    head = [0] * lg.n_teams
    counts = [dict.fromkeys(DRAFTABLE, 0) for _ in range(lg.n_teams)]
    taken = [0] * lg.n_teams
    picks = []

    for overall, (rd, team) in enumerate(lg.snake_order(), start=1):
        board = orders[team]
        c, t = counts[team], taken[team]
        h = head[team]
        while h < n and gone[board[h]]:
            h += 1
        head[team] = h
        for j in range(h, n):
            i = board[j]
            if gone[i] or not lg.legal(c, entries[i]["pos"], t):
                continue
            gone[i] = 1
            c[entries[i]["pos"]] += 1
            taken[team] = t + 1
            picks.append((i, rd, overall, team + 1, j + 1))
            break
        else:
            raise RuntimeError("no legal player at overall pick %d" % overall)
    return picks, counts


SCHEMA = pa.schema(
    [
        ("draft", pa.int32()),
        ("round", pa.int16()),
        ("overall", pa.int16()),
        ("team", pa.int16()),
        ("pid", pa.string()),
        ("player", pa.string()),
        ("nfl_team", pa.string()),
        ("pos", pa.string()),
        ("adp", pa.float32()),
        ("board_rank", pa.int16()),
        ("score", pa.float32()),
        ("vor_weight", pa.float32()),
        ("vor_rank", pa.float32()),
        ("sim_rank", pa.float32()),
    ]
)
COLS = [f.name for f in SCHEMA]


def parse_args():
    ap = add_projection_args(
        add_adp_args(
            add_league_args(
                argparse.ArgumentParser(
                    description=__doc__,
                    formatter_class=argparse.RawDescriptionHelpFormatter,
                )
            )
        )
    )
    ap.add_argument("--drafts", type=int, default=N_DRAFTS)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument(
        "--vor-weight-lo",
        type=float,
        default=W_LO,
        help="low end of each team's VOR weight draw",
    )
    ap.add_argument(
        "--vor-weight-hi",
        type=float,
        default=W_HI,
        help="high end; set equal to --vor-weight-lo for a fixed "
        "weight shared by every team",
    )
    ap.add_argument(
        "--matched-only",
        action="store_true",
        help="draft only players both simulators rank",
    )
    ap.add_argument("--picks-out", default=OUT_PICKS)
    ap.add_argument("--summary-out", default=OUT_SUMM)
    a = ap.parse_args()
    if not 0.0 <= a.vor_weight_lo <= a.vor_weight_hi <= 1.0:
        raise SystemExit("need 0 <= --vor-weight-lo <= --vor-weight-hi <= 1")
    return a


def describe(board, lg, a):
    """What this run is about to do, before it spends ten thousand drafts."""
    src = [e["source"] for e in board.entries]
    print(
        "pool: %d players (both %d | vor only %d | sim only %d)"
        % (
            len(board.entries),
            src.count("both"),
            src.count("vor"),
            src.count("sim"),
        )
    )
    for key, win, lose in board.pid_clashes:
        print(
            "pid %s claimed twice: kept %s, demoted %s to projection only"
            % (key, win, lose)
        )
    print(
        "ranks redrawn per draft: %d ADP slots, %d VOR boards"
        % (board.n_sim, board.n_vor)
    )
    print(
        "per-team vor weight ~ Uniform(%.2f, %.2f), re-rolled each draft"
        % (a.vor_weight_lo, a.vor_weight_hi)
    )
    print(
        "%s: %d teams x %d rounds = %d picks per draft, %d drafts\n"
        % (lg.name, lg.n_teams, lg.roster, lg.n_teams * lg.roster, a.drafts)
    )


def collect(buf, agg, d, picks, entries, scores, weights, vr, sr):
    """Add one draft's picks to the parquet buffer and the totals."""
    for i, rd, overall, team, cr in picks:
        e = entries[i]
        for col, val in (
            ("draft", d),
            ("round", rd),
            ("overall", overall),
            ("team", team),
            ("pid", e["pid"]),
            ("player", e["name"]),
            ("nfl_team", e["nfl"]),
            ("pos", e["pos"]),
            ("adp", e["adp"]),
            ("board_rank", cr),
            ("score", scores[team - 1, i]),
            ("vor_weight", weights[team - 1]),
            ("vor_rank", vr[i]),
            ("sim_rank", sr[i]),
        ):
            buf[col].append(val)
        g = agg.get(i)
        if g is None:
            agg[i] = [1, overall, overall, overall, cr]
        else:
            g[0] += 1
            g[1] += overall
            g[4] += cr
            g[2] = min(g[2], overall)
            g[3] = max(g[3], overall)


def simulate(board, lg, a):
    """Run every draft, streaming the picks out. Returns the totals.

    The pick log at ten thousand drafts is 1.8M rows, too big to hold just to
    write it once, so it goes out in row groups as it is made and only the
    per-player totals stay in memory.
    """
    rng = np.random.default_rng(a.seed)
    agg = {}  # pool index -> [n, sum overall, min, max,
    buf = {c: [] for c in COLS}  #               sum board rank]
    last = None
    with pq.ParquetWriter(a.picks_out, SCHEMA, compression="zstd") as writer:
        for d in range(1, a.drafts + 1):
            vr, sr = board.draw(rng)
            weights = rng.uniform(a.vor_weight_lo, a.vor_weight_hi, lg.n_teams)
            scores, orders = team_boards(vr, sr, weights)
            picks, counts = run_draft(board.entries, orders, lg)
            lg.check(counts)
            last = (picks, vr, sr, weights)
            collect(buf, agg, d, picks, board.entries, scores, weights, vr, sr)
            if d % BATCH_DRAFTS == 0 or d == a.drafts:
                writer.write_table(pa.table(buf, schema=SCHEMA))
                buf = {c: [] for c in COLS}
            if a.drafts > 20 and d % max(1, a.drafts // 10) == 0:
                print(
                    "  %5d / %d drafts (%.0f%%)"
                    % (d, a.drafts, 100 * d / a.drafts)
                )
    return agg, last


def summarise(agg, entries, drafts, path):
    """Per-player draft position over the whole run, written out best first."""
    summ = []
    for i, (cnt, tot_o, lo, hi, tot_r) in agg.items():
        e = entries[i]
        summ.append(
            {
                "pid": e["pid"],
                "player": e["name"],
                "pos": e["pos"],
                "nfl_team": e["nfl"],
                "adp": e["adp"],
                "source": e["source"],
                "times_drafted": cnt,
                "pct_drafted": round(cnt / drafts, 4),
                "mean_pick": round(tot_o / cnt, 2),
                "min_pick": lo,
                "max_pick": hi,
                "mean_board_rank": round(
                    tot_r / cnt, 2
                ),  # on the taker's board
            }
        )
    summ.sort(key=lambda r: r["mean_pick"])
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summ[0].keys()))
        w.writeheader()
        w.writerows(summ)
    return summ


def show_one_draft(entries, lg, last):
    """A single draft is a board to read, not a distribution to summarise."""
    picks, vr, sr, weights = last
    print(
        "\nteam vor weights: %s"
        % "  ".join("T%d %.2f" % (t + 1, w) for t, w in enumerate(weights))
    )
    for rd in range(1, lg.roster + 1):
        print("\nRound %d" % rd)
        for i, r, overall, team, _cr in picks:
            if r != rd:
                continue
            e = entries[i]
            print(
                "  %3d.%02d  T%-2d w%.2f  %-24s %-4s %-4s  "
                "vor %5.0f  sim %5.0f"
                % (
                    rd,
                    (overall - 1) % lg.n_teams + 1,
                    team,
                    weights[team - 1],
                    e["name"][:24],
                    e["pos"],
                    e["nfl"],
                    vr[i],
                    sr[i],
                )
            )


def show_earliest(summ):
    print(
        "\n%4s  %-24s %-4s %-4s %8s %6s %6s %7s"
        % ("#", "player", "pos", "nfl", "mean", "min", "max", "pct")
    )
    for k, r in enumerate(summ[:20], start=1):
        print(
            "%4d  %-24s %-4s %-4s %8.2f %6d %6d %6.1f%%"
            % (
                k,
                r["player"][:24],
                r["pos"],
                r["nfl_team"],
                r["mean_pick"],
                r["min_pick"],
                r["max_pick"],
                100 * r["pct_drafted"],
            )
        )


def main():
    a = parse_args()
    lg = league_from_args(a)
    board_path = adp.path_from_args(a)
    board = Board(
        lg, a.matched_only, board_path, projections_from_args(a, board_path)
    )
    describe(board, lg, a)

    agg, last = simulate(board, lg, a)
    summ = summarise(agg, board.entries, a.drafts, a.summary_out)
    print(
        "\nwrote %s (%d picks, %.1f MB)"
        % (
            a.picks_out,
            a.drafts * lg.n_teams * lg.roster,
            os.path.getsize(a.picks_out) / 1048576,
        )
    )
    print(
        "wrote %s (%d players drafted at least once)"
        % (a.summary_out, len(summ))
    )

    if a.drafts == 1:
        show_one_draft(board.entries, lg, last)
    else:
        show_earliest(summ)


if __name__ == "__main__":
    main()
