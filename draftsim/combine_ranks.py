"""Consolidate the two simulations' player ranks into one weighted board.

Each simulator already writes a per-player summary whose `mean_pick` is that
player's average draft slot over its 10,000 drafts:

    vor_draft_sim.py  -> vor_draft_summary.csv     (10 teams x 15, VOR board)
    draft_sim.py      -> draft_player_summary.csv  (12 teams x 16, ADP board)

The two mean picks are not comparable -- different league sizes, different
roster shapes, one measured in projected value and the other in market ADP --
so the common currency is the *rank* each sim implies: sort its players by
mean pick, ascending, and number them 1..N. Those two rank columns are then
blended,

    combined = w_vor * vor_rank + w_sim * sim_rank

with the weights defaulting to 60% VOR and 40% ADP, and re-ranked 1..N.

A player drafted by one sim but not the other (a kicker or defense the other
pool never sees, a name only one source carries) is scored on that sim as
N + 1 -- one slot past the last player it ever drafted. That is a penalty, not
a neutral value: being undrafted in 10,000 drafts is information. Pass
--matched-only to restrict the board to players both sims rank instead.
"""

import argparse
import csv

VOR_SRC = "vor_draft_summary.csv"  # written by vor_draft_sim
SIM_SRC = "draft_player_summary.csv"  # written by draft_sim
OUT = "combined_ranks.csv"

W_VOR = 0.6  # weight on vor_draft_sim's rank
W_SIM = 0.4  # weight on draft_sim's rank


def first_last(name):
    """ADP.tsv writes "Last, First"; the projections write "First Last"."""
    if "," in name:
        last, _, first = name.partition(",")
        return (first.strip() + " " + last.strip()).strip()
    return name.strip()


def _rank(rows, key):
    """Number rows 1..N by ascending `key`, returning {pid: (rank, record)}."""
    rows.sort(key=lambda r: float(r[key]))
    return {r["pid"]: (i, r) for i, r in enumerate(rows, start=1)}


def load_vor(path=VOR_SRC):
    with open(path, encoding="utf-8-sig") as f:
        rows = [
            {
                "pid": r["pid"],
                "name": r["player"],
                "pos": r["pos"],
                "nfl": r["nfl_team"],
                "adp": r["adp"],
                "mean_pick": r["mean_pick"],
                "pct": r["pct_drafted"],
            }
            for r in csv.DictReader(f)
            if r["pid"]
        ]
    return _rank(rows, "mean_pick")


def load_sim(path=SIM_SRC):
    with open(path, encoding="utf-8-sig") as f:
        rdr = csv.DictReader(f)
        if "Player_ID" not in rdr.fieldnames:
            raise SystemExit(
                "%s has no Player_ID column -- rerun draft_sim.py to "
                "regenerate it" % path
            )
        rows = [
            {
                "pid": r["Player_ID"],
                "name": first_last(r["Player"]),
                "pos": r["Pos"],
                "nfl": r["Team"],
                "adp": r["ADP"],
                "mean_pick": r["Mean_Pick"],
                "pct": r["Pct_Drafted"],
            }
            for r in rdr
            if r["Player_ID"]
        ]
    return _rank(rows, "mean_pick")


def combine(vor, sim, w_vor=W_VOR, w_sim=W_SIM, matched_only=False):
    """Blend the two rank columns into one board, best first."""
    total = w_vor + w_sim
    if total <= 0:
        raise ValueError("weights must sum to a positive number")
    w_vor, w_sim = w_vor / total, w_sim / total  # normalise

    pids = set(vor) & set(sim) if matched_only else set(vor) | set(sim)
    miss_vor, miss_sim = (
        len(vor) + 1,
        len(sim) + 1,
    )  # one past the last drafted

    board = []
    for pid in pids:
        v, s = vor.get(pid), sim.get(pid)
        rec = (v or s)[1]
        vr = v[0] if v else miss_vor
        sr = s[0] if s else miss_sim
        board.append(
            {
                "pid": pid,
                "player": rec["name"],
                "pos": rec["pos"],
                "nfl_team": rec["nfl"],
                "adp": rec["adp"],
                "vor_rank": vr if v else "",
                "vor_mean_pick": v[1]["mean_pick"] if v else "",
                "sim_rank": sr if s else "",
                "sim_mean_pick": s[1]["mean_pick"] if s else "",
                "source": "both" if v and s else ("vor" if v else "sim"),
                "score": round(w_vor * vr + w_sim * sr, 4),
            }
        )
    board.sort(key=lambda r: (r["score"], r["player"]))
    for i, r in enumerate(board, start=1):
        r["combined_rank"] = i
    return board


FIELDS = [
    "combined_rank",
    "pid",
    "player",
    "pos",
    "nfl_team",
    "adp",
    "score",
    "vor_rank",
    "sim_rank",
    "vor_mean_pick",
    "sim_mean_pick",
    "source",
]


def write(board, path=OUT):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows({k: r[k] for k in FIELDS} for r in board)


def build(w_vor=W_VOR, w_sim=W_SIM, matched_only=False):
    return combine(load_vor(), load_sim(), w_vor, w_sim, matched_only)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--vor-weight", type=float, default=W_VOR)
    ap.add_argument("--sim-weight", type=float, default=W_SIM)
    ap.add_argument(
        "--matched-only",
        action="store_true",
        help="keep only players both simulations rank",
    )
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--top", type=int, default=25, help="rows to print")
    a = ap.parse_args()

    vor, sim = load_vor(), load_sim()
    board = combine(vor, sim, a.vor_weight, a.sim_weight, a.matched_only)
    write(board, a.out)

    tot = a.vor_weight + a.sim_weight
    print("vor_draft_summary.csv:    %d ranked players" % len(vor))
    print("draft_player_summary.csv: %d ranked players" % len(sim))
    print(
        "both: %d | vor only: %d | sim only: %d"
        % (
            sum(r["source"] == "both" for r in board),
            sum(r["source"] == "vor" for r in board),
            sum(r["source"] == "sim" for r in board),
        )
    )
    print(
        "weights: %.0f%% vor / %.0f%% sim -> %d players\n"
        % (100 * a.vor_weight / tot, 100 * a.sim_weight / tot, len(board))
    )
    print(
        "%4s  %-24s %-4s %5s %5s %8s"
        % ("#", "player", "pos", "vor", "sim", "score")
    )
    for r in board[: a.top]:
        print(
            "%4d  %-24s %-4s %5s %5s %8.2f"
            % (
                r["combined_rank"],
                r["player"][:24],
                r["pos"],
                r["vor_rank"] or "-",
                r["sim_rank"] or "-",
                r["score"],
            )
        )
    print("\nwrote %s" % a.out)


if __name__ == "__main__":
    main()
