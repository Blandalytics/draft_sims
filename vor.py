"""Replacement levels and VOR from Yahoo default starting rosters.

Yahoo default starters, 12 teams:
    1 QB, 2 RB, 2 WR, 1 TE, 1 FLEX (RB/WR/TE), 1 K, 1 DST

Replacement level for a position is the first player at that position who
would NOT hold a starting job if every team filled its lineup by ADP. Base
demand is teams * starters; the 12 flex spots are then handed to the best
remaining RB/WR/TE by ADP, which is what makes the RB/WR/TE baselines deeper
than their raw starter counts.

NOTE ON UNITS: ADP.tsv carries no projected points, so this cannot be a
points-based VOR. Value here is measured in draft capital:

    VOR_adp = ADP(replacement at pos) - ADP(player)

i.e. how many picks earlier than his own positional replacement a player
comes off the board. Higher is better; positive means he is a starter-grade
asset at that position. Swap `value` for projected points and the same
replacement indices give a conventional points VOR -- see points_vor().
"""
import csv, math

N_TEAMS = 12
STARTERS = {"QB": 1, "RB": 2, "WR": 2, "TE": 1, "K": 1, "DST": 1}
FLEX_POS = ("RB", "WR", "TE")
FLEX_SLOTS = 1
MIN_PICKS = 20
RENAME = {"TDSP": "DST", "TK": "K"}

RANK, PID, PLAYER, TEAM, POS, ADP_C, MINP, MAXP, _DIFF, NPICKS = range(10)


def load(path="ADP.tsv"):
    out = []
    with open(path, newline="", encoding="utf-8-sig") as f:
        rdr = csv.reader(f, delimiter="\t")
        next(rdr)
        for r in rdr:
            try:
                mu, n = float(r[ADP_C]), float(r[NPICKS])
            except (TypeError, ValueError, IndexError):
                continue
            pos = RENAME.get(r[POS], r[POS])
            if n < MIN_PICKS or pos not in STARTERS:
                continue
            out.append({"pid": r[PID], "name": r[PLAYER], "nfl": r[TEAM],
                        "pos": pos, "adp": mu})
    out.sort(key=lambda p: p["adp"])
    return out


def replacement_index(players):
    """Starters demanded at each position, base allocation plus flex."""
    demand = {p: N_TEAMS * s for p, s in STARTERS.items()}

    # hand the flex spots to the best RB/WR/TE not already covered by a
    # base starting slot, taking them in ADP order
    by_pos = {p: [q for q in players if q["pos"] == p] for p in STARTERS}
    leftovers = []
    for p in FLEX_POS:
        leftovers += by_pos[p][demand[p]:]
    leftovers.sort(key=lambda q: q["adp"])
    for q in leftovers[:N_TEAMS * FLEX_SLOTS]:
        demand[q["pos"]] += 1
    return demand, by_pos


def build(players):
    demand, by_pos = replacement_index(players)
    repl = {}
    for p, pool in by_pos.items():
        i = demand[p]                      # 0-based index of first non-starter
        repl[p] = pool[i]["adp"] if i < len(pool) else pool[-1]["adp"]
    for q in players:
        q["pos_rank"] = by_pos[q["pos"]].index(q) + 1
        q["repl_adp"] = repl[q["pos"]]
        q["vor_adp"] = round(repl[q["pos"]] - q["adp"], 2)
        q["starter"] = q["pos_rank"] <= demand[q["pos"]]
    return demand, repl


def points_vor(players, proj):
    """Same replacement indices, but valued in projected points.

    `proj` maps pid -> projected season points. Unused today; here so the
    baselines below can be reused the moment projections exist.
    """
    demand, by_pos = replacement_index(players)
    out = {}
    for p, pool in by_pos.items():
        ranked = sorted((q for q in pool if q["pid"] in proj),
                        key=lambda q: -proj[q["pid"]])
        if not ranked:
            continue
        i = min(demand[p], len(ranked) - 1)
        base = proj[ranked[i]["pid"]]
        for q in ranked:
            out[q["pid"]] = round(proj[q["pid"]] - base, 2)
    return out


players = load()
demand, repl = build(players)

print(f"pool: {len(players)} players, {N_TEAMS} teams\n")
print(f"{'pos':<5}{'starters':>9}{'w/ flex':>9}{'repl rank':>11}{'repl ADP':>10}")
for p in ("QB", "RB", "WR", "TE", "K", "DST"):
    print(f"{p:<5}{N_TEAMS*STARTERS[p]:>9}{demand[p]:>9}"
          f"{demand[p]+1:>11}{repl[p]:>10.2f}")
flex = {p: demand[p] - N_TEAMS * STARTERS[p] for p in FLEX_POS}
print(f"\nflex spots ({N_TEAMS * FLEX_SLOTS}) split: "
      + ", ".join(f"{p} {n}" for p, n in flex.items()))

players.sort(key=lambda q: -q["vor_adp"])
with open("vor.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["Player", "Team", "Pos", "ADP", "Pos_Rank", "Starter",
                "Replacement_ADP", "VOR_ADP"])
    for q in players:
        w.writerow([q["name"], q["nfl"], q["pos"], q["adp"], q["pos_rank"],
                    int(q["starter"]), q["repl_adp"], q["vor_adp"]])
print(f"\n{len(players)} rows -> vor.csv")
print(f"starter-grade players: {sum(q['starter'] for q in players)} "
      f"(= {N_TEAMS} teams x 9 starters = {N_TEAMS*9})")
