"""
Availability curves from ADP data.

Model: each player's draft slot ~ Normal(mu, sigma) where
    mu    = ADP
    sigma = (Max Pick - Min Pick) / (0.61 * ln(# Picks) + 2.235)
(the range-to-sigma estimator, d2 constant for a sample of size n).

"Available at pick k" = the player has not gone in picks 1..k-1
    P(available at k) = P(slot >= k) = 1 - Phi((k - 0.5 - mu) / sigma)
with a 0.5 continuity correction since slots are discrete.

Players with fewer than 20 recorded picks are ignored.
Rows with probability < 1% are dropped.

Outputs:
  <argv[1] or availability.csv>  long form, one row per player/pick
  availability_by_player.csv     one row per player, threshold picks only
"""
import csv, math, sys
from statistics import NormalDist

MIN_P = 0.01
MIN_PICKS = 20                    # players sampled fewer times than this are ignored
MAX_PICK = 300
THRESHOLDS = (0.90, 0.70, 0.50)   # reported as the latest pick still this available
OUT = sys.argv[1] if len(sys.argv) > 1 else "availability.csv"
SUMMARY_OUT = sys.argv[2] if len(sys.argv) > 2 else "availability_by_player.csv"

# ADP.tsv has TWO columns headed "Team" (the team abbreviation, and a trailing
# blank one). csv.DictReader keeps the last, silently blanking the field, so
# read positionally instead.
RANK, PID, PLAYER, TEAM, POS, ADP_C, MINP, MAXP, _DIFF, NPICKS = range(10)


def norm_cdf(z):
    return 0.5 * math.erfc(-z / math.sqrt(2.0))


def latest_pick_at_least(mu, sigma, t):
    """Highest pick k with P(available at k) >= t.

    P(avail at k) = 1 - Phi((k - 0.5 - mu)/sigma) >= t
      <=>  k <= mu + 0.5 + sigma * z,  where z = Phi^-1(1 - t)
    so the answer is the floor of that bound. Returns 0 when even pick 1
    falls short, i.e. the player is already less than t likely to last.
    """
    z = NormalDist().inv_cdf(1.0 - t)
    return max(0, math.floor(mu + 0.5 + sigma * z))


rows, skipped = [], []
with open("ADP.tsv", newline="", encoding="utf-8-sig") as f:
    rdr = csv.reader(f, delimiter="\t")
    next(rdr)
    for r in rdr:
        try:
            mu = float(r[ADP_C])
            lo, hi = float(r[MINP]), float(r[MAXP])
            n = float(r[NPICKS])
        except (TypeError, ValueError, IndexError):
            skipped.append(r[PLAYER] if len(r) > PLAYER else "?"); continue
        sigma = (hi - lo) / (0.61 * math.log(n) + 2.235)
        if n < MIN_PICKS or not (sigma > 0):
            skipped.append(r[PLAYER]); continue
        rows.append((r[RANK], r[PID], r[PLAYER], r[TEAM], r[POS], mu, sigma))

out, summary = [], []
for rank, pid, player, team, pos, mu, sigma in rows:
    marks = [latest_pick_at_least(mu, sigma, t) for t in THRESHOLDS]
    summary.append((rank, pid, player, team, pos,
                    round(mu, 2), round(sigma, 3), *marks))
    for k in range(1, MAX_PICK + 1):
        p = 1.0 - norm_cdf((k - 0.5 - mu) / sigma)
        if p < MIN_P:
            break          # survival is monotone decreasing in k
        out.append((rank, pid, player, team, pos,
                    round(mu, 2), round(sigma, 3), k, round(p, 4)))

with open(OUT, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["Rank", "Player ID", "Player", "Team", "Position(s)",
                "ADP", "SD", "Pick", "P_Available"])
    w.writerows(out)

with open(SUMMARY_OUT, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["Rank", "Player ID", "Player", "Team", "Position(s)",
                "ADP", "SD", "Pick_90pct", "Pick_70pct", "Pick_50pct"])
    w.writerows(summary)

print(f"players modeled: {len(rows)}  skipped: {len(skipped)}")
print(f"long rows: {len(out)} -> {OUT}")
print(f"per-player rows: {len(summary)} -> {SUMMARY_OUT}")
