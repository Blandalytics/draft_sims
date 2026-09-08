"""Per-player skew-normal parameters for draft-pick distributions.

Skew normal has three parameters -- location xi, scale omega, shape alpha --
so unlike the two-parameter Weibull it can match the centre, the spread and
the skew at once. Each feature drives one parameter:

SHAPE alpha  <- ADP, Min Pick, Max Pick.
    p_obs = (ADP - Min) / (Max - Min) says where the mean sits inside the
    observed range. Min and Max are the extremes of n draws, so they are read
    as the 1/(n+1) and n/(n+1) quantiles. For a standard skew normal the same
    ratio depends only on alpha and falls monotonically in it, so alpha is
    solved by bisection against p_obs. alpha = 0 is the symmetric case,
    alpha < 0 a left tail, alpha > 0 a right tail.

SCALE omega  <- the StDev estimate, sd = (Max-Min)/(0.61*ln(n)+2.235).
    var = omega^2 * (1 - 2*delta^2/pi)  with delta = alpha/sqrt(1+alpha^2), so
        omega = sd / sqrt(1 - 2*delta^2/pi)

LOCATION xi  <- ADP.
    mean = xi + omega*delta*sqrt(2/pi), so
        xi = ADP - omega*delta*sqrt(2/pi)

Unlike the Weibull fit, the resulting distribution has mean exactly ADP and
sd exactly the SD estimate -- no location shift needed afterwards.
"""
import csv, math
from scipy.stats import skewnorm

MIN_PICKS = 20
A_LO, A_HI = -60.0, 60.0
RENAME = {"TDSP": "DST", "TK": "K"}
RANK, PID, PLAYER, TEAM, POS, ADP_C, MINP, MAXP, _DIFF, NPICKS = range(10)


def mean_position(a, u_lo, u_hi):
    """Where a standard skew normal's mean falls between two of its quantiles."""
    lo, hi = skewnorm.ppf(u_lo, a), skewnorm.ppf(u_hi, a)
    return (skewnorm.mean(a) - lo) / (hi - lo)


def solve_shape(p_obs, u_lo, u_hi, tol=1e-9):
    """Bisect for alpha. mean_position DECREASES in alpha, hence the flip."""
    lo, hi = A_LO, A_HI
    if p_obs >= mean_position(lo, u_lo, u_hi):
        return lo, True                    # more left skew than alpha=-60
    if p_obs <= mean_position(hi, u_lo, u_hi):
        return hi, True                    # more right skew than alpha=+60
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if mean_position(mid, u_lo, u_hi) > p_obs:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi), False


rows, clamped = [], 0
with open("ADP.tsv", newline="", encoding="utf-8-sig") as f:
    rdr = csv.reader(f, delimiter="\t")
    next(rdr)
    for r in rdr:
        try:
            mu = float(r[ADP_C])
            lo_p, hi_p = float(r[MINP]), float(r[MAXP])
            n = float(r[NPICKS])
        except (TypeError, ValueError, IndexError):
            continue
        if n < MIN_PICKS or hi_p <= lo_p:
            continue

        sd = (hi_p - lo_p) / (0.61 * math.log(n) + 2.235)
        p_obs = (mu - lo_p) / (hi_p - lo_p)
        u_lo, u_hi = 1.0 / (n + 1.0), n / (n + 1.0)

        alpha, was_clamped = solve_shape(p_obs, u_lo, u_hi)
        clamped += was_clamped
        delta = alpha / math.sqrt(1.0 + alpha * alpha)
        omega = sd / math.sqrt(1.0 - 2.0 * delta * delta / math.pi)
        xi = mu - omega * delta * math.sqrt(2.0 / math.pi)
        skew = skewnorm.stats(alpha, moments="s")

        rows.append([r[PID], r[PLAYER], r[TEAM], RENAME.get(r[POS], r[POS]),
                     mu, lo_p, hi_p, int(n), round(sd, 4), round(p_obs, 6),
                     f"{alpha:.10g}", f"{omega:.10g}", f"{xi:.10g}",
                     round(float(skew), 4), int(was_clamped)])

with open("skewnorm_params.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["Player ID", "Player", "Team", "Pos", "ADP", "Min_Pick",
                "Max_Pick", "N_Picks", "SD", "P_Obs", "Alpha_shape",
                "Omega_scale", "Xi_location", "Skewness", "Clamped"])
    w.writerows(rows)

al = sorted(float(r[10]) for r in rows)
print(f"{len(rows)} players fit -> skewnorm_params.csv")
print(f"alpha: min {al[0]:.3f}  p25 {al[len(al)//4]:.3f}  median "
      f"{al[len(al)//2]:.3f}  p75 {al[3*len(al)//4]:.3f}  max {al[-1]:.3f}")
print(f"left-skewed (alpha<0): {sum(a < 0 for a in al)}  "
      f"right-skewed: {sum(a > 0 for a in al)}")
print(f"clamped at bracket ends: {clamped}")
