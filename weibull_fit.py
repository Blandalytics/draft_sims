"""Per-player Weibull parameters for draft-pick distributions.

Two-parameter Weibull, shape k and scale lam, fit exactly as specified:

SHAPE  <- ADP, Min Pick, Max Pick.
    Where the mean sits inside the observed range carries the skew:
        p_obs = (ADP - Min) / (Max - Min)
    p_obs near 0.5 means a symmetric board; below means a right tail (the
    player occasionally slides), above means a left tail. Min and Max are the
    extremes of n observations, so they are read as the 1/(n+1) and n/(n+1)
    quantiles. For a standard Weibull (lam = 1) the same ratio is
        p(k) = (Gamma(1+1/k) - q(u_lo,k)) / (q(u_hi,k) - q(u_lo,k))
    which rises monotonically in k, so k is solved by bisection against
    p_obs. k ~ 3.6 is the near-symmetric case.

SCALE  <- the StDev estimate.
    sd = (Max - Min) / (0.61*ln(n) + 2.235), and a Weibull's sd is
    lam * sqrt(Gamma(1+2/k) - Gamma(1+1/k)^2), so
        lam = sd / sqrt(Gamma(1+2/k) - Gamma(1+1/k)^2)

Because scale is pinned by the spread rather than the centre, the fitted
mean lam*Gamma(1+1/k) will not equal ADP. `loc` is reported as the shift
that re-centres it (a 3-parameter Weibull) for anyone who needs the mean to
land on ADP; k and lam themselves are untouched by it.
"""
import csv, math

MIN_PICKS = 20
K_LO, K_HI = 0.20, 50.0          # bisection bracket
RENAME = {"TDSP": "DST", "TK": "K"}
RANK, PID, PLAYER, TEAM, POS, ADP_C, MINP, MAXP, _DIFF, NPICKS = range(10)

G = math.gamma


def wb_quantile(u, k):
    return (-math.log(1.0 - u)) ** (1.0 / k)


def mean_position(k, u_lo, u_hi):
    """Where a standard Weibull's mean falls between its u_lo and u_hi quantiles."""
    lo, hi = wb_quantile(u_lo, k), wb_quantile(u_hi, k)
    return (G(1.0 + 1.0 / k) - lo) / (hi - lo)


def solve_shape(p_obs, u_lo, u_hi, tol=1e-10):
    """Bisect for the k whose mean sits at p_obs within the range."""
    lo, hi = K_LO, K_HI
    if p_obs <= mean_position(lo, u_lo, u_hi):
        return lo, True                      # clamped: more skew than k=0.2
    if p_obs >= mean_position(hi, u_lo, u_hi):
        return hi, True                      # clamped: tighter than k=50
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if mean_position(mid, u_lo, u_hi) < p_obs:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi), False


def cv_factor(k):
    """sd of a standard Weibull with this shape."""
    return math.sqrt(G(1.0 + 2.0 / k) - G(1.0 + 1.0 / k) ** 2)


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

        k, was_clamped = solve_shape(p_obs, u_lo, u_hi)
        clamped += was_clamped
        lam = sd / cv_factor(k)
        fitted_mean = lam * G(1.0 + 1.0 / k)

        rows.append([r[PID], r[PLAYER], r[TEAM], RENAME.get(r[POS], r[POS]),
                     mu, lo_p, hi_p, int(n), round(sd, 4), round(p_obs, 6),
                     f"{k:.10g}", f"{lam:.10g}", f"{fitted_mean:.10g}",
                     round(mu - fitted_mean, 3), int(was_clamped)])

with open("weibull_params.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["Player ID", "Player", "Team", "Pos", "ADP", "Min_Pick",
                "Max_Pick", "N_Picks", "SD", "P_Obs", "Shape_k", "Scale_lam",
                "Fitted_Mean", "Loc_Shift", "Clamped"])
    w.writerows(rows)

ks = sorted(float(r[10]) for r in rows)
print(f"{len(rows)} players fit -> weibull_params.csv")
print(f"shape k: min {ks[0]:.3f}  p25 {ks[len(ks)//4]:.3f}  median "
      f"{ks[len(ks)//2]:.3f}  p75 {ks[3*len(ks)//4]:.3f}  max {ks[-1]:.3f}")
print(f"clamped at bracket ends: {clamped}")
