"""Ashton Jeanty's fitted skew-normal draft-pick distribution."""
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import skewnorm, norm

NAME = "Jeanty, Ashton"
SURFACE, INK, INK2, MUTED, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#c3c2b7"
S1, S2 = "#2a78d6", "#eb6834"          # reference palette slots 1, 2

row = next(r for r in csv.DictReader(open("skewnorm_params.csv", encoding="utf-8"))
           if r["Player"] == NAME)
a, w, xi = (float(row[k]) for k in ("Alpha_shape", "Omega_scale", "Xi_location"))
adp, sd = float(row["ADP"]), float(row["SD"])
lo_p, hi_p, n = float(row["Min_Pick"]), float(row["Max_Pick"]), int(row["N_Picks"])

XMAX = 68
x = np.linspace(0, XMAX, 1400)
y_sn = skewnorm.pdf(x, a, loc=xi, scale=w)
y_nm = norm.pdf(x, adp, sd)
mode = x[int(np.argmax(y_sn))]
med = skewnorm.ppf(0.5, a, loc=xi, scale=w)
tail = skewnorm.sf(hi_p, a, loc=xi, scale=w)

fig, ax = plt.subplots(figsize=(9.6, 5.4), dpi=200)
fig.patch.set_facecolor(SURFACE); ax.set_facecolor(SURFACE)

ax.fill_between(x, y_sn, color=S1, alpha=0.13, linewidth=0)
ax.plot(x, y_sn, color=S1, lw=2, zorder=5, solid_capstyle="round")
ax.plot(x, y_nm, color=S2, lw=2, ls=(0, (5, 3)), zorder=4, solid_capstyle="round")

ymax = y_sn.max()
# observed range: the span the Min/Max features actually describe
ax.annotate("", xy=(lo_p, ymax * 1.085), xytext=(hi_p, ymax * 1.085),
            arrowprops=dict(arrowstyle="<->", color=AXIS, lw=1.2))
ax.text((lo_p + hi_p) / 2, ymax * 1.11, f"observed range  {lo_p:.0f}-{hi_p:.0f}",
        color=MUTED, fontsize=9, ha="center", va="bottom")
for xv in (lo_p, hi_p):
    ax.axvline(xv, color=AXIS, lw=1, ls=(0, (2, 3)), zorder=2)
ax.axvline(adp, color=MUTED, lw=1.2, ls=(0, (4, 3)), zorder=3)
ax.text(adp + 0.8, ymax * 0.99, f"ADP {adp:.2f}", color=INK2, fontsize=9,
        va="top", ha="left")

ax.text(mode + 1.5, ymax * 0.58, "Skew normal\n(fitted)", color=S1,
        fontsize=10, fontweight="bold", ha="right", va="center")
ax.text(adp + sd * 2.0, norm.pdf(adp + sd * 2.0, adp, sd) + ymax * 0.10,
        "Normal\n(current sim)", color=S2, fontsize=10, ha="left", va="center")

ax.set_title(f"Ashton Jeanty — simulated draft slot", color=INK,
             fontsize=14, fontweight="bold", loc="left", pad=34)
ax.text(0, 1.018, f"skew normal   α={a:g}   ω={w:.2f}   ξ={xi:.2f}"
        f"      mean {adp:.2f}   sd {sd:.2f}   n={n}",
        transform=ax.transAxes, color=INK2, fontsize=9.5, va="bottom")
ax.set_xlabel("Draft pick", color=INK2, fontsize=10)
ax.set_ylabel("Density", color=INK2, fontsize=10)

ax.grid(axis="y", color=AXIS, lw=0.6, alpha=0.55)
ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
for s in ("left", "bottom"):
    ax.spines[s].set_color(AXIS)
ax.tick_params(colors=MUTED, labelsize=9)
ax.set_xlim(0, XMAX); ax.set_ylim(0, ymax * 1.20)
ax.margins(x=0)

fig.tight_layout()
fig.savefig("jeanty_skewnorm.png", facecolor=SURFACE)
print(f"alpha={a:g} omega={w:.3f} xi={xi:.3f}")
print(f"mean {skewnorm.mean(a, loc=xi, scale=w):.2f} (ADP {adp}) | "
      f"sd {skewnorm.std(a, loc=xi, scale=w):.2f} (SD {sd:.2f})")
print(f"mode ~{mode:.1f} | median {med:.2f} | P(pick > Max {hi_p:.0f}) = {tail:.4f}")
print(f"P(pick < Min {lo_p:.0f}) = {skewnorm.cdf(lo_p, a, loc=xi, scale=w):.4f}")
print("-> jeanty_skewnorm.png")
