"""Fantasy football draft simulation, and a tool to draft against it.

Three simulators and an interactive front end, sharing one league definition
and one player pool:

    league          the league's shape and its roster rules, which everything
                    else is priced against
    draft_sim       drafts off simulated ADP: a skew normal per player, fitted
                    to his published ADP, range and pick count
    vor_draft_sim   drafts off sampled projections: a season drawn per player
                    per draft, with replacement levels recomputed from that
                    same sample
    combined_draft  blends the two boards and drafts the result, each team
                    weighting value against market on its own terms
    combine_ranks   the same blend over the two simulators' aggregate output
    pick_sim        finishes a draft already in progress, thousands of times
    draft_tool      the interactive tool: you draft one seat, the sims price
                    each option by what your team is worth at the end

Run the tool with `python -m draftsim`, and any simulator on its own with
`python -m draftsim.combined_draft` and so on.

Projections ship with the package, under DATA. The ADP board does not: it
moves daily, so adp.py scrapes it from nfc.shgn.com over a window you choose
and caches it. Simulators write their output to the current working
directory, not in here.
"""

from pathlib import Path

__version__ = "1.0.0"

# The ADP table and projections, installed alongside the code so a board loads
# the same way wherever the package is run from.
DATA = Path(__file__).resolve().parent / "data"

PROJECTIONS = DATA / "projections_robust.csv"
PROJECTION_STATS = DATA / "projections_stats.csv"
