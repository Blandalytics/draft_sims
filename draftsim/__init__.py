"""Fantasy football draft simulation, and a tool to draft against it.

Three simulators and an interactive front end, sharing one league definition
and one player pool:

    league          the league: its shape, its roster rules and its scoring,
                    which everything else is priced against
    scoring         what each stat pays -- Yahoo's default, 0.5 PPR, until
                    the league says otherwise
    adp             the market, scraped from nfc.shgn.com over a window that
                    defaults to the last fourteen days
    projections     projected points and the component stats behind them,
                    pulled out of the ffanalytics R package and scored by
                    the league's rules
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

No data ships with the package. Both inputs move -- the market daily, the
projections whenever a source updates -- so both are fetched on demand and
cached under DRAFTSIM_CACHE, or ~/.cache/draftsim, the projections under a
name that carries the scoring they were built with. Simulators write their
output to the current working directory.
"""

__version__ = "2.0.0"
