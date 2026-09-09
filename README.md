# draftsim

Fantasy football draft simulation, and an interactive tool to draft one seat
against it.

## Install and run

```bash
pip install -e .
python -m draftsim --slot 5
```

`pip install -e .` is optional — `python -m draftsim` works from this
directory as it stands. Requires Python 3.11+, numpy 1.25+ (for
`Generator.spawn`), scipy and pyarrow.

**R is optional.** Projections come from the ffanalytics R package, but a
prebuilt copy is published in `data/` and fetched automatically, so the tool
runs anywhere with a network connection — a borrowed laptop, a Colab notebook
— with no R at all. Install R and
[ffanalytics](https://github.com/FantasyFootballAnalytics/ffanalytics) only if
you want to build fresher projections yourself with `--refresh-projections`:

```r
install.packages("remotes")
remotes::install_github("FantasyFootballAnalytics/ffanalytics")
```

`Rscript` must then be on the path, or `DRAFTSIM_RSCRIPT` set to it.

## Layout

```
pyproject.toml          packaging, dependencies, and the ruff config
draftsim/
    __init__.py         DATA, and the paths to the shipped projection files
    __main__.py         `python -m draftsim` -> the draft tool
    adp.py              scrapes the ADP board from nfc.shgn.com
    projections.py      pulls projections out of the ffanalytics R package
    league.py           the league: its roster rules and its scoring
    scoring.py          what each stat pays, and the flags that set it
    draft_sim.py        drafts off simulated ADP
    vor_draft_sim.py    drafts off sampled projections, by VOR
    combined_draft.py   blends the two boards and drafts the result
    combine_ranks.py    the same blend over the two aggregate summaries
    pick_sim.py         finishes a draft in progress, thousands of times
    draft_tool.py       the interactive tool
```

Every module runs on its own, sharing one league and one player pool:

```bash
python -m draftsim                      # the interactive draft tool
python -m draftsim.combined_draft       # 10,000 drafts off the blended board
python -m draftsim.draft_sim            # 10,000 drafts off simulated ADP
python -m draftsim.vor_draft_sim        # 10,000 drafts off sampled VOR
python -m draftsim.combine_ranks        # blend the two summaries into a board
python -m draftsim.adp                  # just pull the ADP board and print it
python -m draftsim.projections          # just build the projections and stop
python -m draftsim.scoring              # print the scoring rules in force
```

No data ships with the package. Output (the parquet pick logs and the summary
CSVs) is written to the current working directory, so run a simulator from
wherever you want its results.

## Where the numbers come from

Both inputs move, so neither is carried as a file. Both are fetched on demand
and cached under `~/.cache/draftsim` (or `$DRAFTSIM_CACHE`).

**ADP** is scraped from the NFC ADP page, over a window you choose:

```bash
python -m draftsim --adp-days 30                          # a wider window
python -m draftsim --adp-from 2026-08-01 --adp-to 2026-08-15   # a fixed one
python -m draftsim --adp-teams 12                         # 12-team drafts only
python -m draftsim --refresh-adp                          # ignore the cache
```

The default is the last 14 days, all draft sizes, all non-auction formats —
the same query the page loads with. Boards are cached as plain TSVs keyed by
the window and filters.

**Projections** are looked for in three places, in order: the local cache,
the copy published in this repo's `data/`, and only then a fresh scrape out of
R. The middle step is what makes R optional.

```bash
python -m draftsim --projections-season 2026    # default: the current season
python -m draftsim --projections-week 3         # 0, the preseason, by default
python -m draftsim --refresh-projections        # re-scrape, ignoring the cache
```

`--refresh-projections` skips both caches and insists on the scrape, which is
the only way to get numbers newer than what is published.

`projections.py` holds an R program that calls `scrape_data()` and
`projections_table()` — once for the points board and once with
`return_raw_stats = TRUE` for the component stats and their per-source
standard deviations — and hands both back over stdout. Nothing is written to
a CSV on the way through. A scrape hits six sites per position and takes about
two minutes, so the result is cached as JSON and reused; a failed scrape falls
back on the newest cache.

One column ffanalytics does not have is the ADP board's player id. That join
is made on normalised name and position, and only where the match is unique;
a player with no ADP simply has no market, which is a case the simulators
already handle.

Both caches mean a draft's twelve worker processes share one fetch between
them, and a cached board stands in if a source cannot be reached.

## Scoring

Scoring is a league setting, alongside the roster — what a receiver is worth
depends on whether the league pays a point a catch as surely as on how many
receivers a team starts. It defaults to Yahoo's rules, 0.5 PPR, and is set
one category at a time or as a whole:

```bash
python -m draftsim --score rec=1                 # full PPR
python -m draftsim --score rec=1,pass_tds=6      # ...and 6-point passing
python -m draftsim --scoring my_league.json      # a whole rule set
python -m draftsim.scoring                       # the rules in force
python -m draftsim.scoring --json > my_league.json    # a file to edit
```

A scoring file is flat JSON carrying only what differs from the default —
`{"rec": 1, "pass_tds": 6}` — and the category names are ffanalytics', which
`python -m draftsim.scoring` lists.

Changing the scoring changes the projections, so the board is rebuilt and
cached under its own name; the plain filename stays the default rules', which
is the one published here. On a machine with R that means a fresh scrape. On
one without — a Colab notebook — the published *component stats* are scored
locally instead, which is exact for the `average` and `weighted`
reconciliations and approximate for `robust`: see `projections.rescore()`,
which says what the difference is and why.

Two rules are worth knowing. The scoring the projections were built with
travels with them, and a board scored by other rules than the league's is
refused rather than drafted. And `pts_bracket`, which pays a defense on the
points it allows, is carried through to R but scores nothing today: no source
ffanalytics scrapes projects points allowed.

## The draft tool

You sit in one seat. The other eleven teams draft themselves, and every time
the pick comes back around the tool stops and prices each of your options by
simulating the rest of the draft 500 times per option — reporting not a rank
but the projected starting lineup you would carry into the season.

```bash
python -m draftsim --slot 5                    # mock draft, rivals auto-pick
python -m draftsim --slot 5 --no-mock-draft    # follow a real draft, enter
                                               # every pick yourself
```

Every default written out, for reference:

```bash
python -m draftsim --slot 1 --sims 500 --seed 960122 --my-weight 0.6 --options 10 --workers 0 --mock-draft --adp-days 14 --adp-teams 0 --adp-draft-type 0 --projections-week 0 --vor-weight-lo 0.3333333333333333 --vor-weight-hi 0.6666666666666666 --teams 12 --roster 15 --flex 1 --qb 1 --rb 2 --wr 3 --te 1
```

`python -m draftsim --help` lists them all; each module's docstring explains
what it does and why.

## Development

Lint, format and the complexity ceiling are configured in `pyproject.toml`;
the package is clean against all three.

```bash
ruff check draftsim/
ruff format draftsim/
```

Cyclomatic complexity is capped at 8 (`C901`). Percent formatting (`UP031`) is
the house style and is the one rule turned off — the format strings are column
layouts for the printed tables, and f-strings read no better there.
