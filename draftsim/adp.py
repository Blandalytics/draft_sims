"""The ADP board, pulled from nfc.shgn.com rather than carried as a file.

A shipped copy of the market goes stale the day it is written, and ADP is the
one input to these simulators that moves daily. So the board is scraped on
demand from the NFC ADP page,

    https://nfc.shgn.com/adp/football

over a window that defaults to the last fourteen days and is yours to set --
`--adp-days`, or `--adp-from`/`--adp-to` for a fixed range.

That page renders its table from an XHR its form posts, so this asks the same
endpoint the same question:

    POST https://nfc.shgn.com/adp.data.php
    team_id=0&from_date=...&to_date=...&num_teams=0&draft_type=0
    &sport=football&position=&league_teams=0

and gets back a fragment of <tr> rows -- rank, player, team, position, ADP,
min, max, diff, picks -- with the player's id in the link on his name. Those
are exactly the columns the ADP.tsv this replaces carried, in the same order
and the same id space as the `pid` the projections join on, so the fragment is
written out as that same TSV and draft_sim reads it as it always did.

The result is cached under DRAFTSIM_CACHE (or ~/.cache/draftsim) keyed by the
window and the filters, for three reasons. A draft asks for the board once per
process and the simulations run twelve of those, which should not be twelve
requests. The file is a plain TSV you can read, diff or keep. And when the
network or the site is not there, a cached board is a far better answer than
no board -- a stale market beats a tool that will not open on draft night.
"""

import argparse
import csv
import os
import re
import sys
import urllib.parse
import urllib.request
from datetime import date, timedelta
from html.parser import HTMLParser
from pathlib import Path

PAGE = "https://nfc.shgn.com/adp/football"
ENDPOINT = "https://nfc.shgn.com/adp.data.php"
DEFAULT_DAYS = 14
TIMEOUT = 30
USER_AGENT = "draftsim (+https://nfc.shgn.com/adp/football)"

# the header ADP.tsv carried, kept so draft_sim.load_players is unchanged
COLUMNS = [
    "Rank",
    "Player ID",
    "Player",
    "Team",
    "Position(s)",
    "ADP",
    "Min Pick",
    "Max Pick",
    "Difference",
    "# Picks",
    "Team",
    "Team Pick",
]
N_CELLS = 11  # cells per row; the player id is in the name's link
PLAYER_HREF = re.compile(r"/player/football/(\d+)/")


def cache_dir():
    """Where scraped boards are kept between runs."""
    env = os.environ.get("DRAFTSIM_CACHE")
    return Path(env) if env else Path.home() / ".cache" / "draftsim"


def default_range(days=DEFAULT_DAYS, today=None):
    """The window to ask for: the last `days` days, ending today."""
    end = today or date.today()
    return end - timedelta(days=days), end


class _Rows(HTMLParser):
    """The <tr> rows of the fragment adp.data.php returns.

    Cells are collected as text, with one exception: the player's numeric id
    is an attribute of the page rather than a column, sitting in the href on
    his name, so it is picked out of the link and inserted as the second
    field. A row that yields no id is not a player row and is dropped, which
    is what keeps a header or a spacer out of the output.
    """

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.rows = []
        self._cells = None
        self._text = None
        self._pid = ""

    def handle_starttag(self, tag, attrs):
        if tag == "tr":
            self._cells, self._pid = [], ""
        elif tag == "td" and self._cells is not None:
            self._text = []
        elif tag == "a":
            href = dict(attrs).get("href") or ""
            m = PLAYER_HREF.search(href)
            if m and not self._pid:
                self._pid = m.group(1)

    def handle_data(self, data):
        if self._text is not None:
            self._text.append(data)

    def handle_endtag(self, tag):
        if tag == "td" and self._text is not None:
            self._cells.append("".join(self._text).strip())
            self._text = None
        elif tag == "tr" and self._cells is not None:
            if self._pid and len(self._cells) >= N_CELLS:
                c = self._cells
                self.rows.append([c[0], self._pid] + c[1:N_CELLS])
            self._cells, self._text = None, None


def parse(html):
    """Rows in ADP.tsv's column order, best ADP first."""
    p = _Rows()
    p.feed(html)
    return p.rows


def fetch(from_date, to_date, num_teams=0, draft_type=0, timeout=TIMEOUT):
    """Ask the page's own endpoint for one window of the board."""
    body = urllib.parse.urlencode(
        {
            "team_id": 0,
            "from_date": from_date.isoformat(),
            "to_date": to_date.isoformat(),
            "num_teams": num_teams,
            "draft_type": draft_type,
            "sport": "football",
            "position": "",
            "league_teams": 0,
        }
    ).encode()
    req = urllib.request.Request(
        ENDPOINT,
        data=body,
        headers={
            "User-Agent": USER_AGENT,
            "X-Requested-With": "XMLHttpRequest",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8", "replace")


def cache_path(from_date, to_date, num_teams=0, draft_type=0):
    return cache_dir() / (
        "adp_football_%s_%s_t%d_d%d.tsv"
        % (from_date.isoformat(), to_date.isoformat(), num_teams, draft_type)
    )


def write(rows, path):
    """Write the board out as a TSV, atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(COLUMNS)
        w.writerows(rows)
    os.replace(tmp, path)


def newest_cached():
    """Any board we have kept, most recent first, or None."""
    found = sorted(
        cache_dir().glob("adp_football_*.tsv"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return found[0] if found else None


def table(
    from_date=None,
    to_date=None,
    num_teams=0,
    draft_type=0,
    refresh=False,
    quiet=False,
):
    """A TSV of the board for this window, scraping it if we do not have it.

    Returns the path. A window already cached is not re-fetched unless
    `refresh` says so, which is what keeps a draft's twelve worker processes
    to one request between them.
    """
    if from_date is None or to_date is None:
        lo, hi = default_range()
        from_date, to_date = from_date or lo, to_date or hi
    path = cache_path(from_date, to_date, num_teams, draft_type)
    if path.exists() and not refresh:
        return path
    try:
        rows = parse(fetch(from_date, to_date, num_teams, draft_type))
    except Exception as exc:  # network, DNS, HTTP, parse
        stale = newest_cached()
        if stale is None:
            raise SystemExit(
                "could not reach %s (%s), and no cached board to fall back "
                "on. Check the connection, or point --adp-from/--adp-to at a "
                "window you have already pulled." % (ENDPOINT, exc)
            ) from exc
        print(
            "could not reach %s (%s)\n  falling back on %s"
            % (ENDPOINT, exc, stale),
            file=sys.stderr,
        )
        return stale
    if not rows:
        raise SystemExit(
            "%s returned no players for %s..%s"
            % (ENDPOINT, from_date, to_date)
        )
    write(rows, path)
    if not quiet:
        print(
            "adp: %d players from %s, %s..%s"
            % (len(rows), PAGE, from_date, to_date)
        )
    return path


def add_adp_args(ap):
    """The flags that choose which market the board is drawn from."""
    g = ap.add_argument_group(
        "adp source",
        "scraped from %s and cached under %s" % (PAGE, cache_dir()),
    )
    g.add_argument(
        "--adp-days",
        type=int,
        default=DEFAULT_DAYS,
        help="window ending today, in days (default %d)" % DEFAULT_DAYS,
    )
    g.add_argument(
        "--adp-from",
        type=date.fromisoformat,
        metavar="YYYY-MM-DD",
        help="start of a fixed window, overriding --adp-days",
    )
    g.add_argument(
        "--adp-to",
        type=date.fromisoformat,
        metavar="YYYY-MM-DD",
        help="end of that window (default today)",
    )
    g.add_argument(
        "--adp-teams",
        type=int,
        default=0,
        choices=(0, 10, 12, 14),
        help="only drafts of this size; 0 is all of them",
    )
    g.add_argument(
        "--adp-draft-type",
        type=int,
        default=0,
        help="the page's draft_type filter; 0 is all non-auction",
    )
    g.add_argument(
        "--refresh-adp",
        action="store_true",
        help="re-scrape even if this window is already cached",
    )
    return ap


def path_from_args(a, quiet=False):
    """The board those flags describe, scraped or cached."""
    lo, hi = default_range(a.adp_days)
    return table(
        a.adp_from or lo,
        a.adp_to or hi,
        a.adp_teams,
        a.adp_draft_type,
        a.refresh_adp,
        quiet,
    )


def main():
    ap = add_adp_args(
        argparse.ArgumentParser(
            description=__doc__,
            formatter_class=argparse.RawDescriptionHelpFormatter,
        )
    )
    ap.add_argument("--out", help="also copy the board here")
    a = ap.parse_args()
    path = path_from_args(a)
    print(path)
    if a.out:
        Path(a.out).write_bytes(path.read_bytes())
        print("copied to %s" % a.out)


if __name__ == "__main__":
    main()
