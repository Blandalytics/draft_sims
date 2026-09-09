"""A league: its roster rules and its scoring, which are one setting each.

Split out of draft_sim.py so that anything drafting off a board --
combined_draft.py, say -- can reuse the exact same legality checks without
dragging in numpy, pyarrow and scipy. Nothing here imports anything but
functools and scoring.py, which imports nothing at all.

One league configuration serves all three simulators. DEFAULT is it -- 12
teams and 15 spots: 1 QB, 2 RB, 3 WR, 1 TE, 1 FLEX (RB/WR/TE), 1 DST, 1 K,
5 bench, scored by Yahoo's default rules -- and add_league_args /
league_from_args give every script the same flags to override it. draft_sim.py
and vor_draft_sim.py must agree on the league, since combined_draft.py blends
their two boards and drafts the result: a VOR rank priced against one league's
replacement levels is meaningless in another league's draft.

Scoring belongs here for the same reason the roster does. Both are the
league's, both are set with the same flags, and a board is only worth
anything in the league it was priced for -- what a receiver is worth depends
on whether the league pays a point a catch as surely as it depends on how
many receivers a team starts. What each category pays lives in scoring.py;
League carries the rules, and projections.py builds the board from them.

Each simulator keeps its own pick heuristic (draft_sim takes the lowest
simulated ADP slot, vor_draft_sim the highest VOR); only the league -- team
count, roster size, starting lineup, flex -- is shared.

The rules and draft-order policy that follow from a league:
    bench is QB/RB/WR/TE only -- never a second DST or K
    QB and TE cap at starters + 1; exactly 1 DST and 1 K, and the K is
      always a team's final pick
    at most 1 bench player before the starting RB and WR slots are full
    no bench QB or TE at all until those RB/WR starters are full
    QB, TE and DST starters may be deferred as long as the roster can still
      be completed, so a team may fill bench spots ahead of them
"""

from functools import lru_cache

from .scoring import Scoring, add_scoring_args
from .scoring import from_args as scoring_from_args

DEF = "DST"  # defense/special teams
K = "K"  # kicker
# K and DST are streamed, so vor_draft_sim pins their replacement baselines
# rather than deriving them from a roster requirement, which would inflate them
BASELINE_OVERRIDE = {K: 3, DEF: 3}
FLEX_POS = ("RB", "WR", "TE")
DRAFTABLE = ("QB", "RB", "WR", "TE", DEF, K)
MAX_BENCH_BEFORE_RBWR = 1  # bench picks allowed before RB/WR starters fill


class League:
    """One league's shape, and the roster legality that follows from it.

    Everything past the starting lineup is derived: the flex requirement, the
    QB/TE caps and the bench size all fall out of `starters`, `n_flex` and
    `roster`, so a league is defined once and cannot disagree with itself.

    `scoring` is the other half of what a league is -- what each stat pays.
    Nothing here uses it, since roster legality does not care what a catch
    is worth; it is carried because it is the league's, and because
    everything that prices a player against this league takes the league.
    """

    def __init__(
        self, name, n_teams, roster, starters, n_flex=1, scoring=None
    ):
        self.name = name
        self.n_teams = n_teams
        self.roster = roster
        self.starters = dict(starters)
        self.n_flex = n_flex
        self.scoring = scoring or Scoring()
        # RB/WR/TE bodies a legal roster owes: each position's own starters
        # plus the flex slots they share
        self.flex_total = sum(starters[p] for p in FLEX_POS) + n_flex
        # QB/TE cap at one more than they start, so a team can carry a single
        # backup at each but not stockpile them. DST is start-one, hold-one.
        self.caps = {p: starters[p] + 1 for p in ("QB", "TE")} | {DEF: 1, K: 1}
        # every mandatory starting slot, including the one DST and one K that
        # `starters` leaves implicit -- what replacement levels are keyed on
        self.all_starters = dict(starters) | {DEF: 1, K: 1}
        self.bench = roster - sum(starters.values()) - n_flex - 2
        if self.bench < 0:
            raise ValueError(
                "%s: %d spots cannot hold the starting lineup" % (name, roster)
            )
        # see legal() -- the cache is per league, so two leagues never share
        # an answer
        self._cached = lru_cache(maxsize=None)(self._legal)

    def __repr__(self):
        return "<League %s: %d teams x %d>" % (
            self.name,
            self.n_teams,
            self.roster,
        )

    def __getstate__(self):
        """A league without its cache, so it can cross a process boundary.

        `_cached` wraps a bound method and cannot be pickled. Dropping it
        loses nothing: it is a pure function of the league, so a League that
        arrives in a worker process rebuilds an empty one and refills it as it
        goes. pick_sim.py hands a league to its workers this way.
        """
        return {k: v for k, v in self.__dict__.items() if k != "_cached"}

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._cached = lru_cache(maxsize=None)(self._legal)

    def legal(self, counts, pos, taken):
        """Can a team holding `counts` add `pos` and still fill a legal roster?

        A thin caching front for _legal. The answer depends only on how many
        of each position the roster already holds, and those are capped hard,
        so a whole simulation only ever visits a few thousand distinct states
        -- far fewer than the millions of calls made against them. Keying on
        the count tuple turns almost every call into a dict lookup.
        """
        return self._cached(
            (
                counts["QB"],
                counts["RB"],
                counts["WR"],
                counts["TE"],
                counts[DEF],
                counts[K],
            ),
            pos,
            taken,
        )

    def _legal(self, state, pos, taken):
        counts = dict(zip(DRAFTABLE, state, strict=True))
        if counts[pos] + 1 > self.caps.get(pos, self.roster):
            return False
        # Kickers go last: a K is only ever a team's final pick, and conversely
        # the final pick must be a K because the reserve below leaves no room.
        if pos == K and taken + 1 < self.roster:
            return False

        c = dict(counts)
        c[pos] += 1
        remaining = self.roster - (taken + 1)

        # shortfalls against the starting requirements
        need = {p: max(0, self.starters[p] - c[p]) for p in self.starters}
        need_def = max(0, 1 - c[DEF])
        need_k = max(0, 1 - c[K])
        flex_have = sum(c[p] for p in FLEX_POS)
        # flex-eligible bodies still owed: enough to cover each position's own
        # shortfall, and enough to reach the total the RB/WR/TE/FLEX slots want
        flex_need = max(
            sum(need[p] for p in FLEX_POS), self.flex_total - flex_have
        )
        if need["QB"] + need_def + need_k + flex_need > remaining:
            return False

        # Draft-order policy. Only binds while the starting RB/WR slots are
        # short; a pick that completes them is judged on the resulting roster,
        # so it is never blocked by the state it just fixed.
        if not self.rb_wr_started(c):
            nb = self.bench_count(c)
            if nb > MAX_BENCH_BEFORE_RBWR:
                return False
            if nb > self.bench_count(counts) and pos in ("QB", "TE"):
                return False  # this pick would be a bench QB/TE
        return True

    def bench_count(self, c):
        """Players on a roster that fill no starting slot.

        Slots are assigned greedily: each position's own starters first, then
        the flex from any RB/WR/TE surplus, then the single DST and K.
        Whatever is left over is bench. So an extra RB taken while WR is short
        lands in FLEX, not on the bench -- flex is a starting spot.
        """
        starters = sum(min(c[p], self.starters[p]) for p in self.starters)
        surplus = sum(max(0, c[p] - self.starters[p]) for p in FLEX_POS)
        return (
            sum(c.values())
            - starters
            - min(self.n_flex, surplus)
            - min(c[DEF], 1)
            - min(c[K], 1)
        )

    def rb_wr_started(self, c):
        """Are the starting RB and WR slots both filled?"""
        return (
            c["RB"] >= self.starters["RB"] and c["WR"] >= self.starters["WR"]
        )

    def snake_order(self):
        """(round, team) for every pick of the draft, in order."""
        for rd in range(self.roster):
            order = (
                range(self.n_teams)
                if rd % 2 == 0
                else reversed(range(self.n_teams))
            )
            for team in order:
                yield rd + 1, team

    def check(self, counts):
        """Every team must satisfy the roster rules exactly."""
        for t, c in enumerate(counts, start=1):
            total = sum(c.values())
            assert total == self.roster, f"team {t}: {total} players"
            for p, need in self.starters.items():
                assert c[p] >= need, f"team {t}: {c[p]} {p} < {need}"
            for p, cap in self.caps.items():
                assert c[p] <= cap, f"team {t}: {c[p]} {p} > cap {cap}"
            flex = sum(c[p] for p in FLEX_POS)
            assert flex >= self.flex_total, (
                f"team {t}: {flex} RB/WR/TE < {self.flex_total}"
            )
            assert c[DEF] == 1, f"team {t}: {c[DEF]} DST, must be exactly 1"
            assert c[K] == 1, f"team {t}: {c[K]} K, must be exactly 1"


DEFAULTS = {
    "teams": 12,
    "roster": 15,
    "flex": 1,
    "QB": 1,
    "RB": 2,
    "WR": 3,
    "TE": 1,
}
DEFAULT = League(
    "default",
    n_teams=DEFAULTS["teams"],
    roster=DEFAULTS["roster"],
    n_flex=DEFAULTS["flex"],
    starters={p: DEFAULTS[p] for p in ("QB", "RB", "WR", "TE")},
)


def add_league_args(ap):
    """Add the league flags every simulator shares.

    All three scripts take the same options and the same defaults, so
    draft_sim.py and vor_draft_sim.py cannot silently drift apart, and
    combined_draft.py can hand both of them one league it was given.
    """
    g = ap.add_argument_group(
        "league settings",
        "shared by draft_sim, vor_draft_sim and combined_draft",
    )
    add_scoring_args(ap)  # the league's scoring, in its own group
    g.add_argument("--teams", type=int, default=DEFAULTS["teams"])
    g.add_argument(
        "--roster",
        type=int,
        default=DEFAULTS["roster"],
        help="total spots per team, starters and bench",
    )
    g.add_argument(
        "--flex",
        type=int,
        default=DEFAULTS["flex"],
        help="FLEX (RB/WR/TE) slots per team",
    )
    for p in ("QB", "RB", "WR", "TE"):
        g.add_argument(
            "--" + p.lower(),
            type=int,
            default=DEFAULTS[p],
            help="starting %s slots" % p,
        )
    return ap


def league_from_args(a):
    """Build the League those flags describe, scoring and all."""
    starters = {p: getattr(a, p.lower()) for p in ("QB", "RB", "WR", "TE")}
    scoring = scoring_from_args(a)
    same = (
        a.teams == DEFAULTS["teams"]
        and a.roster == DEFAULTS["roster"]
        and a.flex == DEFAULTS["flex"]
        and all(starters[p] == DEFAULTS[p] for p in starters)
        and scoring.is_default()
    )
    return League(
        "default" if same else "custom",
        n_teams=a.teams,
        roster=a.roster,
        starters=starters,
        n_flex=a.flex,
        scoring=scoring,
    )
