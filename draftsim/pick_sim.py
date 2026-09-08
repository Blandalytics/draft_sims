"""Finish a draft in progress, hundreds of times, from one team's seat.

combined_draft.py answers "where does a player go?" over ten thousand drafts
run start to finish. This answers the question a drafter actually asks on the
clock: "if I take *him* here, what do I end up with?" -- so it starts from a
draft already in progress and only ever simulates forward.

One evaluation is a candidate player and a state. The user's team is forced to
take that candidate at the current pick; every pick after it, that team's
included, is made greedily off a freshly drawn board. Each simulation redraws
everything combined_draft.py redraws per draft:

    the ADP rank      one skew normal slot per player (draft_sim.draw_slots)
    the VOR rank      one sampled season per player, replacement levels
                      recomputed from that same sample (vor_draft_sim)
    the weights       w ~ Uniform(1/3, 2/3) per team, on the VOR rank

so a simulation is one coherent guess at how the rest of the board falls, not
a jitter around a fixed ranking. Note the weights are redrawn *per simulation*
here, the user's own included: on the clock you do not know how the room reads
the board, so every simulation asks a differently-minded room. The live draft
outside these simulations works the other way -- see draft_tool.py, where each
rival team is dealt one weight at the first pick and keeps it all draft.

Candidates share their draws. All of them are run against the same N boards --
common random numbers -- so a difference between two options is measured on
identical futures and reflects the choice rather than the sampling. That is
what makes a few points off 500 simulations mean anything.

The number reported is the projected points of the best starting lineup the
final roster can field -- 1 QB, 2 RB, 3 WR, 1 TE, 1 FLEX, 1 DST, 1 K at the
default league, bench excluded.

Those points are the projection as published, one fixed number per player:
the robust `points` ffanalytics reconciles, not a sampled season and not
the components re-scored. Sampling belongs to the board and stops there. It
decides who is *available* when your pick comes back, which is a genuine
draft unknown worth simulating; carrying it into the report would instead
price your roster at what it happened to draw, crediting the drafter with
hindsight he does not have. So the spread across simulations here is draft
uncertainty and nothing else -- the range of teams this pick leads to, with
every one of them valued off the same projection.

The speed this needs comes from one observation: legality depends only on a
position, never on which player fills it, so the only candidate worth pricing
at any pick is the best available player at each of the six positions. Each
team keeps six pointers into its own board, one per position, and they only
walk forward. A pick is then six pointer checks and a cached legality lookup
instead of a scan down a 500-man board.
"""

import os
from collections import namedtuple
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache

import numpy as np

from . import vor_draft_sim
from .league import DRAFTABLE, FLEX_POS

POS_CODE = {p: i for i, p in enumerate(DRAFTABLE)}
N_POS = len(DRAFTABLE)
N_SIMS = 500
BIG = float("inf")
ALL_LEGAL = (True,) * N_POS  # the fallback in Engine.finish
N_CHUNKS = 12  # pieces a decision splits into, pool or no

# Everything a simulation needs from the draft in progress, and nothing that
# cannot be pickled -- the Draft object itself owns an Engine and a league
# full of caches, none of which a worker wants a copy of.
PickState = namedtuple("PickState", "gone counts roster order user_team")

_ENGINE = None  # one per worker process, built once


def _init_worker(lg, matched_only, w_lo, w_hi, adp_path, source):
    """Give this worker its own Engine.

    The board costs a second and a half to load and never changes, so it is
    built once when the process starts rather than shipped with every task.
    That is also why the pool is created once for the whole draft: paying
    this per pick would cost more than the parallelism returns.
    """
    global _ENGINE
    from . import combined_draft  # worker-only; keeps the import out of the

    _ENGINE = Engine(
        combined_draft.Board(lg, matched_only, adp_path, source),
        lg,
        w_lo,
        w_hi,
    )


def _run_chunk(task):
    return _ENGINE.chunk(*task)


class Parallel:
    """A pool of worker processes, each holding its own Engine.

    Created once and reused for every pick of the draft. The workers start on
    the first decision, not here, so the first pick carries the board-loading
    cost; every one after it is warm.
    """

    def __init__(
        self,
        lg,
        matched_only,
        w_lo,
        w_hi,
        workers=0,
        adp_path=None,
        source=None,
    ):
        self.workers = workers or os.cpu_count() or 1
        self._pool = ProcessPoolExecutor(
            max_workers=self.workers,
            initializer=_init_worker,
            initargs=(lg, matched_only, w_lo, w_hi, adp_path, source),
        )

    def map(self, tasks):
        return self._pool.map(_run_chunk, tasks)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self._pool.shutdown(wait=False, cancel_futures=True)


def best_available(bd, hd, sc, gone, ok):
    """The best-scored live player at any legal position, or -1.

    `bd[pc]` is one team's board for position `pc`, best first, and `hd[pc]`
    its pointer into it. Pointers walk past players taken since they last
    moved and never rewind, so the drafted prefix of a position is skipped
    once rather than rescanned every pick. A position the roster cannot
    legally add is left alone entirely -- its pointer catches up the next time
    it is legal.
    """
    best_i, best_s = -1, BIG
    for pc in range(N_POS):
        if not ok[pc]:
            continue
        lst = bd[pc]
        h = hd[pc]
        n = len(lst)
        while h < n and gone[lst[h]]:
            h += 1
        hd[pc] = h
        if h == n:
            continue
        i = lst[h]
        s = sc[i]
        if s < best_s or (s == best_s and i < best_i):
            best_i, best_s = i, s
    return best_i


class Engine:
    """One league, one pooled board, and the machinery to finish drafts on it.

    Everything fixed for the whole draft is built once: each player's position
    code, his projected points, the starting slots his position owes. Only the
    two rank vectors and the twelve weights change per simulation.
    """

    def __init__(self, board, lg, w_lo, w_hi):
        self.board = board
        self.lg = lg
        self.w_lo, self.w_hi = w_lo, w_hi
        self.entries = board.entries
        self.n = len(board.entries)
        self.pc = [POS_CODE[e["pos"]] for e in board.entries]

        # Projected points per pooled player, as the projection source
        # publishes them -- not `points`, which load_board has by now replaced
        # with the figure this player's component stats imply. The two are
        # close in the mean and up to 27 points apart on an individual, and
        # the components are the board's business rather than the report's.
        # A player only the ADP pool carries has no projection at all, so he
        # scores nothing -- which is also how the VOR side already ranks him,
        # dead last.
        self.proj = [0.0] * self.n
        for i, vi in enumerate(board.vor_row.tolist()):
            if vi >= 0:
                self.proj[i] = board.vor_players[vi]["src_points"]

        # ADP as ADP.tsv publishes it, which is the market the sim rank is
        # fitted to. Board.entries carries an `adp` too, but that one prefers
        # the projections' own adp column and only falls back to this -- two
        # different snapshots of the market, and quoting the one the ranks are
        # actually drawn from is the one that explains where a player went.
        self.adp = [float("nan")] * self.n
        for i, si in enumerate(board.sim_row.tolist()):
            if si >= 0:
                self.adp[i] = board.sim_players[si]["adp"]
        self._static_ranks()

        # starting slots each position owes, in position-code order; the flex
        # is filled afterwards out of whatever RB/WR/TE is left over
        self.starts = [lg.all_starters.get(p, 0) for p in DRAFTABLE]
        self.flex_codes = [POS_CODE[p] for p in FLEX_POS]
        self._legal_cached = lru_cache(maxsize=None)(self._legal_uncached)

    def _static_ranks(self):
        """The unsampled board: one VOR rank, one ADP rank, drawn from nothing.

        Redrawing both ranks every simulation is the right way to guess how a
        room will act, and the wrong way to show someone his own board -- a
        rank that moves under you between picks is not a rank. So the board
        the drafter reads, and the options it proposes, are ranked once off
        the published numbers and never resampled:

            rank_vor   projected points less the replacement level those same
                       projections imply, descending
            rank_adp   ADP.tsv's ADP, ascending

        Both are computed exactly as combined_draft.Board.draw computes their
        sampled counterparts, down to scoring a player the other side does not
        rank one slot past that board's last player, so the two are directly
        comparable -- the only difference is the sampling.
        """
        b = self.board
        adp = np.array([p["adp"] for p in b.sim_players], dtype=np.float64)
        sim_rank = np.empty(b.n_sim, dtype=np.float64)
        sim_rank[np.argsort(adp, kind="stable")] = np.arange(1, b.n_sim + 1)

        pts = [p["src_points"] for p in b.vor_players]
        repl = vor_draft_sim.replacement_levels(pts, b.pos_of, self.lg)
        vor = np.array(pts) - np.array([repl[p] for p in b.pos_of])
        vor_rank = np.empty(b.n_vor, dtype=np.float64)
        vor_rank[np.argsort(-vor, kind="stable")] = np.arange(1, b.n_vor + 1)

        # The VOR itself, not just its rank: what a player is worth over the
        # replacement at his position, for every player in the pool. A player
        # with no projection scores nothing, so his VOR is the whole of that
        # replacement level negated -- a real number, and a bad one, which is
        # what he is worth. roster_vor sums these.
        self.vor = [
            self.proj[i] - repl[e["pos"]] for i, e in enumerate(self.entries)
        ]

        self.rank_vor = np.where(
            b.vor_row >= 0, vor_rank[b.vor_row], b.n_vor + 1.0
        )
        self.rank_adp = np.where(
            b.sim_row >= 0, sim_rank[b.sim_row], b.n_sim + 1.0
        )

    def _legal_uncached(self, state):
        """Which of the six positions a roster in `state` may still add.

        One cached lookup per pick rather than six calls into League.legal.
        A team's pick count is the sum of its counts, so the state alone keys
        the answer.
        """
        counts = dict(zip(DRAFTABLE, state, strict=True))
        return tuple(self.lg.legal(counts, p, sum(state)) for p in DRAFTABLE)

    def legal_positions(self, counts):
        return self._legal_cached(tuple(counts))

    def lineup_points(self, roster):
        """Projected points of the best lineup this roster can start."""
        by = [[] for _ in range(N_POS)]
        for i in roster:
            by[self.pc[i]].append(self.proj[i])
        total, spare = 0.0, []
        for pc in range(N_POS):
            v = sorted(by[pc], reverse=True)
            k = self.starts[pc]
            total += sum(v[:k])
            if pc in self.flex_codes:
                spare += v[k:]
        spare.sort(reverse=True)
        return total + sum(spare[: self.lg.n_flex])

    def fills_a_slot(self, counts, pc):
        """Would a player at this position start, or take the flex?

        The question draft_tool asks before it decides how to rank a pick.
        A roster that can still seat a position has something to gain in
        projected points from taking one; a roster that cannot is drafting a
        bench player, whose points do not enter the lineup and so cannot
        tell one option from another.
        """
        if counts[pc] < self.starts[pc]:
            return True
        if pc not in self.flex_codes:
            return False
        spare = sum(
            max(0, counts[c] - self.starts[c]) for c in self.flex_codes
        )
        return spare < self.lg.n_flex

    def roster_vor(self, roster):
        """Total value over replacement of everyone on the roster.

        The measure that still says something once the starting lineup is
        settled. lineup_points counts starters, so a bench pick moves it not
        at all and every option comes back with the same team; this counts
        all fifteen, so a better bench player is worth more than a worse one,
        and the pick's effect on who is left for the rest of the draft shows
        up too.
        """
        return sum(self.vor[i] for i in roster)

    def lineup_slots(self, roster):
        """Which slot each player fills: START, FLEX or bench.

        The same assignment lineup_points scores, kept separate from it
        because it is only ever wanted once, for a finished roster, while
        lineup_points is called on every simulated roster.
        """
        slots, spare, used = {}, [], [0] * N_POS
        for i in sorted(roster, key=lambda i: -self.proj[i]):
            pc = self.pc[i]
            if used[pc] < self.starts[pc]:
                used[pc] += 1
                slots[i] = "START"
            else:
                spare.append(i)
        flex = 0
        for i in spare:
            if flex < self.lg.n_flex and self.pc[i] in self.flex_codes:
                slots[i], flex = "FLEX", flex + 1
            else:
                slots[i] = "bench"
        return slots

    # ---- per-simulation board construction ------------------------------

    def live_by_pos(self, gone):
        """The undrafted pool, split by position, as index arrays.

        Fixed for one decision and reused by every simulation and every
        candidate under it, so the split is paid for once rather than once per
        finished draft. Ascending index order makes a stable sort break score
        ties on pool order, exactly as combined_draft.py does.
        """
        out = [[] for _ in range(N_POS)]
        for i in range(self.n):
            if not gone[i]:
                out[self.pc[i]].append(i)
        return [np.array(v, dtype=np.int64) for v in out]

    def sim_boards(self, vr, sr, weights, live):
        """Every team's board for one simulation, split by position.

        The twelve blends are the same two rank vectors under twelve weights,
        so the scores are one matrix and only the ordering is per team.
        Sorting within a position is enough because best_available only ever
        looks at each position's leader.
        """
        w = weights[:, None]
        scores = w * vr[None, :] + (1.0 - w) * sr[None, :]
        boards = []
        for t in range(self.lg.n_teams):
            st = scores[t]
            boards.append(
                [
                    idx[np.argsort(st[idx], kind="stable")].tolist()
                    for idx in live
                ]
            )
        return boards, scores.tolist()

    def draw(self, rng):
        """One simulation's two rank vectors and its per-team weights."""
        vr, sr = self.board.draw(rng)
        return vr, sr, rng.uniform(self.w_lo, self.w_hi, self.lg.n_teams)

    # ---- finishing a draft ----------------------------------------------

    def finish(self, boards, scores, order, counts, user_team, roster, gone):
        """Play `order` out greedily and return the user team's final roster.

        `gone` and `counts` are consumed, so callers hand over copies. Only
        the user's roster is collected -- nothing else is reported on, and
        building twelve rosters per simulation is pure cost.
        """
        legal = self._legal_cached
        pc = self.pc
        heads = [[0] * N_POS for _ in range(self.lg.n_teams)]
        roster = list(roster)
        for team in order:
            c = counts[team]
            i = best_available(
                boards[team], heads[team], scores[team], gone, legal(tuple(c))
            )
            if i < 0:
                # A roster entered by hand -- draft_tool.py's --no-mock-draft
                # records what the room did, rules or no rules -- can be one
                # these rules can no longer complete: three QBs and a kicker
                # in the eighth. Stopping the simulation over someone else's
                # roster helps nobody, so that team takes the best man left.
                i = best_available(
                    boards[team], heads[team], scores[team], gone, ALL_LEGAL
                )
            if i < 0:
                raise RuntimeError("no player left for team %d" % (team + 1))
            gone[i] = 1
            c[pc[i]] += 1
            if team == user_team:
                roster.append(i)
        return roster

    def chunk(self, snap, candidates, n_sims, rng):
        """Both measures for every candidate over `n_sims` shared futures.

        Returns a (2 x options x sims) array: the starting lineup's projected
        points, then the roster's total VOR. The first is what the pick is
        worth while it can still reach the lineup, the second what it is
        worth once it cannot -- see draft_tool.rank_options.

        Every candidate is finished off the same drawn board, which is the
        whole point: the options differ by the pick and not by the draw. This
        is the unit of work a worker process gets, and the unit evaluate runs
        inline when there is no pool.
        """
        live = self.live_by_pos(snap.gone)
        me = snap.user_team
        out = np.empty((2, len(candidates), n_sims), dtype=np.float64)

        for s in range(n_sims):
            vr, sr, weights = self.draw(rng)
            boards, scores = self.sim_boards(vr, sr, weights, live)
            for k, cand in enumerate(candidates):
                gone = bytearray(snap.gone)
                gone[cand] = 1
                counts = [list(c) for c in snap.counts]
                counts[me][self.pc[cand]] += 1
                roster = self.finish(
                    boards,
                    scores,
                    snap.order,
                    counts,
                    me,
                    snap.roster + [cand],
                    gone,
                )
                out[0, k, s] = self.lineup_points(roster)
                out[1, k, s] = self.roster_vor(roster)
        return out

    def evaluate(self, state, candidates, n_sims, rng, pool=None):
        """The whole decision, split into N_CHUNKS independent pieces.

        The split is by simulation, never by candidate: the options have to
        share their draws for the gaps between them to mean anything, and a
        chunk keeps that property because it draws its own boards and finishes
        every candidate on each.

        The chunk count is fixed rather than taken from the pool size, so the
        answer does not depend on how many cores ran it -- --workers 1 and
        --workers 12 return bit-identical numbers, and a seed reproduces a
        draft on any machine. Each chunk gets its own spawned generator off
        the caller's, which keeps the streams independent and the whole thing
        reproducible from --seed.
        """
        snap = PickState(
            bytes(state.gone),
            [list(c) for c in state.counts],
            list(state.rosters[state.user_team]),
            [t for _, t in state.remaining()[1:]],  # after this
            state.user_team,
        )
        sizes = [
            n_sims // N_CHUNKS + (i < n_sims % N_CHUNKS)
            for i in range(N_CHUNKS)
        ]
        tasks = [
            (snap, candidates, n, r)
            for n, r in zip(sizes, rng.spawn(N_CHUNKS), strict=True)
            if n
        ]
        parts = (
            [self.chunk(*t) for t in tasks]
            if pool is None
            else list(pool.map(tasks))
        )
        return np.concatenate(parts, axis=2)
