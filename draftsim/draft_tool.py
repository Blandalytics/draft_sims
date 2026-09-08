"""Draft one team's roster off the combined board, with the clock simulated.

You sit in one seat. The other eleven teams draft themselves, and every time
the pick comes back around, the tool stops and shows what each of your options
is worth -- not as a rank, but as the projected starting lineup you would take
into the season if you took that player and let the rest of the draft happen.

    python -m draftsim --slot 5

On the clock you are shown ten options -- --options sets how many -- each
priced by 500 simulated finishes of the draft (pick_sim.py):

    the best available QB, RB, WR and TE
    the combined VOR/ADP board, filling the rest of the count

and listed best first -- by what your team is worth at the end of the draft,
not by where the board rates the player. You always get the full count: a
position you cannot legally add, a capped QB or a kicker before the final
pick, buys another player off the board rather than costing you an option.
Kickers and defenses are held back until the last three rounds, where they
belong; before that the board fills past them, and `p <name>` overrides it.

Every option is another 500 finishes, so --options is what a decision costs.
Ten of them is a fifth of a second on a warm pool; there is room to raise it.

Two different randomisations are at work, and the difference is the point.

The live draft is one world. At the first pick every rival team draws a VOR
weight w ~ Uniform(1/3, 2/3) -- how much it reads the board off projected
value rather than off the market -- and *keeps it for the whole draft*, and
one set of ranks is drawn for the draft as a whole. So the room has a settled
character: team 7 reaches for value all night, team 3 never does, and the run
on tight ends that happens, happens.

The simulations are many worlds. Every simulation redraws the ranks and deals
all twelve teams fresh weights, your own included. On the clock you do not
know how the room reads the board or how the projections will land, and the
options are priced over that ignorance rather than against the one world you
happen to be in. Priced any other way, the tool would be grading your options
against a room it had already read the mind of.

Options are compared on shared draws, so the spread between two of them is a
real difference and not sampling noise; `best%` is how often an option came
out ahead of the others on the same simulated future.

By default this is a mock draft: the eleven rivals draft themselves and the
tool only stops on your pick. Pass --no-mock-draft to follow a real one, and
it stops at all 180 picks for you to enter. The simulations run on your own
pick either way -- a rival's pick is not a decision of yours to price -- and
what you get at his is your board, --options deep, to read the pick off. Any
available player is accepted there whether or not this league's roster rules
would allow it, since the room does what it does; your own picks are still
held to the rules.

Commands on the clock:

    1..n        take that option, or that player off the board
    p <name>    take any available player instead (name, or #rank)
    b [POS]     show the board -- best available, or best at a position
    r [team]    rosters
    s           re-run the simulations on fresh draws (your pick only)
    u           undo one stop
    q           quit

League shape comes from league.py and the shared --teams/--roster/--qb...
flags, so this drafts the same league combined_draft.py simulates.
"""

import argparse
import sys

import numpy as np

from . import adp, combined_draft, pick_sim
from .adp import add_adp_args
from .combined_draft import W_HI, W_LO
from .league import DEF, DRAFTABLE, K, add_league_args, league_from_args
from .pick_sim import N_POS, N_SIMS, best_available
from .projections import add_projection_args
from .projections import from_args as projections_from_args

MY_WEIGHT = 0.6  # the VOR/ADP blend the board is shown to you in
SEED = 960122  # fixed, so the same flags give the same draft
N_OPTIONS = 10  # options priced per pick, however they break down
LATE_ROUNDS = 3  # rounds at the end where K and DST are offered
TOP_POS = ("QB", "RB", "WR", "TE")


class Draft:
    """One live draft: who is gone, who has what, and whose turn it is.

    The rival teams' boards are built once. Their weights and the draft's two
    rank vectors are drawn at the first pick and never redrawn, so the room
    keeps one character for the whole draft; the pointers into those boards
    only ever walk forward. `snapshot`/`restore` copy the whole of that state,
    which is what makes undo exact rather than approximate.
    """

    def __init__(
        self, engine, user_team, rng, my_weight=MY_WEIGHT, n_options=N_OPTIONS
    ):
        self.eng = engine
        self.lg = engine.lg
        self.user_team = user_team
        self.n_options = n_options
        self.n = engine.n
        self.order = list(self.lg.snake_order())
        self.ptr = 0
        self.gone = bytearray(self.n)
        self.counts = [[0] * N_POS for _ in range(self.lg.n_teams)]
        self.rosters = [[] for _ in range(self.lg.n_teams)]
        self.log = []

        # the one world the draft actually happens in
        self.vr, self.sr, self.weights = engine.draw(rng)
        live = engine.live_by_pos(self.gone)
        self.boards, self.scores = engine.sim_boards(
            self.vr, self.sr, self.weights, live
        )
        self.heads = [[0] * N_POS for _ in range(self.lg.n_teams)]

        # Your own view of the board. You have no weight in the draft itself,
        # only a lens to read it through -- and unlike the rivals' boards
        # above, yours is ranked off the published projections and ADPs rather
        # than off this draft's draw, so it does not reshuffle under you from
        # one pick to the next. See pick_sim.Engine._static_ranks.
        self.my_weight = my_weight
        self.my_score = (
            my_weight * engine.rank_vor + (1 - my_weight) * engine.rank_adp
        ).tolist()

    # ---- state ----------------------------------------------------------

    def done(self):
        return self.ptr >= len(self.order)

    def on_clock(self):
        rd, team = self.order[self.ptr]
        return rd, team, self.ptr + 1

    def remaining(self):
        return self.order[self.ptr :]

    def snapshot(self):
        return (
            self.ptr,
            bytes(self.gone),
            [list(c) for c in self.counts],
            [list(r) for r in self.rosters],
            [list(h) for h in self.heads],
            len(self.log),
        )

    def restore(self, snap):
        (self.ptr, gone, counts, rosters, heads, nlog) = snap
        self.gone = bytearray(gone)
        self.counts = [list(c) for c in counts]
        self.rosters = [list(r) for r in rosters]
        self.heads = [list(h) for h in heads]
        del self.log[nlog:]

    # ---- picking --------------------------------------------------------

    def take(self, i):
        """Record the player at pool index `i` as this pick."""
        rd, team, overall = self.on_clock()
        self.gone[i] = 1
        self.counts[team][self.eng.pc[i]] += 1
        self.rosters[team].append(i)
        self.log.append((overall, rd, team, i))
        self.ptr += 1
        return rd, team, overall

    def auto(self):
        """The team on the clock takes the best player its own board allows."""
        _, team, _ = self.on_clock()
        i = best_available(
            self.boards[team],
            self.heads[team],
            self.scores[team],
            self.gone,
            self.eng.legal_positions(self.counts[team]),
        )
        if i < 0:
            raise RuntimeError("no legal player at pick %d" % (self.ptr + 1))
        return self.take(i)

    def legal_for(self, team, i):
        if self.gone[i]:
            return False
        return self.eng.legal_positions(self.counts[team])[self.eng.pc[i]]

    # ---- boards ---------------------------------------------------------

    def my_board(self, pos=None):
        """Undrafted players, best first, in your own blend of the ranks."""
        idx = [
            i
            for i in range(self.n)
            if not self.gone[i]
            and (pos is None or self.eng.entries[i]["pos"] == pos)
        ]
        idx.sort(key=lambda i: (self.my_score[i], i))
        return idx

    def candidates(self):
        """The options to price: best at each position, then down the board.

        The best available QB, RB, WR and TE come first -- one is always the
        board leader, since the leader is by definition the best at his own
        position -- and the board fills the rest out to --options with the
        players not already named, which late in a draft is where a DST or K
        turns up.

        A position the roster cannot legally add is dropped rather than shown
        as an option you are not allowed to take, and the board covers for it,
        so the count holds however the four break down. That matters most in
        the rounds where it bites: with the QB and TE caps reached and a
        kicker only legal at the final pick, all four can be gone at once, and
        that many board players is exactly the choice you actually have.

        The four leaders are collected in board order rather than in QB, RB,
        WR, TE order, which only shows when --options is set below four: what
        survives the trim is then the best-ranked of them, not whichever
        position this file happens to name first.

        Kickers and defenses are withheld until the last LATE_ROUNDS rounds,
        because the blended board climbs them long before anyone would take
        one. Both are priced off a pinned baseline -- the third best at the
        position, since they are streamed and a starters x teams baseline
        would inflate them -- while a WR is priced off the 36th WR, three
        starters times twelve teams. No team may take a kicker before the
        final pick, so the best one available holds +10 VOR the whole draft,
        while the best WR left crosses below that around the ninth round and
        keeps going: WR43 and -15 by the tenth, WR52 and -28 by the
        fourteenth. So K and DST reach the top of the board on merit, and
        offering them there would spend options on a pick nobody makes. The
        board fills past them instead, and `p <name>` still takes one.
        """
        board = self.my_board()
        ok = self.eng.legal_positions(self.counts[self.user_team])
        out = self._position_leaders(board, ok)[: self.n_options]
        self._fill_from_board(out, board, ok)
        return out, board

    def _position_leaders(self, board, ok):
        """The best available QB, RB, WR and TE, in board order.

        A leader his roster cannot legally add is dropped rather than passed
        down to the next man at that position: the option is "the best TE",
        and if that is not available to you there is no second-best version
        of it worth pricing.
        """
        out, found = [], set()
        for i in board:
            pos = self.eng.entries[i]["pos"]
            if pos not in TOP_POS or pos in found:
                continue
            found.add(pos)
            if ok[self.eng.pc[i]]:
                out.append((i, "top " + pos))
            if len(found) == len(TOP_POS):
                break
        return out

    def _fill_from_board(self, out, board, ok):
        """Top `out` up to --options with the best players not on it."""
        late = self.on_clock()[0] > self.lg.roster - LATE_ROUNDS
        seen = {i for i, _ in out}
        for i in board:
            if len(out) >= self.n_options:
                return
            skip = (
                i in seen
                or not ok[self.eng.pc[i]]
                or (not late and self.eng.entries[i]["pos"] in (DEF, K))
            )
            if not skip:
                out.append((i, "board"))
                seen.add(i)


# ---- display ------------------------------------------------------------


def fmt_player(e, width=22):
    return "%-*s %-3s %-4s" % (width, e["name"][:width], e["pos"], e["nfl"])


def show_roster(draft, team, label=None):
    e = draft.eng.entries
    proj = draft.eng.proj
    ids = sorted(
        draft.rosters[team],
        key=lambda i: (DRAFTABLE.index(e[i]["pos"]), -proj[i]),
    )
    head = label or ("team %d" % (team + 1))
    if not ids:
        print("  %-9s (empty)" % head)
        return
    print(
        "  %-9s %s"
        % (head, ", ".join("%s %s" % (e[i]["pos"], e[i]["name"]) for i in ids))
    )


def show_board(draft, pos=None, n=15):
    e, eng = draft.eng.entries, draft.eng
    idx = draft.my_board(pos)[:n]
    print(
        "\n  best available%s (your blend w_vor=%.2f)"
        % ("" if pos is None else " " + pos, draft.my_weight)
    )
    print(
        "  %4s  %-31s %7s %7s %6s %8s"
        % ("#", "player", "vor_rk", "adp_rk", "adp", "proj")
    )
    for k, i in enumerate(idx, start=1):
        a = eng.adp[i]
        print(
            "  %4d  %-31s %7.0f %7.0f %6s %8.1f"
            % (
                k,
                fmt_player(e[i]),
                eng.rank_vor[i],
                eng.rank_adp[i],
                "-" if a != a else "%.1f" % a,
                eng.proj[i],
            )
        )


def by_points(opts, res):
    """Options and their simulations, best expected finish first.

    Sorted once, before anything is printed, so the number you type at the
    prompt is the number in the table. The sort is stable, which leaves
    options that finish *exactly* level -- a deep bench pick that never
    reaches the starting lineup, so every one of them scores the same -- in
    the order they were proposed, positional leaders ahead of the board.
    """
    order = np.argsort(-res.mean(axis=1), kind="stable")
    return [opts[k] for k in order], res[order]


def show_options(draft, opts, res):
    """The options table: what each pick is worth, and how often it wins."""
    e = draft.eng.entries
    adp = draft.eng.adp
    mean = res.mean(axis=1)
    p10, p90 = np.percentile(res, [10, 90], axis=1)
    best = mean.max()
    wins = np.bincount(res.argmax(axis=0), minlength=len(opts)) / res.shape[1]
    rank = {i: k for k, i in enumerate(draft.my_board(), start=1)}

    print(
        "\n  %3s %-7s %-31s %5s %6s %9s %6s %6s %8s %6s"
        % (
            "#",
            "option",
            "player",
            "rank",
            "adp",
            "team pts",
            "p10",
            "p90",
            "vs best",
            "best%",
        )
    )
    for k, (i, why) in enumerate(opts, start=1):
        gap = mean[k - 1] - best
        a = adp[i]
        print(
            "  %3d %-7s %-31s %5d %6s %9.1f %6.0f %6.0f %8s %5.0f%%"
            % (
                k,
                why,
                fmt_player(e[i]),
                rank[i],
                "-" if a != a else "%.1f" % a,
                mean[k - 1],
                p10[k - 1],
                p90[k - 1],
                "best" if gap == 0 else "%+.1f" % gap,
                100 * wins[k - 1],
            )
        )


def show_header(draft, team=None):
    """The banner for whichever pick is on the clock.

    Yours gets the heavy rule and your roster under it, since that is the one
    you are deciding; a rival's gets a light one, because with --no-mock-draft
    those come round eleven times as often and should not shout.
    """
    rd, on_clock, overall = draft.on_clock()
    slot = (overall - 1) % draft.lg.n_teams + 1
    mine = team is None
    print("\n" + ("=" if mine else "-") * 100)
    print(
        " round %d, pick %d (overall %d) -- %s"
        % (
            rd,
            slot,
            overall,
            "YOUR PICK" if mine else "team %d on the clock" % (on_clock + 1),
        )
    )
    print(("=" if mine else "-") * 100)
    if mine:
        show_roster(draft, draft.user_team, "you have")


def show_between(draft, picks, mock=True):
    if not picks:
        return
    e = draft.eng.entries
    print("\n  picks since your last:")
    for overall, rd, team, i in picks:
        # the weight explains a pick the tool made; it explains nothing about
        # one you typed in, so it is left off when you are entering them
        print(
            "    %3d.%02d  team %-2d  %-31s%s"
            % (
                rd,
                (overall - 1) % draft.lg.n_teams + 1,
                team + 1,
                fmt_player(e[i]),
                "  (w %.2f)" % draft.weights[team] if mock else "",
            )
        )


def show_final(draft, mock=True):
    e, eng = draft.eng.entries, draft.eng
    print("\n" + "=" * 100)
    print(" draft complete")
    print("=" * 100)
    mine = draft.rosters[draft.user_team]
    slots = eng.lineup_slots(mine)
    print("\n  your roster")
    for i in sorted(
        mine, key=lambda i: (DRAFTABLE.index(e[i]["pos"]), -eng.proj[i])
    ):
        print(
            "    %-6s %-31s %8.1f" % (slots[i], fmt_player(e[i]), eng.proj[i])
        )
    print(
        "\n  starting lineup, flex included: %.1f projected points"
        % eng.lineup_points(mine)
    )
    print("\n  every team's projected starting points")
    tot = [(eng.lineup_points(r), t) for t, r in enumerate(draft.rosters)]
    for k, (pts, t) in enumerate(sorted(tot, reverse=True), start=1):
        you = t == draft.user_team
        # a weight only means something for a team the tool drafted
        w = (
            ""
            if not mock
            else "  w %-5s" % ("-" if you else "%.2f" % draft.weights[t])
        )
        print(
            "    %2d.  team %-2d%s %8.1f%s"
            % (k, t + 1, w, pts, "  <- you" if you else "")
        )


# ---- the loop -----------------------------------------------------------


def resolve(draft, text):
    """A player index from a name fragment or a `#rank` off your board."""
    board = draft.my_board()
    text = text.strip()
    if text.startswith("#"):
        try:
            k = int(text[1:])
        except ValueError:
            return None, "not a board rank: %s" % text
        if not 1 <= k <= len(board):
            return None, "board rank out of range: %s" % text
        return board[k - 1], None
    hits = [
        i
        for i in board
        if text.lower() in draft.eng.entries[i]["name"].lower()
    ]
    if not hits:
        return None, "nobody available matches %r" % text
    if len(hits) > 1:
        names = ", ".join(draft.eng.entries[i]["name"] for i in hits[:6])
        return None, "%d players match %r: %s" % (len(hits), text, names)
    return hits[0], None


class Console:
    """The clock: what is shown at a stop, and what a typed command does.

    Split out of one long loop so that no single piece of it has to hold the
    whole draft in its head. The state that loop threaded through itself --
    the undo stack, the picks since your last, the options on screen and what
    they simulated to -- lives here instead, and each command is a method that
    reads it.

    Mock drafting, the default, means the eleven rivals draft themselves off
    their own weighted boards and the only stop is your own pick. Turn it off
    and the tool stops at all 180: it is then following a real draft rather
    than inventing one, so it has no business guessing what the room does.

    The simulations run on your pick either way and only there. A rival's pick
    is not a decision you are making, so there is nothing to price -- what you
    get instead is your own board, --options deep, to read his pick off. Any
    available player is accepted there, legal by this league's roster rules or
    not, because the room does what it does and the tool's job is to record
    it. Your own picks are still held to the rules.

    One snapshot is pushed per stop, so `u` rewinds one stop: your previous
    pick when mock drafting, the previous pick of any team when not.
    """

    HELP = "  ? 1-%d to pick, p <name>, b [POS], r, s, u, q"

    def __init__(self, draft, n_sims, rng, auto=False, mock=True, pool=None):
        self.draft = draft
        self.n_sims = n_sims
        self.rng = rng
        self.auto = auto
        self.mock = mock
        self.pool = pool
        self.snaps = []  # one per stop; `u` pops two and restores
        self.between = []  # picks since your last, for the next header
        self.opts = []  # what is on screen, best first
        self.res = None  # its simulations, or None at a rival's pick

    # ---- the loop -------------------------------------------------------

    def run(self):
        """Walk the draft, stopping wherever a pick has to be entered."""
        d = self.draft
        while not d.done():
            _, team, _ = d.on_clock()
            mine = team == d.user_team
            if self.mock and not mine:
                d.auto()
                self.between.append(d.log[-1])
                continue
            prompt = self._present(mine, team)
            self.snaps.append(d.snapshot())
            if not self._on_clock(prompt, mine, team):
                return False
        return True

    def _present(self, mine, team):
        """Show the stop, and return the prompt it should be answered at."""
        d = self.draft
        show_header(d, None if mine else team)
        if not mine:
            self.opts = [(i, "board") for i in d.my_board()[: d.n_options]]
            self.res = None
            show_board(d, None, d.n_options)
            return "\n  team %d pick> " % (team + 1)
        show_between(d, self.between, self.mock)
        self.between = []
        self.opts, _ = d.candidates()
        print(
            "\n  simulating %d finishes for each of %d options..."
            % (self.n_sims, len(self.opts)),
            end=" ",
            flush=True,
        )
        self._simulate()
        return "\n  pick> "

    def _on_clock(self, prompt, mine, team):
        """Read commands until the clock moves on. False means quit."""
        while True:
            choice = self._read(prompt)
            if choice is None:
                return False
            if not choice:
                continue
            cmd, _, arg = choice.partition(" ")
            if cmd.lower() == "q":
                return False
            if self._act(cmd.lower(), arg.strip(), mine, team):
                return True

    def _read(self, prompt):
        """One line from the drafter, or None if he is done with us."""
        if self.auto:
            print("%s1 (auto)" % prompt)  # the table is sorted, so 1 is best
            return "1"
        try:
            return input(prompt).strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  quit")
            return None

    def _act(self, cmd, arg, mine, team):
        """One command. True once the pick is settled and the clock moves."""
        if cmd == "b":
            return self._show_board(arg)
        if cmd == "r":
            return self._show_rosters(arg)
        if cmd == "s":
            return self._resimulate()
        if cmd == "u":
            return self._undo()
        return self._take(cmd, arg, mine, team)

    # ---- commands -------------------------------------------------------

    def _show_board(self, arg):
        words = arg.split()
        pos = [w.upper() for w in words if not w.isdigit()]
        num = [int(w) for w in words if w.isdigit()]
        if pos and pos[0] not in DRAFTABLE:
            print("  no such position: %s" % pos[0])
        else:
            show_board(
                self.draft,
                pos[0] if pos else None,
                num[0] if num else (25 if pos else 15),
            )
        return False

    def _show_rosters(self, arg):
        d = self.draft
        if not arg.isdigit():
            print()
            for t in range(d.lg.n_teams):
                show_roster(
                    d, t, "you" if t == d.user_team else "team %d" % (t + 1)
                )
        elif 1 <= int(arg) <= d.lg.n_teams:
            show_roster(d, int(arg) - 1)
        else:
            print("  no such team: %s" % arg)
        return False

    def _simulate(self):
        """Price the options on screen, best first by what they return."""
        d = self.draft
        res = d.eng.evaluate(
            d, [i for i, _ in self.opts], self.n_sims, self.rng, self.pool
        )
        print("done")
        self.opts, self.res = by_points(self.opts, res)
        show_options(d, self.opts, self.res)

    def _resimulate(self):
        if self.res is None:
            print("  nothing to re-simulate -- this is not your pick")
            return False
        print("  re-simulating...", end=" ", flush=True)
        self._simulate()
        return False

    def _undo(self):
        # two pops: the stop we are standing at, then the one to go back to,
        # which the loop pushes again when it arrives there
        if len(self.snaps) < 2:
            print("  nothing to undo")
            return False
        self.snaps.pop()
        self.draft.restore(self.snaps.pop())
        self.between = []
        return True

    def _take(self, cmd, arg, mine, team):
        d = self.draft
        i = self._chosen(cmd, arg)
        if i is None:
            return False
        # your roster is held to the league's rules; a rival's is whatever the
        # room actually did, and the tool is only writing it down
        if mine and not d.legal_for(team, i):
            print(
                "  %s is not a legal pick for your roster"
                % d.eng.entries[i]["name"]
            )
            return False
        d.take(i)
        if mine:
            print("  you take %s" % fmt_player(d.eng.entries[i]))
        else:
            self.between.append(d.log[-1])
            print(
                "  team %d takes %s" % (team + 1, fmt_player(d.eng.entries[i]))
            )
        return True

    def _chosen(self, cmd, arg):
        """The pool index a command names, or None with the reason printed."""
        if cmd == "p":
            i, err = resolve(self.draft, arg)
            if err:
                print("  " + err)
            return i
        if cmd.isdigit() and 1 <= int(cmd) <= len(self.opts):
            return self.opts[int(cmd) - 1][0]
        print(self.HELP % len(self.opts))
        return None


def run(draft, n_sims, rng, auto=False, mock=True, pool=None):
    """Draft to the last pick. False if the drafter quit part way."""
    return Console(draft, n_sims, rng, auto, mock, pool).run()


def main():
    ap = add_projection_args(
        add_adp_args(
            add_league_args(
                argparse.ArgumentParser(
                    description=__doc__,
                    formatter_class=argparse.RawDescriptionHelpFormatter,
                )
            )
        )
    )
    ap.add_argument(
        "--slot", type=int, default=1, help="your draft slot, 1..teams"
    )
    ap.add_argument(
        "--sims",
        type=int,
        default=N_SIMS,
        help="simulated finishes per option",
    )
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument(
        "--my-weight",
        type=float,
        default=MY_WEIGHT,
        help="the VOR weight your own board is shown in; it "
        "orders the options, it does not draft for you",
    )
    ap.add_argument(
        "--mock-draft",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="let the rivals draft themselves; --no-mock-draft "
        "stops at every pick and you enter them all, for "
        "following a real draft",
    )
    ap.add_argument("--vor-weight-lo", type=float, default=W_LO)
    ap.add_argument("--vor-weight-hi", type=float, default=W_HI)
    ap.add_argument(
        "--matched-only",
        action="store_true",
        help="draft only players both simulators rank",
    )
    ap.add_argument(
        "--options",
        type=int,
        default=N_OPTIONS,
        help="options priced at each of your picks. Every one is "
        "another %d simulated finishes, so this is also the "
        "cost of a decision" % N_SIMS,
    )
    ap.add_argument(
        "--workers",
        type=int,
        default=0,
        help="worker processes for the simulations; 0 is one per "
        "core, 1 runs them inline. The answer does not "
        "depend on this -- only how fast it arrives",
    )
    ap.add_argument(
        "--auto",
        action="store_true",
        help="take the highest-scoring option every time, no "
        "prompt -- for checking the tool end to end",
    )
    a = ap.parse_args()

    lg = league_from_args(a)
    if not 1 <= a.slot <= lg.n_teams:
        raise SystemExit("--slot must be 1..%d" % lg.n_teams)
    if a.options < 1:
        raise SystemExit("--options must be at least 1")

    # scraped once here and handed down, so the workers below read the
    # board this process pulled rather than each pulling its own
    adp_path = adp.path_from_args(a)
    source = projections_from_args(a, adp_path)
    print("loading the board...", end=" ", flush=True)
    board = combined_draft.Board(lg, a.matched_only, adp_path, source)
    eng = pick_sim.Engine(board, lg, a.vor_weight_lo, a.vor_weight_hi)
    print("%d players" % eng.n)
    for key, win, lose in board.pid_clashes:
        print(
            "  pid %s claimed twice: kept %s, demoted %s to projection only"
            % (key, win, lose)
        )

    rng = np.random.default_rng(a.seed)
    draft = Draft(eng, a.slot - 1, rng, a.my_weight, a.options)
    print(
        "%s: %d teams x %d rounds, you are team %d (seed %d)"
        % (lg.name, lg.n_teams, lg.roster, a.slot, a.seed)
    )
    if a.mock_draft:
        print(
            "rival vor weights: %s"
            % "  ".join(
                "T%d %.2f" % (t + 1, w)
                for t, w in enumerate(draft.weights)
                if t != draft.user_team
            )
        )
    else:
        print("entering every pick by hand; simulations run on yours only")

    # One pool for the whole draft: its workers each load the board once, and
    # rebuilding that per pick would cost more than the parallelism returns.
    # --workers 1 skips it entirely and runs the chunks inline, which is the
    # same arithmetic in the same order -- see Engine.evaluate.
    pool = None
    if a.workers != 1:
        pool = pick_sim.Parallel(
            lg,
            a.matched_only,
            a.vor_weight_lo,
            a.vor_weight_hi,
            a.workers,
            adp_path,
            source,
        )
        print("simulating on %d worker processes" % pool.workers)
    try:
        if run(draft, a.sims, rng, a.auto, a.mock_draft, pool):
            # only the tool's picks are guaranteed to obey the roster rules,
            # so a hand-entered draft is checked where it was enforced: yours
            counts = [
                dict(zip(DRAFTABLE, c, strict=True)) for c in draft.counts
            ]
            lg.check(counts if a.mock_draft else [counts[draft.user_team]])
            show_final(draft, a.mock_draft)
    finally:
        if pool is not None:
            pool.__exit__(None, None, None)
    return 0


if __name__ == "__main__":
    sys.exit(main())
