"""League scoring: what each stat is worth, and everything that follows.

Scoring is a property of the league, not of the projections, so it is an
input rather than a constant. It defaults to Yahoo's default rules -- 0.5
PPR, four points a passing touchdown -- and is changed either one category
at a time or as a whole:

    python -m draftsim --score rec=1                 full PPR
    python -m draftsim --score rec=1,pass_tds=6      ...and 6-point passing
    python -m draftsim --scoring my_league.json      a whole rule set

The file is flat JSON and carries only what differs from the default:

    {"rec": 1, "pass_tds": 6, "pass_yds": 0.05}

`python -m draftsim.scoring` prints the rules in force and the names to use;
`--json` prints them in the form the file takes, ready to edit and hand back.

The rules go two places, and the point of holding them in one object is that
those two cannot drift apart:

    into calc_projections, where projections_table() scores each source's
      projection into the points board -- the rules also decide which
      component stats come back at all, since a category worth nothing is
      never aggregated
    into vor_draft_sim, which scores the component stats it samples on the
      Python side, a fresh season per simulated draft

So changing the scoring changes the projections, which is why a Scoring
carries a `key`: a short hash of the rules that goes into the projections
cache filename. The default hashes to nothing and keeps the plain filename,
which is the one this project publishes; every other rule set caches beside
it under its own name and cannot be served a board scored by other rules.

What a category is called is ffanalytics' business, not this project's, and
the names below are its names. `all_pos = TRUE` means the category counts
for every position rather than only the obvious one -- a quarterback's
receptions, a receiver's passing touchdowns.

One rule is carried but inert, and the reason is worth knowing. The sliding
scale a defense is paid on, `pts_bracket`, is scored -- the sources do
project points allowed -- but what the bracket produces is then multiplied
by `dst_pts_allowed`, which is 0 here as it is in Yahoo's rules and in
ffanalytics'. So every defense's points are exactly the sum of its other
categories, as the published board shows. Set `dst_pts_allowed` to 1 and the
bracket pays what it says; the rules are kept whole either way, so that a
league that does pay for it is scored correctly.
"""

import argparse
import copy
import hashlib
import json
import sys

BRACKET = "pts_bracket"
FLAGS = ("all_pos",)

# Yahoo default scoring, 0.5 PPR.
# help.yahoo.com/kb/default-league-settings-fantasy-football-sln6489.html
#
# Every category ffanalytics scores for a drafted position is listed, worth
# nothing where Yahoo pays nothing, so that the names are all here to be set
# and the rule set reads as a complete answer rather than a partial one.
DEFAULT = {
    "pass": {
        "pass_att": 0,
        "pass_comp": 0,
        "pass_inc": 0,
        "pass_yds": 0.04,
        "pass_tds": 4,
        "pass_int": -1,
        "pass_40_yds": 0,
        "pass_300_yds": 0,
        "pass_350_yds": 0,
        "pass_400_yds": 0,
    },
    "rush": {
        "all_pos": True,
        "rush_yds": 0.1,
        "rush_tds": 6,
        "rush_att": 0,
        "rush_40_yds": 0,
        "rush_100_yds": 0,
        "rush_150_yds": 0,
        "rush_200_yds": 0,
    },
    "rec": {
        "all_pos": True,
        "rec": 0.5,
        "rec_yds": 0.1,
        "rec_tds": 6,
        "rec_40_yds": 0,
        "rec_100_yds": 0,
        "rec_150_yds": 0,
        "rec_200_yds": 0,
    },
    "misc": {
        "all_pos": True,
        "fumbles_lost": -2,
        "fumbles_total": 0,
        "sacks": 0,
        "two_pts": 2,
    },
    "kick": {
        "fg_0019": 3,
        "fg_2029": 3,
        "fg_3039": 3,
        "fg_4049": 4,
        "fg_50": 5,
        "xp": 1,
        "fg_miss": 0,
    },
    "ret": {"all_pos": True, "return_tds": 6, "return_yds": 0},
    "dst": {
        "dst_int": 2,
        "dst_fum_rec": 2,
        "dst_sacks": 1,
        "dst_safety": 2,
        "dst_td": 6,
        "dst_blk": 2,
        "dst_ret_yds": 0,
        "dst_pts_allowed": 0,
    },
    BRACKET: [
        [0, 10],
        [6, 7],
        [13, 4],
        [20, 1],
        [27, 0],
        [34, -1],
        [99, -4],
    ],
}

# which group each category belongs to, since a rule set is given flat and
# R wants it grouped. No name appears in two groups.
CATEGORY = {
    stat: group
    for group, rules in DEFAULT.items()
    if group != BRACKET
    for stat in rules
    if stat not in FLAGS
}


def _num(text):
    """A scoring value, as a number."""
    try:
        return float(text)
    except (TypeError, ValueError) as exc:
        raise SystemExit(
            "scoring values must be numbers, not %r" % text
        ) from exc


def _plain(v):
    """A number as a league would write it: 0.5, 6, -1."""
    return "%g" % v


def _canon(rules):
    """A rule set as one string, so two of them compare and hash alike.

    Values arrive as whatever wrote them -- 4 from the table below, 4.0 from
    a file or a flag -- and those two rule sets are the same rule set.
    """
    out = {}
    for group, rules_of in rules.items():
        if group == BRACKET:
            out[group] = [[float(t), float(p)] for t, p in rules_of]
        else:
            out[group] = {
                k: v if isinstance(v, bool) else float(v)
                for k, v in rules_of.items()
            }
    return json.dumps(out, sort_keys=True)


def override(rules, changes):
    """`rules` with flat `changes` applied, as a new rule set."""
    out = copy.deepcopy(rules)
    for stat, value in changes.items():
        if stat == BRACKET:
            out[BRACKET] = [[_num(t), _num(p)] for t, p in value]
            continue
        group = CATEGORY.get(stat)
        if group is None:
            raise SystemExit(
                "unknown scoring category %r. The names are ffanalytics': "
                "run `python -m draftsim.scoring` to list them" % stat
            )
        out[group][stat] = _num(value)
    return out


class Scoring:
    """One league's scoring rules, and the two forms they are needed in.

    `values` is the flat category -> points table both the aggregation and
    the simulators score with, holding only what is worth something, and
    `rules` is the whole rule set including the categories worth nothing.
    `key` is what tells two rule sets' cached projections apart, and is
    empty for the default so that the published cache keeps its name.
    """

    def __init__(self, rules=None):
        self.rules = copy.deepcopy(DEFAULT) if rules is None else rules
        self.values = {
            stat: float(v)
            for group, rules_of in self.rules.items()
            if group != BRACKET
            for stat, v in rules_of.items()
            if stat not in FLAGS and v
        }
        canon = _canon(self.rules)
        self.key = (
            ""
            if canon == _canon(DEFAULT)
            else hashlib.sha1(canon.encode()).hexdigest()[:8]
        )

    def is_default(self):
        return not self.key

    def diff(self):
        """The categories this scores differently from the default."""
        return {
            stat: self.rules[group][stat]
            for stat, group in CATEGORY.items()
            if self.rules[group][stat] != DEFAULT[group][stat]
        }

    def summary(self):
        """One line, for a tool that has to say what it is drafting under."""
        if self.is_default():
            return "Yahoo default, 0.5 PPR"
        changed = [
            "%s %s" % (stat, _plain(v)) for stat, v in self.diff().items()
        ]
        if self.rules[BRACKET] != DEFAULT[BRACKET]:
            changed.append("pts_bracket")
        return "custom [%s]: %s" % (self.key, ", ".join(changed))

    def flat(self):
        """Every category and its value, the form a scoring file takes."""
        out = {
            stat: self.rules[group][stat] for stat, group in CATEGORY.items()
        }
        out[BRACKET] = self.rules[BRACKET]
        return out

    def table(self):
        """The rules as lines to print: what pays, and what was changed."""
        changed = self.diff()
        out = []
        for group, rules_of in self.rules.items():
            if group == BRACKET:
                continue
            shown = [
                "%s %s%s" % (k, _plain(v), "*" if k in changed else "")
                for k, v in rules_of.items()
                if k not in FLAGS and (v or k in changed)
            ]
            if shown:
                out.append("  %-5s %s" % (group, "  ".join(shown)))
        out.append(
            "  %-5s %s"
            % (
                BRACKET,
                " ".join(
                    "%s:%s" % (_plain(t), _plain(p))
                    for t, p in self.rules[BRACKET]
                ),
            )
        )
        return out


def load_file(path):
    """A scoring file: flat JSON of category -> points."""
    try:
        with open(path, encoding="utf-8") as f:
            blob = json.load(f)
    except OSError as exc:
        raise SystemExit(
            "cannot read scoring file %s: %s" % (path, exc)
        ) from exc
    except ValueError as exc:
        raise SystemExit(
            "scoring file %s is not JSON: %s" % (path, exc)
        ) from exc
    if not isinstance(blob, dict):
        raise SystemExit(
            "a scoring file is a JSON object of category -> points, "
            'like {"rec": 1, "pass_tds": 6}'
        )
    return blob


def parse_pairs(items):
    """`--score rec=1 --score pass_tds=6,pass_yds=0.05` as a flat dict."""
    changes = {}
    for item in items:
        for part in item.split(","):
            if not part.strip():
                continue
            stat, sep, value = part.partition("=")
            if not sep:
                raise SystemExit(
                    "--score takes category=points, like rec=1, not %r" % part
                )
            changes[stat.strip()] = _num(value)
    return changes


def add_scoring_args(ap):
    g = ap.add_argument_group(
        "scoring",
        "Yahoo default (0.5 PPR) unless changed. Changing it changes the "
        "projections, which are then cached under their own name; run "
        "`python -m draftsim.scoring` to see the rules and the category "
        "names",
    )
    g.add_argument(
        "--scoring",
        metavar="FILE",
        default=None,
        help="JSON file of category -> points, carrying only what "
        "differs from the default",
    )
    g.add_argument(
        "--score",
        action="append",
        default=[],
        metavar="CAT=PTS",
        help="one override, or several comma-separated; repeatable. "
        "Applied after --scoring",
    )
    return ap


def from_args(a):
    """The scoring the flags describe, defaults included."""
    changes = {}
    if getattr(a, "scoring", None):
        changes.update(load_file(a.scoring))
    changes.update(parse_pairs(getattr(a, "score", None) or []))
    return Scoring(override(DEFAULT, changes)) if changes else Scoring()


def main(argv=None):
    ap = add_scoring_args(
        argparse.ArgumentParser(
            description=__doc__,
            formatter_class=argparse.RawDescriptionHelpFormatter,
        )
    )
    ap.add_argument(
        "--json",
        action="store_true",
        help="print the rules as a scoring file, ready to edit and "
        "pass back with --scoring",
    )
    a = ap.parse_args(argv)
    s = from_args(a)
    if a.json:
        print(json.dumps(s.flat(), indent=2))
        return 0
    print("scoring: %s" % s.summary())
    for line in s.table():
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
