"""The four bits of rvest the scrapes lean on, out of the standard library.

source_scrapes.R reads pages with rvest, and four of its verbs carry almost
all of the work: `read_html`, `html_elements` with a CSS selector,
`html_attr`, and `html_table`. Nothing else in this project has ever needed
lxml or BeautifulSoup -- adp.py parses its fragment with `html.parser` -- so
rather than take a dependency for six pages a year, those four verbs are
built here on top of `html.parser`:

    parse(html)          read_html
    node.select(sel)     html_elements / html_element
    node.attr(name)      html_attr
    node.table()         html_table

The selector engine understands what those scrapes actually write: tag, #id,
.class, [attr='v'] and [attr *='v'], :first-child and :nth-child(n), joined
by descendant (space) or child (>) combinators. It is not CSS, it is the
subset that reads the pages, and an unsupported selector raises rather than
quietly matching nothing.

Two text functions, because rvest has two and the R uses both, for different
reasons:

    text()   the raw text under a node, as `xml_text` gives it -- indentation
             and newlines from the markup left in. `table()` uses it, and the
             CBS and FFToday scrapes depend on that: CBS pulls a player, his
             position and his team out of one cell by splitting on runs of
             two or more spaces, which exist only because the whitespace
             between the cell's spans survived
    text2()  the text as rendered, which is `html_text2`: whitespace squished,
             a newline where a block element or a <br> ends a line, a tab
             between table cells. Header rows are read with this, because a
             CBS header cell holds both the short label and the tooltip that
             spells it out, and only the line break between them tells the
             two apart

Real pages are not well-formed -- FFToday still writes 1990s markup with
unclosed <tr> and <td> -- so the parser closes tags the way a browser does
rather than trusting the document.
"""

import re
from html.parser import HTMLParser

# Elements that never have children, so a start tag is the whole element.
VOID = frozenset(
    (
        "area", "base", "basefont", "br", "col", "embed", "hr", "img",
        "input", "link", "meta", "param", "source", "track", "wbr",
    )
)

# Elements a browser lays out on their own line. text2() breaks lines here.
BLOCK = frozenset(
    (
        "address", "article", "aside", "blockquote", "body", "caption",
        "center", "div", "dl", "dt", "dd", "fieldset", "figcaption",
        "figure", "footer", "form", "h1", "h2", "h3", "h4", "h5", "h6",
        "header", "hr", "li", "main", "nav", "ol", "p", "pre", "section",
        "table", "tbody", "tfoot", "thead", "tr", "ul",
    )
)

# What an open tag implies about the tags already open: seeing any of these
# closes the ones listed, which is how a page with no </td> still parses.
CLOSES = {
    "li": ("li",),
    "p": ("p",),
    "option": ("option",),
    "tr": ("td", "th", "tr"),
    "td": ("td", "th"),
    "th": ("td", "th"),
    "thead": ("td", "th", "tr", "thead", "tbody", "tfoot"),
    "tbody": ("td", "th", "tr", "thead", "tbody", "tfoot"),
    "tfoot": ("td", "th", "tr", "thead", "tbody", "tfoot"),
}

_SIMPLE = re.compile(
    r"""
      (?P<star>\*)
    | (?P<tag>[A-Za-z][\w-]*)
    | \#(?P<id>[\w:.-]+)
    | \.(?P<cls>[\w-]+)
    | \[\s*(?P<attr>[\w:-]+)\s*
        (?:(?P<op>[*^$~|]?=)\s*(?P<val>'[^']*'|"[^"]*"|[^\]]+?))?\s*\]
    | :(?P<pseudo>[\w-]+)(?:\((?P<arg>[^)]*)\))?
    """,
    re.VERBOSE,
)


class Node:
    """One element. Text is kept as plain strings among the children."""

    __slots__ = ("tag", "attrs", "children", "parent", "order")

    def __init__(self, tag, attrs=None, parent=None, order=0):
        self.tag = tag
        self.attrs = attrs or {}
        self.children = []
        self.parent = parent
        self.order = order

    def __repr__(self):
        return "<%s %r>" % (self.tag, self.attrs.get("class", ""))

    def attr(self, name):
        """html_attr: the attribute, or None where the element has none."""
        return self.attrs.get(name)

    def elements(self):
        """Child elements, skipping the text between them."""
        return [c for c in self.children if isinstance(c, Node)]

    def walk(self):
        """Every element under this one, in document order."""
        for child in self.elements():
            yield child
            yield from child.walk()

    def text(self):
        """xml_text: everything under here, markup whitespace and all."""
        out = []
        for child in self.children:
            out.append(child if isinstance(child, str) else child.text())
        return "".join(out)

    def text2(self):
        """html_text2: the text as a browser would lay it out."""
        return _tidy("".join(_render(self)))

    def select(self, selector):
        """html_elements: every descendant the selector matches."""
        found = {}
        for node in _match_selector(self, selector):
            found[node.order] = node
        return [found[k] for k in sorted(found)]

    def select_one(self, selector):
        """html_element: the first match, or None."""
        found = self.select(selector)
        return found[0] if found else None

    def table(self):
        """html_table: the rows of a table, as text, one list per row.

        A cell spanning columns is repeated across them and a cell spanning
        rows is repeated down them, so every row comes back the same width
        and column *n* is column *n* in each -- which is what lets the
        scrapes name their columns by position.
        """
        return _table(self)


def parse(html):
    """read_html: a document, as a `Node` whose tag is "[document]"."""
    parser = _Parser()
    parser.feed(html)
    parser.close()
    return parser.root


class _Parser(HTMLParser):
    """Tag soup in, tree out, closing tags the way a browser would."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.root = Node("[document]")
        self._open = [self.root]
        self._n = 0

    def handle_starttag(self, tag, attrs):
        for implied in CLOSES.get(tag, ()):
            if self._find(implied) is not None:
                self._close(implied)
                break
        node = self._append(tag, attrs)
        if tag not in VOID:
            self._open.append(node)

    def handle_startendtag(self, tag, attrs):
        self._append(tag, attrs)

    def handle_endtag(self, tag):
        if tag not in VOID and self._find(tag) is not None:
            self._close(tag)

    def handle_data(self, data):
        self._open[-1].children.append(data)

    def _append(self, tag, attrs):
        self._n += 1
        node = Node(tag, dict(attrs), self._open[-1], self._n)
        self._open[-1].children.append(node)
        return node

    def _find(self, tag):
        """How far up the stack the nearest open `tag` is, or None.

        The search stops at a table, so that a nested table's first <tr>
        closes a cell of its own table and not the outer cell it sits in --
        which is what would otherwise flatten a page like FFToday's, where
        the whole layout is tables inside tables and few of them close.
        """
        for i in range(len(self._open) - 1, 0, -1):
            if self._open[i].tag == tag:
                return i
            if self._open[i].tag == "table" and tag != "table":
                return None
        return None

    def _close(self, tag):
        i = self._find(tag)
        if i is not None:
            del self._open[i:]


def _render(node):
    """text2's pieces: squished text, with breaks where a browser has them."""
    if node.tag == "br":
        return ["\n"]
    out = []
    if node.tag in BLOCK:
        out.append("\n")
    elif node.tag in ("td", "th") and _preceding_cell(node):
        out.append("\t")
    for child in node.children:
        if isinstance(child, str):
            out.append(re.sub(r"\s+", " ", child))
        else:
            out.extend(_render(child))
    if node.tag in BLOCK:
        out.append("\n")
    return out


def _preceding_cell(node):
    """Is there a cell before this one in the row? Then a tab goes first."""
    if node.parent is None:
        return False
    for sibling in node.parent.elements():
        if sibling is node:
            return False
        if sibling.tag in ("td", "th"):
            return True
    return False


def _tidy(text):
    """Trim each line and drop the blank ones, as html_text2 leaves it."""
    lines = [line.strip(" ") for line in text.split("\n")]
    return "\n".join([line for line in lines if line.strip()])


def _table(node):
    """The rows of a table, nested tables' rows left to their own table."""
    if node.tag == "tr":
        rows = [node]
    else:
        rows = [r for r in node.walk() if r.tag == "tr" and _own(r, node)]
    spans, out = {}, []
    for row in rows:
        cells = [c for c in row.walk() if c.tag in ("td", "th")]
        out.append(_row(cells, spans))
    width = max([len(r) for r in out], default=0)
    return [r + [""] * (width - len(r)) for r in out]


def _row(cells, spans):
    """One row's text, with whatever earlier rowspans put into it."""
    out, col = [], 0
    for cell in cells:
        col = _fill(out, spans, col)
        text = cell.text().strip()
        rowspan = _span(cell, "rowspan")
        for _ in range(_span(cell, "colspan")):
            out.append(text)
            if rowspan > 1:
                spans[col] = [text, rowspan - 1]
            col += 1
    _fill(out, spans, col)
    return out


def _fill(out, spans, col):
    """Take whatever a cell above is still spanning into this row."""
    while spans.get(col, (None, 0))[1] > 0:
        out.append(spans[col][0])
        spans[col][1] -= 1
        col += 1
    return col


def _span(cell, name):
    value = (cell.attrs.get(name) or "").strip()
    return int(value) if value.isdigit() and int(value) > 0 else 1


def _own(row, node):
    """Is this row `node`'s own, rather than a nested table's?"""
    parent = row.parent
    while parent is not None and parent is not node:
        if parent.tag == "table":
            return False
        parent = parent.parent
    return parent is node


def _match_selector(root, selector):
    """Walk the selector left to right, narrowing the candidates."""
    current = [root]
    for combinator, compound in _split(selector):
        simple = _compile(compound)
        found = []
        for node in current:
            pool = node.elements() if combinator == ">" else node.walk()
            found.extend([n for n in pool if _matches(n, simple)])
        current = found
        if not current:
            break
    return current


def _split(selector):
    """The selector as (combinator, compound) pairs, left to right."""
    parts, state = [], {"combinator": " ", "buf": "", "depth": 0}
    for char in selector.strip():
        _step(parts, state, char)
    if state["buf"]:
        parts.append((state["combinator"], state["buf"]))
    if not parts:
        raise ValueError("empty selector")
    return parts


def _step(parts, state, char):
    """One character of the selector, minding the brackets it may be in."""
    if char == "[":
        state["depth"] += 1
    elif char == "]":
        state["depth"] -= 1
    if state["depth"] or char not in " >":
        state["buf"] += char
        return
    if state["buf"]:
        parts.append((state["combinator"], state["buf"]))
        state["buf"], state["combinator"] = "", " "
    if char == ">":
        state["combinator"] = ">"


def _compile(compound):
    """One compound selector, as the tests to run against a node."""
    tests, pos = [], 0
    while pos < len(compound):
        m = _SIMPLE.match(compound, pos)
        if m is None:
            raise ValueError("unsupported selector: %r" % compound)
        tests.append(m.groupdict())
        pos = m.end()
    return tests


def _matches(node, tests):
    return all(_test(node, t) for t in tests)


def _test(node, t):
    if t["star"]:
        return True
    if t["tag"]:
        return node.tag == t["tag"].lower()
    if t["id"]:
        return node.attrs.get("id") == t["id"]
    if t["cls"]:
        return t["cls"] in (node.attrs.get("class") or "").split()
    if t["attr"]:
        return _test_attr(node, t)
    return _test_pseudo(node, t)


def _test_attr(node, t):
    value = node.attrs.get(t["attr"])
    if value is None:
        return False
    if not t["op"]:
        return True
    wanted = (t["val"] or "").strip().strip("'\"")
    if t["op"] == "=":
        return value == wanted
    if t["op"] == "*=":
        return wanted in value
    if t["op"] == "^=":
        return value.startswith(wanted)
    if t["op"] == "$=":
        return value.endswith(wanted)
    raise ValueError("unsupported attribute operator: %r" % t["op"])


def _test_pseudo(node, t):
    pseudo = t["pseudo"]
    if node.parent is None:
        return False
    siblings = node.parent.elements()
    if pseudo == "first-child":
        return siblings[0] is node
    if pseudo == "last-child":
        return siblings[-1] is node
    if pseudo == "nth-child":
        wanted = int(t["arg"])
        return len(siblings) >= wanted and siblings[wanted - 1] is node
    raise ValueError("unsupported pseudo-class: %r" % pseudo)
