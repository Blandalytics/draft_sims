"""Enough of readxl to read one spreadsheet a year.

WalterFootball is the odd source out: it does not publish a page to scrape,
it publishes a workbook, and the R reads it with `readxl::read_xlsx`. An
.xlsx is a zip of XML, both of which the standard library already opens, so
this reads the one sheet asked for rather than adding openpyxl to a project
whose dependencies are numpy, scipy and pyarrow.

What it handles is what that workbook uses: shared strings, inline strings,
numbers, and gaps -- a row that skips column D leaves an empty cell there
rather than shifting everything left. Dates come back as the number the file
stores, which is fine here because nothing in these sheets is a date.
"""

import re
import zipfile
from xml.etree import ElementTree

MAIN = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
RELS = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"
DOC_RELS = (
    "{http://schemas.openxmlformats.org/package/2006/relationships}"
)

_CELL_REF = re.compile(r"([A-Z]+)(\d+)")


def sheet_names(path):
    """The sheets in the workbook, in the order the file lists them."""
    with zipfile.ZipFile(path) as book:
        workbook = ElementTree.fromstring(book.read("xl/workbook.xml"))
        sheets = workbook.find(MAIN + "sheets")
        return [s.get("name") for s in sheets]


def read_sheet(path, name):
    """One sheet as rows of strings, the first row being its headers."""
    with zipfile.ZipFile(path) as book:
        target = _sheet_path(book, name)
        strings = _shared_strings(book)
        return _rows(book.read(target), strings)


def to_records(rows):
    """Rows with a header row on top, as a list of dicts.

    Blank headers are dropped, which is what `read_xlsx` does with the
    spacer columns these sheets are full of, and a trailing row with
    nothing in it is dropped too.
    """
    if not rows:
        return []
    headers = rows[0]
    keep = [i for i, h in enumerate(headers) if h.strip()]
    out = []
    for row in rows[1:]:
        record = {headers[i]: (row[i] if i < len(row) else "") for i in keep}
        if any(v.strip() for v in record.values()):
            out.append(record)
    return out


def _sheet_path(book, name):
    workbook = ElementTree.fromstring(book.read("xl/workbook.xml"))
    sheet = None
    for element in workbook.find(MAIN + "sheets"):
        if element.get("name") == name:
            sheet = element
    if sheet is None:
        raise KeyError("no sheet named %r" % name)
    rels = ElementTree.fromstring(book.read("xl/_rels/workbook.xml.rels"))
    wanted = sheet.get(RELS + "id")
    for rel in rels:
        if rel.get("Id") == wanted:
            return "xl/" + rel.get("Target").lstrip("/")
    raise KeyError("sheet %r has no file" % name)


def _shared_strings(book):
    if "xl/sharedStrings.xml" not in book.namelist():
        return []
    table = ElementTree.fromstring(book.read("xl/sharedStrings.xml"))
    return ["".join(item.itertext()) for item in table]


def _rows(xml, strings):
    sheet = ElementTree.fromstring(xml)
    data = sheet.find(MAIN + "sheetData")
    out = []
    for row in data or []:
        cells, column = [], 0
        for cell in row:
            column = _pad(cells, cell, column)
            cells.append(_value(cell, strings))
            column += 1
        out.append(cells)
    width = max([len(r) for r in out], default=0)
    return [r + [""] * (width - len(r)) for r in out]


def _pad(cells, cell, column):
    """Put empty cells where the file skipped a column."""
    match = _CELL_REF.match(cell.get("r") or "")
    if match:
        wanted = _column(match.group(1))
        cells.extend([""] * max(0, wanted - column))
        return max(column, wanted)
    return column


def _column(letters):
    """A1-style letters as a zero-based column number."""
    number = 0
    for letter in letters:
        number = number * 26 + (ord(letter) - 64)
    return number - 1


def _value(cell, strings):
    kind = cell.get("t")
    if kind == "s":
        index = cell.findtext(MAIN + "v")
        return strings[int(index)] if index is not None else ""
    if kind == "inlineStr":
        node = cell.find(MAIN + "is")
        return "".join(node.itertext()) if node is not None else ""
    return cell.findtext(MAIN + "v") or ""
