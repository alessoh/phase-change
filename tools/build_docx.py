#!/usr/bin/env python3
"""Render the manuscript as Word .docx in the house style of PhaseAug23.docx.

House style, measured from the reference document rather than assumed:
    page        US Letter, 8.5 x 11 in, 1 in margins on all four sides
    body        Garamond 12 pt, single spaced, 18 pt before, 4 pt after,
                first line indented 0.5 in
    Title       48 pt, used once for the book title
    Heading 1   24 pt, used for the author name, Synopsis, the chapter number,
                the chapter title, References and Glossary
    references  Surname (Year). Title. Venue. URL
    equations   real Word equation objects (OMML), not LaTeX source text
    urls        live hyperlinks

Inline $...$ LaTeX in the markdown is converted through
LaTeX -> MathML (latex2mathml) -> OMML (Microsoft's MML2OMML.XSL) so the
equations are editable in Word exactly as they are in the reference document.

Usage:
    python tools/build_docx.py              # per-chapter files + the whole book
    python tools/build_docx.py --chapters 1 2
"""

import argparse
import copy
import io
import json
import os
import re
import sys

import docx
from docx.enum.style import WD_STYLE_TYPE
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Inches, Pt
from lxml import etree

import latex2mathml.converter as l2m

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BOOK = os.path.join(ROOT, "book")
OUTDIR = os.path.join(BOOK, "docx")
CORPUS = os.path.join(ROOT, "research", "corpus.json")

CITE_HEADING = "## Citations used in this chapter"

TEMPLATE = os.path.join(ROOT, "tools", "assets", "house-style-template.docx")

MML2OMML_CANDIDATES = [
    r"C:/Program Files/Microsoft Office/root/Office16/MML2OMML.XSL",
    r"C:/Program Files (x86)/Microsoft Office/root/Office16/MML2OMML.XSL",
    r"C:/Program Files/Microsoft Office/Office16/MML2OMML.XSL",
]

ALSO_BY = [
    "Self-Improving AI",
    "AI Operating Systems",
    "Absolute Zero Reasoner",
    "On the Cusp of SuperIntelligence",
    "AI Predictive Modeling",
    "AI Game Theory",
    "Puzzling AI Trends 2025-26",
    "AI Quantum Field Theory",
    "AI Langlands Program",
    "Tomorrow's Film Studio Today",
]

BOOK_TITLE = "The Discontinuous World"
BOOK_SUBTITLE = "Artificial Intelligence and the Physics of Phase Change"
AUTHOR = "H. Peter Alesso"


# --------------------------------------------------------------------------
# math
# --------------------------------------------------------------------------

class MathConverter:
    """LaTeX -> OMML, with a cache because the same symbols recur constantly."""

    def __init__(self):
        path = next((p for p in MML2OMML_CANDIDATES if os.path.exists(p)), None)
        self.xslt = etree.XSLT(etree.parse(path)) if path else None
        self.cache = {}
        self.failures = []

    @property
    def available(self):
        return self.xslt is not None

    def to_omml(self, tex):
        if tex in self.cache:
            return self.cache[tex]
        result = None
        if self.xslt is not None:
            try:
                mathml = l2m.convert(tex)
                converted = self.xslt(etree.fromstring(mathml.encode("utf-8")))
                node = converted.getroot()
                result = node if node is not None and "oMath" in node.tag else None
            except Exception as exc:  # noqa: BLE001 - fall back to literal text
                self.failures.append((tex, str(exc)[:120]))
                result = None
        self.cache[tex] = result
        return result


# --------------------------------------------------------------------------
# document construction
# --------------------------------------------------------------------------

def _set_black(style):
    """Strip Word's blue theme colour from a heading style.

    The reference document's headings carry no colour at all, so they render
    black. Word's built-in Heading styles default to accent1 blue, which would
    be wrong for an academic monograph.
    """
    rpr = style.element.get_or_add_rPr()
    for existing in rpr.findall(qn("w:color")):
        rpr.remove(existing)
    color = rpr.makeelement(qn("w:color"), {qn("w:val"): "000000"})
    rpr.append(color)


def new_document():
    """Start from the author's own document so styles match exactly.

    Earlier versions rebuilt the styles by hand from measurements and got three
    things wrong that no numeric comparison caught: Normal carries
    <w:contextualSpacing/>, which suppresses the paragraph gap between
    consecutive body paragraphs; Heading 1 is centred and bold; and Word's
    stock Title style ships with blue text and a blue bottom border that must
    be removed. Cloning the template inherits all of it, including the theme.
    """
    doc = docx.Document(TEMPLATE)

    body = doc.element.body
    for child in list(body):
        if child.tag == qn("w:sectPr"):
            continue
        body.remove(child)

    section = doc.sections[0]
    section.page_width = Inches(8.5)
    section.page_height = Inches(11)
    for attr in ("left_margin", "right_margin", "top_margin", "bottom_margin"):
        setattr(section, attr, Inches(1))
    return doc


def _disable_contextual_spacing(paragraph):
    """Turn off <w:contextualSpacing/> for one paragraph.

    Normal inherits contextualSpacing from the template, which cancels
    space-before between adjacent paragraphs of the same style. Section
    headings are Normal-styled, so without this they sit flush against the
    body paragraph above them and any space_before is silently ignored.
    """
    pPr = paragraph._p.get_or_add_pPr()
    for existing in pPr.findall(qn("w:contextualSpacing")):
        pPr.remove(existing)
    node = pPr.makeelement(qn("w:contextualSpacing"), {qn("w:val"): "0"})
    pPr.append(node)


def ensure_section_style(doc):
    """Define the in-chapter section heading style.

    It must NOT be Normal. The template's Normal carries <w:contextualSpacing/>,
    which suppresses spacing between consecutive paragraphs of the same style, so
    a bold Normal paragraph butts straight against the body text above it. A
    distinct style restores the gap while keeping the house typeface and size.
    """
    styles = doc.styles
    try:
        st = styles["BookSection"]
    except KeyError:
        st = styles.add_style("BookSection", WD_STYLE_TYPE.PARAGRAPH)
        st.base_style = styles["Normal"]
    pf = st.paragraph_format
    pf.first_line_indent = Inches(0)
    pf.space_before = Pt(16)
    pf.space_after = Pt(4)
    pf.keep_with_next = True
    # Clear the inherited contextualSpacing so the space_before actually applies.
    pPr = st.element.get_or_add_pPr()
    for tag in ("w:contextualSpacing",):
        for el in pPr.findall(qn(tag)):
            pPr.remove(el)
    st.font.bold = True
    st.font.name = "Garamond"
    st.font.size = Pt(12)
    return st


def add_section_heading(doc, text):
    """Section heading: bold Garamond at body size, flush left, with air above."""
    ensure_section_style(doc)
    para = doc.add_paragraph(style="BookSection")
    para.add_run(text)
    return para


def add_display_math(doc, tex, mathconv):
    """A display equation on its own line: centred, unindented, with air around it.

    contextualSpacing must be cleared on the paragraph or the space above and
    below is silently dropped, because the surrounding body paragraphs share the
    Normal style.
    """
    para = doc.add_paragraph()
    pf = para.paragraph_format
    pf.first_line_indent = Inches(0)
    pf.alignment = WD_ALIGN_PARAGRAPH.CENTER
    pf.space_before = Pt(10)
    pf.space_after = Pt(10)
    _disable_contextual_spacing(para)
    omml = mathconv.to_omml(tex)
    if omml is not None:
        para._p.append(copy.deepcopy(omml))
    else:
        # Never leak raw TeX delimiters onto the page; fall back to the body text.
        para.add_run(tex)
    return para


URL_RE = re.compile(r"https?://[^\s<>\"')\]]+")


def add_hyperlink(paragraph, url, text):
    part = paragraph.part
    r_id = part.relate_to(
        url,
        "http://schemas.openxmlformats.org/officeDocument/2006/relationships/hyperlink",
        is_external=True,
    )
    link = etree.SubElement(paragraph._p, qn("w:hyperlink"))
    link.set(qn("r:id"), r_id)
    run = etree.SubElement(link, qn("w:r"))
    rpr = etree.SubElement(run, qn("w:rPr"))
    style = etree.SubElement(rpr, qn("w:rStyle"))
    style.set(qn("w:val"), "Hyperlink")
    node = etree.SubElement(run, qn("w:t"))
    node.text = text
    node.set(qn("xml:space"), "preserve")


def add_text_run(paragraph, text, italic=False):
    if not text:
        return
    run = paragraph.add_run(text)
    run.italic = italic


SPLIT_RE = re.compile(r"(\$[^$]+\$)")


def add_rich_paragraph(doc, text, mathconv, style=None):
    """Add a paragraph, converting $...$ to OMML and URLs to hyperlinks."""
    paragraph = doc.add_paragraph(style=style)
    for chunk in SPLIT_RE.split(text):
        if not chunk:
            continue
        if chunk.startswith("$") and chunk.endswith("$") and len(chunk) > 2:
            tex = chunk[1:-1]
            omml = mathconv.to_omml(tex)
            if omml is not None:
                # deepcopy is essential: lxml append() MOVES an element that
                # already has a parent, so reusing a cached node would silently
                # delete every earlier occurrence of the same equation.
                paragraph._p.append(copy.deepcopy(omml))
            else:
                add_text_run(paragraph, tex, italic=True)
            continue
        pos = 0
        for match in URL_RE.finditer(chunk):
            add_text_run(paragraph, chunk[pos:match.start()])
            add_hyperlink(paragraph, match.group(0), match.group(0))
            pos = match.end()
        add_text_run(paragraph, chunk[pos:])
    return paragraph


# --------------------------------------------------------------------------
# markdown parsing
# --------------------------------------------------------------------------

MATH_SPAN_RE = re.compile(r"\$\$.+?\$\$|\$[^$]+?\$")


def _strip_emphasis(text):
    """Strip markdown emphasis and link syntax from ordinary prose."""
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    text = re.sub(r"(?<!\w)\*(?!\s)(.+?)(?<!\s)\*(?!\w)", r"\1", text)
    text = re.sub(r"\[([^\]]+)\]\((https?://[^)]+)\)", r"\1 \2", text)
    return text


def strip_inline_markdown(text):
    """Strip emphasis from prose while leaving math spans untouched.

    The italic pattern would otherwise eat the asterisks in expressions such as
    the renormalization-group fixed point condition, silently corrupting the
    LaTeX so the equation fails to convert to OMML. Math spans are skipped
    rather than substituted through.
    """
    parts = []
    last = 0
    for match in MATH_SPAN_RE.finditer(text):
        parts.append(_strip_emphasis(text[last:match.start()]))
        parts.append(match.group(0))
        last = match.end()
    parts.append(_strip_emphasis(text[last:]))
    return "".join(parts)


def parse_chapter(path):
    with io.open(path, encoding="utf-8") as fh:
        raw = fh.read()
    body, citations = raw, []
    if CITE_HEADING in raw:
        body, tail = raw.split(CITE_HEADING, 1)
        for line in tail.splitlines():
            line = line.strip().lstrip("-*+ ").strip()
            if line and "http" in line and not line.startswith("#"):
                citations.append(line)

    number, title, blocks = None, None, []
    for line in body.splitlines():
        stripped = line.strip()
        if stripped.startswith("# ") and number is None:
            heading = stripped[2:].strip()
            match = re.match(r"Chapter\s+(\d+)\.?\s*(.*)", heading, re.I)
            if match:
                number, title = int(match.group(1)), match.group(2).strip()
            else:
                title = heading
            continue
        if stripped.startswith("## "):
            blocks.append(("h2", stripped[3:].strip()))
            continue
        if not stripped:
            continue
        if stripped.startswith("$$") and stripped.endswith("$$") and len(stripped) > 4:
            blocks.append(("math", stripped[2:-2].strip()))
            continue
        # A paragraph that is nothing but one $...$ equation is display math too.
        # Earlier drafts wrote these with single delimiters and they leaked to the
        # page as literal dollar signs.
        if (stripped.startswith("$") and stripped.endswith("$")
                and len(stripped) > 2 and stripped.count("$") == 2):
            blocks.append(("math", stripped[1:-1].strip()))
            continue
        blocks.append(("p", strip_inline_markdown(stripped)))
    return {"number": number, "title": title, "blocks": blocks, "citations": citations}


def parse_synopsis():
    path = os.path.join(BOOK, "00-SYNOPSIS-AND-OUTLINE.md")
    if not os.path.exists(path):
        return []
    with io.open(path, encoding="utf-8") as fh:
        text = fh.read()
    if "## Synopsis" not in text:
        return []
    body = text.split("## Synopsis", 1)[1].split("\n---", 1)[0]
    return [strip_inline_markdown(l.strip())
            for l in body.splitlines() if l.strip() and not l.strip().startswith("#")]


# --------------------------------------------------------------------------
# emit
# --------------------------------------------------------------------------

def add_front_matter(doc, mathconv):
    para = doc.add_paragraph(BOOK_TITLE, style="Title")
    para.alignment = WD_ALIGN_PARAGRAPH.CENTER
    para = doc.add_paragraph(BOOK_SUBTITLE)
    para.alignment = WD_ALIGN_PARAGRAPH.CENTER
    para.paragraph_format.first_line_indent = Inches(0)
    doc.add_paragraph()

    para = doc.add_paragraph("Also by " + AUTHOR)
    para.paragraph_format.first_line_indent = Inches(0)
    para.alignment = WD_ALIGN_PARAGRAPH.CENTER
    for book in ALSO_BY:
        line = doc.add_paragraph(book)
        line.paragraph_format.first_line_indent = Inches(0)
        line.paragraph_format.space_before = Pt(0)
        line.paragraph_format.space_after = Pt(0)
        line.alignment = WD_ALIGN_PARAGRAPH.CENTER

    doc.add_heading(AUTHOR, level=1)

    synopsis = parse_synopsis()
    if synopsis:
        doc.add_heading("Synopsis", level=1)
        for block in synopsis:
            add_rich_paragraph(doc, block, mathconv)


def add_chapter(doc, chapter, mathconv):
    if chapter["number"] is not None:
        doc.add_heading("Chapter %d" % chapter["number"], level=1)
    doc.add_heading(chapter["title"] or "", level=1)
    for kind, text in chapter["blocks"]:
        if kind == "h2":
            add_section_heading(doc, text)
        elif kind == "math":
            add_display_math(doc, text, mathconv)
        else:
            add_rich_paragraph(doc, text, mathconv)


def add_references(doc, entries, mathconv):
    if not entries:
        return
    doc.add_heading("References", level=1)
    for entry in entries:
        para = add_rich_paragraph(doc, entry, mathconv)
        para.paragraph_format.first_line_indent = Inches(0)
        para.paragraph_format.space_before = Pt(6)
        para.paragraph_format.space_after = Pt(0)


def sort_key(entry):
    return entry.lower()


def add_glossary(doc, path, mathconv):
    """Render the glossary with bold headwords and its section headings intact.

    The generic paragraph path strips ** markers without applying bold, which
    would leave every headword visually identical to its definition, and it
    skips lines beginning with #, which would silently drop the section
    headings. Both matter for a reference section a reader scans rather than
    reads, so the glossary gets its own renderer.
    """
    doc.add_page_break()
    doc.add_heading("Glossary", level=1)
    with io.open(path, encoding="utf-8") as fh:
        lines = fh.read().splitlines()
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("# "):
            continue
        if line.startswith("## "):
            add_section_heading(doc, line[3:].strip())
            continue
        match = re.match(r"\*\*(.+?)\*\*\s*(.*)$", line)
        if match:
            para = doc.add_paragraph()
            para.paragraph_format.first_line_indent = Inches(0)
            para.paragraph_format.space_before = Pt(8)
            _disable_contextual_spacing(para)
            head = para.add_run(match.group(1))
            head.bold = True
            rest = strip_inline_markdown(match.group(2))
            if rest:
                add_text_run(para, " " + rest)
            continue
        para = add_rich_paragraph(doc, strip_inline_markdown(line), mathconv)
        para.paragraph_format.first_line_indent = Inches(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chapters", nargs="*", type=int)
    args = ap.parse_args()

    os.makedirs(OUTDIR, exist_ok=True)
    mathconv = MathConverter()
    if not mathconv.available:
        print("WARNING: MML2OMML.XSL not found; equations fall back to italic text")

    files = sorted(f for f in os.listdir(BOOK) if re.fullmatch(r"ch\d{2}\.md", f))
    if args.chapters:
        want = {"ch%02d.md" % n for n in args.chapters}
        files = [f for f in files if f in want]
    if not files:
        sys.exit("no chapter files found in %s" % BOOK)

    chapters = [parse_chapter(os.path.join(BOOK, f)) for f in files]

    # per-chapter documents
    for chapter in chapters:
        doc = new_document()
        add_chapter(doc, chapter, mathconv)
        add_references(doc, sorted(set(chapter["citations"]), key=sort_key), mathconv)
        name = "Chapter-%02d.docx" % (chapter["number"] or 0)
        doc.save(os.path.join(OUTDIR, name))
        words = sum(len(t.split()) for k, t in chapter["blocks"] if k == "p")
        print("wrote %-22s %5d words  %3d refs" % (name, words, len(set(chapter["citations"]))))

    # whole book
    doc = new_document()
    add_front_matter(doc, mathconv)
    seen, ordered = set(), []
    for chapter in chapters:
        doc.add_page_break()
        add_chapter(doc, chapter, mathconv)
        for entry in chapter["citations"]:
            key = entry.lower()
            if key not in seen:
                seen.add(key)
                ordered.append(entry)
    doc.add_page_break()
    add_references(doc, sorted(ordered, key=sort_key), mathconv)

    glossary = os.path.join(BOOK, "97-GLOSSARY.md")
    if os.path.exists(glossary):
        add_glossary(doc, glossary, mathconv)

    out = os.path.join(OUTDIR, "The-Discontinuous-World.docx")
    doc.save(out)
    print()
    print("wrote %s" % out)
    print("equations converted: %d distinct" % len([v for v in mathconv.cache.values() if v is not None]))
    if mathconv.failures:
        print("equation failures  : %d" % len(mathconv.failures))
        for tex, err in mathconv.failures[:8]:
            print("   %-40s %s" % (tex[:40], err[:60]))


if __name__ == "__main__":
    main()
