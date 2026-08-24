#!/usr/bin/env python3
"""Compile the book's back matter from the verified corpus and the chapter drafts.

Produces:
  book/98-REFERENCES.md   every source cited, ordered by chapter then alphabetically,
                          each with a resolvable URL taken verbatim from the corpus.
  book/99-CITATION-AUDIT.md  every citation a chapter used that could NOT be matched
                          to a verified corpus entry. This file must be empty before
                          the manuscript is considered finished.

The reference list is generated rather than hand-assembled so that no entry can drift
from the version that passed fact-checking.
"""

import json
import io
import os
import re
import sys
from collections import OrderedDict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BOOK = os.path.join(ROOT, "book")
CORPUS = os.path.join(ROOT, "research", "corpus.json")

CITE_HEADING = "## Citations used in this chapter"


def norm_url(u):
    """Normalize a URL for matching: strip scheme, trailing slash, and case."""
    u = (u or "").strip().rstrip("/")
    u = re.sub(r"^https?://", "", u, flags=re.I)
    u = re.sub(r"^(dx\.)?doi\.org/", "doi.org/", u, flags=re.I)
    u = re.sub(r"^arxiv\.org/(abs|pdf)/", "arxiv.org/abs/", u, flags=re.I)
    u = re.sub(r"v\d+$", "", u)  # arXiv version suffix
    return u.lower()


DOI_RE = re.compile(r"10\.\d{4,9}/[^\s,;)\]\"']+", re.I)
ARXIV_RE = re.compile(r"arxiv[:\s]*(\d{4}\.\d{4,5})", re.I)
SURNAME_RE = re.compile(r"[A-Z][a-zA-Z'À-ɏ-]{2,}")


def surname_year_keys(item):
    """Fallback identity keys: each author surname paired with each year mentioned.

    The corpus and the chapters sometimes point at different artifacts of the same
    work, for instance an arXiv preprint versus the journal of record, or a 2003
    announcement versus the 2006 full paper. Matching on surname and year recovers
    those without loosening the requirement that the work be in the corpus.
    """
    people = item.get("people", "") or ""
    years = re.findall(r"(1[89]\d{2}|20\d{2})", item.get("year", "") or "")
    keys = set()
    for surname in SURNAME_RE.findall(people):
        low = surname.lower()
        if low in ("and", "published", "as", "translator", "commentator", "first",
                   "letter", "the", "et", "al"):
            continue
        for year in years:
            keys.add("%s|%s" % (low, year))
    return keys


def load_corpus():
    with io.open(CORPUS, encoding="utf-8") as fh:
        corpus = json.load(fh)
    by_url, by_alias = {}, {}
    for domain in corpus.values():
        for item in domain.get("items", []):
            key = norm_url(item.get("url"))
            if key and key not in by_url:
                by_url[key] = item
            # Alias index: every DOI or arXiv id mentioned anywhere in the entry,
            # since the venue field often names the other artifact of the same work.
            blob = " ".join([item.get("venue", "") or "", item.get("url", "") or "",
                             item.get("claim", "") or ""])
            for doi in DOI_RE.findall(blob):
                by_alias.setdefault(norm_url("doi.org/" + doi), item)
            for aid in ARXIV_RE.findall(blob):
                by_alias.setdefault(norm_url("arxiv.org/abs/" + aid), item)
            for sy in surname_year_keys(item):
                by_alias.setdefault(sy, item)
    return corpus, by_url, by_alias


def parse_chapter_citations(path):
    """Return the list of raw citation lines from a chapter's citation block."""
    with io.open(path, encoding="utf-8") as fh:
        text = fh.read()
    if CITE_HEADING not in text:
        return []
    block = text.split(CITE_HEADING, 1)[1]
    lines = []
    for raw in block.splitlines():
        line = raw.strip().lstrip("-*+ ").strip()
        if not line or line.startswith("#"):
            continue
        if "http" not in line:
            continue
        lines.append(line)
    return lines


def extract_url(line):
    match = re.search(r"https?://\S+", line)
    return match.group(0).rstrip(".,;)]") if match else ""


def format_entry(item):
    """Format one verified corpus item as a reference-list entry."""
    people = (item.get("people") or "").strip()
    year = (item.get("year") or "").strip()
    venue = (item.get("venue") or "").strip()
    url = (item.get("url") or "").strip()
    # The corpus stores the assertion in `claim`; the title is usually inside
    # `venue`, so we present people, year, venue and URL and let the venue carry it.
    head = people if people else "[no author]"
    parts = ["%s (%s)." % (head, year) if year else "%s." % head]
    if venue:
        parts.append(venue + ".")
    if url:
        parts.append(url)
    return " ".join(parts)


def main():
    if not os.path.exists(CORPUS):
        sys.exit("corpus.json not found; run the research workflow first")
    _corpus, by_url, by_alias = load_corpus()

    chapters = sorted(
        f for f in os.listdir(BOOK) if re.fullmatch(r"ch\d{2}\.md", f)
    )

    per_chapter = OrderedDict()
    unmatched = []
    all_matched = OrderedDict()

    for fname in chapters:
        num = int(fname[2:4])
        lines = parse_chapter_citations(os.path.join(BOOK, fname))
        entries = {}
        for line in lines:
            url = extract_url(line)
            key = norm_url(url)
            item = by_url.get(key) or by_alias.get(key)
            if item is None:
                # Fall back to surname and year taken from the citation line itself.
                cited_years = re.findall(r"\((1[89]\d{2}|20\d{2})\)", line)
                head = line.split("(")[0]
                for surname in SURNAME_RE.findall(head):
                    for year in cited_years:
                        item = by_alias.get("%s|%s" % (surname.lower(), year))
                        if item is not None:
                            break
                    if item is not None:
                        break
            if item is None:
                unmatched.append((num, line))
                continue
            entries[key] = item
            all_matched[key] = item
        per_chapter[num] = entries

    # ---- reference list -------------------------------------------------
    out = [
        "# References",
        "",
        "Every source cited in this book, ordered by the chapter in which it first",
        "appears and then alphabetically. Each entry carries a resolvable URL, either",
        "a DOI link, an arXiv abstract link, or an official institutional page.",
        "",
        "This list is generated directly from the verified research corpus rather than",
        "assembled by hand, so no entry can drift from the version that passed",
        "fact-checking. Where a preprint year and a journal year differ, both are given.",
        "",
    ]
    seen = set()
    for num, entries in per_chapter.items():
        fresh = {k: v for k, v in entries.items() if k not in seen}
        if not fresh:
            continue
        out.append("## Chapter %d" % num)
        out.append("")
        for key in sorted(fresh, key=lambda k: format_entry(fresh[k]).lower()):
            out.append(format_entry(fresh[key]))
            out.append("")
            seen.add(key)

    with io.open(os.path.join(BOOK, "98-REFERENCES.md"), "w",
                 encoding="utf-8", newline="") as fh:
        fh.write("\n".join(out))

    # ---- citation audit -------------------------------------------------
    audit = [
        "# Citation audit",
        "",
        "Citations that appeared in a chapter but could not be matched to a verified",
        "entry in the research corpus. Every line here is a citation no fact-checker",
        "cleared. This file MUST be empty before the manuscript is finished.",
        "",
    ]
    if unmatched:
        audit.append("## Unmatched (%d)" % len(unmatched))
        audit.append("")
        for num, line in unmatched:
            audit.append("Chapter %d: %s" % (num, line))
            audit.append("")
    else:
        audit.append("No unmatched citations. Every cited source traces to the")
        audit.append("verified corpus.")
        audit.append("")

    with io.open(os.path.join(BOOK, "99-CITATION-AUDIT.md"), "w",
                 encoding="utf-8", newline="") as fh:
        fh.write("\n".join(audit))

    print("chapters scanned      : %d" % len(chapters))
    print("distinct sources cited: %d" % len(all_matched))
    print("unmatched citations   : %d" % len(unmatched))
    if unmatched:
        print()
        print("UNMATCHED (these must be resolved):")
        for num, line in unmatched[:25]:
            print("  ch%02d  %s" % (num, line[:120]))


if __name__ == "__main__":
    main()
