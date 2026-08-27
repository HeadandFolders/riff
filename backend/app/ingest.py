"""Ingest: turn a link or a dropped PDF into a prepared paper.

Fetching, sectioning and persistence are tools, not agents — none of them needs
its own reasoning loop. Gemini is used for exactly one judgement here: where the
sections begin and what each one assumes you already know.

The model never echoes the paper back. It returns section boundaries plus a
short verbatim anchor per section, and the prose is sliced locally against those
anchors. That keeps ingest cheap on a paper of any length, and guarantees the
reader sees the author's words rather than a paraphrase of them.
"""

from __future__ import annotations

import re
from typing import Iterable

from pydantic import BaseModel, Field

from . import alphaxiv, gemini
from .schemas import Paper, PrepStatus, SectionBody, SectionRef
from .store import GraphStore, new_id

ARXIV_ID = re.compile(r"(\d{4}\.\d{4,5})(v\d+)?")

OUTLINE_SYSTEM = """\
You map a research paper's structure so it can be read one section at a time.

Return the sections a reader would actually work through: abstract, the numbered \
body sections, and any discussion, limitations, or future work. Merge trivially \
short subsections into their parent. Skip references, acknowledgements, and \
appendices unless an appendix carries a method the body depends on.

For each section:

- `starts_with` must be the first 8-15 words of that section's body text, copied \
  verbatim from the source, exactly as they appear including punctuation. It is \
  used to locate the section in the raw text, so an approximation is useless. Do \
  not include the section heading itself.
- `concept_labels` are the things the section assumes the reader already \
  understands, not what it teaches. Name them as a knowledgeable reader would \
  ("softmax saturation", "layer normalisation"), lowercase, 2-6 concepts per \
  section, most load-bearing first. Omit anything the section itself defines \
  from first principles.

`future_work_number` is the number of the section stating what remains to be \
done. Papers bury this in a conclusion; pick that section if there is no \
dedicated one, and leave it null only if the paper genuinely says nothing about \
future work.
"""


class OutlineSection(BaseModel):
    number: str = Field(description="Section number as printed, e.g. '3' or '3.2'")
    title: str
    starts_with: str = Field(description="First 8-15 words of the body, verbatim")
    concept_labels: list[str] = Field(default_factory=list)


class PaperOutline(BaseModel):
    title: str
    authors: list[str] = Field(default_factory=list)
    published: str | None = Field(
        default=None, description="Publication date as YYYY-MM-DD if stated"
    )
    repo_url: str | None = Field(
        default=None, description="Code repository URL if the paper gives one"
    )
    future_work_number: str | None = None
    sections: list[OutlineSection] = Field(default_factory=list)


def canonical_arxiv_url(reference: str) -> str:
    """Accept an id, an abs link, or a pdf link; return the abs URL.

    alphaXiv resolves all three, but storing one canonical form keeps a paper
    from being ingested twice under two spellings of the same reference.
    """
    match = ARXIV_ID.search(reference.strip())
    if not match:
        raise ValueError(f"no arXiv id found in {reference!r}")
    return f"https://arxiv.org/abs/{match.group(1)}"


def _normalised_with_offsets(text: str) -> tuple[str, list[int]]:
    """Collapse whitespace and lowercase, keeping a map back to real offsets.

    Extracted PDF text breaks lines mid-sentence, so an anchor only matches once
    both sides are normalised — but the slice has to be taken from the original.
    """
    chars: list[str] = []
    offsets: list[int] = []
    in_space = False
    for index, char in enumerate(text):
        if char.isspace():
            if in_space:
                continue
            chars.append(" ")
            offsets.append(index)
            in_space = True
        else:
            chars.append(char.lower())
            offsets.append(index)
            in_space = False
    return "".join(chars), offsets


def _anchor_positions(raw: str, anchors: Iterable[str]) -> list[int | None]:
    """Locate each anchor in order, never searching behind the previous match."""
    haystack, offsets = _normalised_with_offsets(raw)
    positions: list[int | None] = []
    cursor = 0
    for anchor in anchors:
        needle, _ = _normalised_with_offsets(anchor)
        needle = needle.strip()
        found = haystack.find(needle, cursor) if needle else -1
        if found == -1 and needle:
            # Long anchors pick up transcription noise; the opening words are
            # usually still intact, so retry with a shorter prefix.
            words = needle.split()
            while len(words) > 4 and found == -1:
                words = words[:-1]
                found = haystack.find(" ".join(words), cursor)
        if found == -1:
            positions.append(None)
            continue
        positions.append(offsets[found])
        cursor = found + 1
    return positions


def outline_from_text(raw: str) -> PaperOutline:
    return gemini.structured(
        [
            "Map the structure of this paper.",
            f"Raw extracted text:\n{raw}",
        ],
        PaperOutline,
        system_instruction=OUTLINE_SYSTEM,
    )


def outline_from_pdf(pdf: bytes) -> PaperOutline:
    """Gemini reads the PDF directly; there is no text-extraction layer."""
    return gemini.structured(
        [
            "Map the structure of this paper.",
            gemini.pdf_part(pdf),
        ],
        PaperOutline,
        system_instruction=OUTLINE_SYSTEM,
    )


def slice_sections(
    paper_id: str, raw: str, outline: PaperOutline
) -> list[SectionBody]:
    """Cut the raw text at the anchors. Sections that fail to anchor are dropped.

    A section with no locatable text would render as an empty reading pane, which
    is worse than it not being offered at all.
    """
    positions = _anchor_positions(raw, [s.starts_with for s in outline.sections])
    bodies: list[SectionBody] = []

    for index, (section, start) in enumerate(zip(outline.sections, positions)):
        if start is None:
            continue
        end = len(raw)
        for later in positions[index + 1 :]:
            if later is not None and later > start:
                end = later
                break
        text = raw[start:end].strip()
        if not text:
            continue
        bodies.append(
            SectionBody(
                id=_section_id(section.number, index),
                paper_id=paper_id,
                number=section.number,
                title=section.title,
                text=text,
            )
        )
    return bodies


def _section_id(number: str, index: int) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", number.lower()).strip("-")
    return f"sec-{slug or index + 1}"


def _persist(
    store: GraphStore,
    paper: Paper,
    bodies: list[SectionBody],
    outline: PaperOutline,
) -> Paper:
    kept = {body.id for body in bodies}
    paper.section_index = [
        SectionRef(
            id=_section_id(section.number, index),
            number=section.number,
            title=section.title,
            concept_labels=section.concept_labels,
        )
        for index, section in enumerate(outline.sections)
        if _section_id(section.number, index) in kept
    ]

    if outline.future_work_number:
        target = _section_id(outline.future_work_number, 0)
        paper.future_work_section_id = target if target in kept else None
    if paper.future_work_section_id is None and paper.section_index:
        # Every paper must end in a hypothesis, so fall back to the last section
        # rather than leaving the gate with nothing to interrogate.
        paper.future_work_section_id = paper.section_index[-1].id

    paper.prep_status = PrepStatus.READY
    paper.prep_error = None
    store.save_paper(paper)
    for body in bodies:
        store.save_section_body(body)

    labels = [l for ref in paper.section_index for l in ref.concept_labels]
    store.register_paper_concepts(paper.id, labels)
    store.recompute_frontier([paper.id])
    return store.get_paper(paper.id) or paper


def stub_paper(
    store: GraphStore, source: str, source_ref: str, title: str
) -> Paper:
    """Record the paper as pending before any slow work starts.

    Preparation costs a retrieval round trip and two Gemini calls, far longer
    than a request should block for, so the caller gets an id to poll instead.
    """
    paper = Paper(
        id=new_id("paper"),
        title=title,
        source=source,  # type: ignore[arg-type]
        source_ref=source_ref,
        prep_status=PrepStatus.PENDING,
    )
    store.save_paper(paper)
    return paper


def prepare_arxiv(store: GraphStore, paper_id: str) -> Paper:
    """Raw text comes from alphaXiv, structure from Gemini."""
    paper = _claim(store, paper_id)
    try:
        raw = alphaxiv.paper_content(paper.source_ref, full_text=True)
        outline = outline_from_text(raw)
        bodies = slice_sections(paper.id, raw, outline)
        if not bodies:
            raise ValueError("no sections could be located in the extracted text")
    except Exception as exc:
        store.set_prep_status(paper.id, PrepStatus.FAILED, str(exc)[:500])
        raise
    return _persist(store, _with_metadata(paper, outline), bodies, outline)


def prepare_pdf(store: GraphStore, paper_id: str, pdf: bytes) -> Paper:
    """Gemini maps the structure and transcribes the prose.

    The outline's anchors need text to be sliced against, and there is no
    extraction layer, so the transcription is the text. It is paid once per
    paper at ingest, never per session.
    """
    paper = _claim(store, paper_id)
    try:
        outline = outline_from_pdf(pdf)
        raw = _transcribe(pdf)
        bodies = slice_sections(paper.id, raw, outline)
        if not bodies:
            raise ValueError("no sections could be located in the transcription")
    except Exception as exc:
        store.set_prep_status(paper.id, PrepStatus.FAILED, str(exc)[:500])
        raise
    return _persist(store, _with_metadata(paper, outline), bodies, outline)


def _claim(store: GraphStore, paper_id: str) -> Paper:
    paper = store.get_paper(paper_id)
    if paper is None:
        raise ValueError(f"unknown paper {paper_id}")
    store.set_prep_status(paper_id, PrepStatus.PREPARING)
    return paper


def _with_metadata(paper: Paper, outline: PaperOutline) -> Paper:
    paper.title = outline.title or paper.title
    paper.authors = outline.authors
    paper.published = outline.published
    paper.repo_url = outline.repo_url
    return paper


class Transcription(BaseModel):
    text: str = Field(description="The paper's body text, verbatim and in order")


def _transcribe(pdf: bytes) -> str:
    result = gemini.structured(
        [
            "Transcribe this paper's body text verbatim, in reading order, from "
            "the abstract to the end of the conclusion. Include section headings "
            "on their own line. Skip figures, tables, page numbers, and "
            "references. Do not summarise or rephrase anything.",
            gemini.pdf_part(pdf),
        ],
        Transcription,
        temperature=0.0,
    )
    return result.text
