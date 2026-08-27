"""The Assessor: decides what the reader actually understands.

Vector similarity can only say whether the graph holds material near a concept.
It cannot tell the difference between a reader who can derive a result and one
who has merely seen the words. That difference is the entire point, so the
verdict is produced by reasoning over something the reader said.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from .. import gemini
from ..schemas import Misconception, Understanding, UnderstandingVerdict

ASSESSOR_SYSTEM = """\
You assess whether a reader understands a specific concept, based only on an \
explanation they just gave in their own words. You are a diagnostician, not an \
encourager.

Grade the mechanism, not the vocabulary:

- SOLID — they can state the mechanism and say what it depends on. Correct \
  reasoning expressed clumsily is still SOLID.
- PARTIAL — the shape is right but a load-bearing piece is missing or vague. \
  Fluent use of the correct terminology with no mechanism behind it is PARTIAL, \
  never SOLID.
- MISCONCEIVED — they confidently assert something that is wrong in a way that \
  will cause downstream errors. This is worse than knowing nothing, because it \
  will silently corrupt everything built on top of it. Prefer this over PARTIAL \
  whenever a stated belief is actively false.
- ABSENT — they say they do not know, or the answer is unrelated to the concept. \
  Admitting ignorance is ABSENT, not MISCONCEIVED. Never penalise honesty.

Rules:

1. Quote or closely paraphrase the reader's own words in `reasoning`. A verdict \
   that cannot point at what they said is not a verdict.
2. Name misconceptions concretely. "Confuses the estimator with the estimand" is \
   useful; "some confusion about the basics" is not. State the belief in their \
   terms, then say precisely why it fails.
3. Do not teach in `reasoning`. Explaining is another agent's job. You may put \
   the correct account in a misconception's `correction` field only.
4. If prior misconceptions are supplied, check explicitly whether each one \
   recurs in this explanation. Restate a recurring one rather than inventing a \
   new phrasing for it, so it can be tracked across sessions.
5. `followup_question` is the single question that would most cheaply resolve \
   your remaining doubt. Omit it when the verdict is unambiguous.
"""

PROBE_SYSTEM = """\
You write one question that reveals whether a reader understands a concept.

A good probe cannot be answered by pattern-matching the paper's wording. Ask \
for a mechanism, a consequence, a boundary condition, or what would break if an \
assumption failed. Never ask for a definition. Never ask a yes/no question. One \
sentence.
"""


class Probe(BaseModel):
    concept_label: str
    question: str = Field(description="One sentence, asks for mechanism not definition")
    what_a_good_answer_contains: list[str] = Field(default_factory=list)


def probe(concept_label: str, section_title: str, section_excerpt: str) -> Probe:
    """Ask the question whose answer can be graded."""
    return gemini.structured(
        [
            f"Concept: {concept_label}",
            f"Section: {section_title}",
            f"Excerpt:\n{section_excerpt[:4000]}",
        ],
        Probe,
        system_instruction=PROBE_SYSTEM,
        temperature=0.6,
    )


def _prior_block(priors: list[Misconception]) -> str:
    if not priors:
        return "Prior misconceptions on record: none."
    lines = ["Prior misconceptions on record — check whether each recurs:"]
    for item in priors:
        lines.append(
            f"- believed: {item.belief}\n"
            f"  actual: {item.correction}\n"
            f"  seen {item.times_observed}x, severity {item.severity}"
        )
    return "\n".join(lines)


def assess_text(
    concept_label: str,
    explanation: str,
    *,
    section_title: str = "",
    section_excerpt: str = "",
    priors: list[Misconception] | None = None,
) -> UnderstandingVerdict:
    verdict = gemini.structured(
        [
            f"Concept under assessment: {concept_label}",
            f"Section context: {section_title}",
            f"Source excerpt (ground truth):\n{section_excerpt[:6000]}",
            _prior_block(priors or []),
            f"The reader's explanation, verbatim:\n\"\"\"\n{explanation}\n\"\"\"",
        ],
        UnderstandingVerdict,
        system_instruction=ASSESSOR_SYSTEM,
    )
    return _normalise(verdict, concept_label)


def assess_audio(
    concept_label: str,
    audio: bytes,
    *,
    mime_type: str = "audio/webm",
    section_title: str = "",
    section_excerpt: str = "",
    priors: list[Misconception] | None = None,
) -> UnderstandingVerdict:
    """Grade a spoken explanation without a separate transcription step.

    Hesitation, restarts and self-correction carry signal that a transcript
    flattens, and this also avoids depending on browser speech recognition.
    """
    verdict = gemini.structured(
        [
            f"Concept under assessment: {concept_label}",
            f"Section context: {section_title}",
            f"Source excerpt (ground truth):\n{section_excerpt[:6000]}",
            _prior_block(priors or []),
            "The reader's spoken explanation follows. Judge the content, not "
            "the fluency; ignore filler and false starts.",
            gemini.audio_part(audio, mime_type),
        ],
        UnderstandingVerdict,
        system_instruction=ASSESSOR_SYSTEM,
    )
    return _normalise(verdict, concept_label)


def _normalise(
    verdict: UnderstandingVerdict, concept_label: str
) -> UnderstandingVerdict:
    verdict.concept_label = concept_label
    # A misconception was found, so the verdict cannot be a pass regardless of
    # what the model labelled it.
    if verdict.misconceptions and verdict.level in (
        Understanding.SOLID,
        Understanding.UNASSESSED,
    ):
        verdict.level = Understanding.MISCONCEIVED
    for detected in verdict.misconceptions:
        detected.concept_label = detected.concept_label or concept_label
    return verdict
