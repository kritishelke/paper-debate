from __future__ import annotations

import base64
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

import httpx

from backend.llm.claude_client import ClaudeClient
from backend.pipeline.models import PaperStructure


CACHE_DIR = Path(os.getenv("PAPERDEBATE_CACHE", ".cache/ingest"))
SECTION_ALIASES = {
    "abstract": "intro",
    "introduction": "intro",
    "background": "related_work",
    "related work": "related_work",
    "method": "methods",
    "methods": "methods",
    "experiment": "results",
    "experiments": "results",
    "results": "results",
    "discussion": "limitations",
    "limitations": "limitations",
    "conclusion": "limitations",
}


async def ingest_input(
    *,
    pdf_bytes: str | None = None,
    abstract: str | None = None,
    focus: str = "",
    claude_client: ClaudeClient | None = None,
) -> PaperStructure:
    if pdf_bytes:
        raw = await extract_pdf_text(base64.b64decode(pdf_bytes))
    elif abstract:
        raw = abstract
    else:
        raise ValueError("Either pdf_bytes or abstract is required.")
    return await extract_structure(raw, focus=focus, claude_client=claude_client)


async def extract_pdf_text(pdf_bytes: bytes) -> str:
    grobid_url = os.getenv("GROBID_URL", "http://localhost:8070").rstrip("/")
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(
                f"{grobid_url}/api/processFulltextDocument",
                files={"input": ("paper.pdf", pdf_bytes, "application/pdf")},
            )
        if response.status_code == 200 and response.text.strip():
            sections = parse_tei_sections(response.text)
            if sections:
                return "\n\n".join(f"{name}\n{text}" for name, text in sections.items())
    except Exception:
        pass
    return extract_pdf_text_pymupdf(pdf_bytes)


def parse_tei_sections(tei_xml: str) -> dict[str, str]:
    try:
        root = ElementTree.fromstring(tei_xml)
    except ElementTree.ParseError:
        return {}
    ns = {"tei": "http://www.tei-c.org/ns/1.0"}
    sections: dict[str, list[str]] = {}
    for div in root.findall(".//tei:text//tei:body//tei:div", ns):
        head = div.find("tei:head", ns)
        name = normalize_section(head.text if head is not None else "body")
        text = " ".join(t.strip() for t in div.itertext() if t and t.strip())
        if text:
            sections.setdefault(name, []).append(text)
    return {name: "\n".join(parts) for name, parts in sections.items()}


def extract_pdf_text_pymupdf(pdf_bytes: bytes) -> str:
    try:
        import fitz

        with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
            return "\n".join(page.get_text("text") for page in doc)
    except Exception:
        return pdf_bytes.decode("utf-8", errors="ignore")


async def extract_structure(
    paper_text: str,
    *,
    focus: str = "",
    claude_client: ClaudeClient | None = None,
) -> PaperStructure:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(paper_text.encode("utf-8")).hexdigest()
    cache_path = CACHE_DIR / f"{digest}.json"
    if cache_path.exists():
        return PaperStructure(**json.loads(cache_path.read_text()))

    client = claude_client or ClaudeClient()
    system = (
        "You are a scientific paper analyst. Extract structured information from the "
        "following paper text. Return only valid JSON, nothing else."
    )
    user = (
        f"{paper_text}\n\nFocus question: {focus}\n\n"
        "Extract and return JSON with keys:\n"
        "  claims       - list of assertion strings (max 30)\n"
        "  methods      - list of methodology descriptions (max 30)\n"
        "  results      - list of findings (max 30)\n"
        "  assumptions  - list of implicit or explicit assumptions (max 30)\n"
        "  section_tags - parallel list mapping each claim to its source section: "
        "intro / methods / results / limitations / related_work"
    )
    try:
        parsed: dict[str, Any] = json.loads(await client.complete_json(system, user))
        structure = PaperStructure(
            claims=clean_list(parsed.get("claims"))[:30],
            methods=clean_list(parsed.get("methods"))[:30],
            results=clean_list(parsed.get("results"))[:30],
            assumptions=clean_list(parsed.get("assumptions"))[:30],
            section_tags=clean_list(parsed.get("section_tags"))[:30],
            title=infer_title(paper_text),
            abstract=paper_text[:1500],
            raw_text=paper_text,
        )
    except Exception:
        structure = heuristic_structure(paper_text)
    if len(structure.section_tags) < len(structure.claims):
        structure.section_tags.extend(["intro"] * (len(structure.claims) - len(structure.section_tags)))
    cache_path.write_text(json.dumps(structure.__dict__, indent=2))
    return structure


def heuristic_structure(text: str) -> PaperStructure:
    sentences = split_sentences(text)
    claims = [s for s in sentences if looks_like_claim(s)][:30] or sentences[:5]
    methods = [s for s in sentences if re.search(r"\b(method|model|train|evaluate|simulate|measure)\b", s, re.I)][:30]
    results = [s for s in sentences if re.search(r"\b(result|improve|outperform|find|show|demonstrate)\b", s, re.I)][:30]
    assumptions = [s for s in sentences if re.search(r"\b(assume|requires|depends|under the condition)\b", s, re.I)][:30]
    return PaperStructure(
        claims=claims,
        methods=methods or ["Method details were not explicitly extracted."],
        results=results or ["Result details were not explicitly extracted."],
        assumptions=assumptions or ["The analysis assumes the abstract faithfully summarizes the paper."],
        section_tags=[guess_section(c) for c in claims],
        title=infer_title(text),
        abstract=text[:1500],
        raw_text=text,
    )


def clean_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def split_sentences(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+", text.replace("\n", " ")) if len(part.strip()) > 20]


def looks_like_claim(sentence: str) -> bool:
    return bool(re.search(r"\b(we|this paper|our|the method|results?|model|framework)\b", sentence, re.I))


def infer_title(text: str) -> str:
    for line in text.splitlines():
        line = line.strip()
        if 8 <= len(line) <= 180 and not line.lower().startswith(("abstract", "introduction")):
            return line
    return "Untitled paper"


def normalize_section(section: str | None) -> str:
    if not section:
        return "intro"
    lowered = section.strip().lower()
    for key, value in SECTION_ALIASES.items():
        if key in lowered:
            return value
    return "intro"


def guess_section(sentence: str) -> str:
    return normalize_section(sentence)

