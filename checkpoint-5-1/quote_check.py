r"""Quote checker for the capstone — fixed at Checkpoint 5.1.

The question it answers: does each quotation in an answer appear on the page the answer
cites for it?

The 3.1 checker (`metrics_for` in capstone_checkpoint_3_1_evaluation.py) got this wrong in
five ways, all found by re-checking its failures by hand at 3.1 and 4.1:

  1. Ellipsis. "A ... B" is normal quoting, but a substring match looks for the dots.
  2. Bracket edits. "pass[es] weak checks" — the page says "passes weak checks".
  3. Spacing. PDF extraction leaves stray or missing spaces ("state-of-the- art").
  4. Wrong citation credited. It accepted a quote found on ANY page cited anywhere in the
     answer, or anywhere in the retrieved text, not the page cited for that quote.
  5. Grouped citations. "[A p.2; B p.1]" was not recognised as a citation at all.

This file fixes all five and leaves the 3.1 file untouched (it is submitted and frozen).
Fixes 1-3 make the checker fairer; fixes 4-5 make it stricter.

Run:  python quote_check.py --validate     re-checks the 3.1 answers, old vs new
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "checkpoint-2-1"))
sys.path.insert(0, str(HERE.parent / "checkpoint-3-1"))
import capstone_checkpoint_3_1_evaluation as ev  # noqa: E402  - page_text, normalize, MISATTRIBUTED

# ─── Finding quotes and citations ─────────────────────────────────────
QUOTE_RE = ev.QUOTE_RE                                   # same quote finder as 3.1: 25+ chars in quotes
BRACKET_RE = re.compile(r"\[([^\[\]]*?\bp\.?\s*\d+[^\[\]]*)\]")   # any [...] holding a "p.N"
ONE_CITE_RE = re.compile(r"([A-Za-z0-9_.\-]+)\s+p\.?\s*(\d+)")    # one "source p.N" inside it
ELLIPSIS_RE = re.compile(r"\.\s?\.\s?\.|…")
EDIT_RE = re.compile(r"\[[^\]]*\]")                      # a bracket edit inside a quote
MIN_PIECE_WORDS = 3                                      # shorter pieces are too common to prove anything


BARE_PAGE_RE = re.compile(r"^\s*p\.?\s*(\d+)\s*$")          # "p.6" with no source name


def citation_groups(answer: str) -> list[tuple[int, list[tuple[str, int]]]]:
    """Every citation bracket in the answer: (position, [(source, page), ...]).
    Handles "[A p.2]", grouped "[A p.2; B p.1]", and shorthand "[A p.1; p.2]" where a
    bare page reuses the last source named (fix 5)."""
    groups = []
    for m in BRACKET_RE.finditer(answer):
        cites, last = [], None
        for item in re.split(r"[;,]", m.group(1)):
            full = ONE_CITE_RE.search(item)
            bare = BARE_PAGE_RE.match(item)
            if full:
                last = full.group(1)
                cites.append((last, int(full.group(2))))
            elif bare and last:
                cites.append((last, int(bare.group(1))))
        if cites:
            groups.append((m.start(), cites))
    return groups


def all_citations(answer: str) -> list[tuple[str, int]]:
    return [c for _, cites in citation_groups(answer) for c in cites]


def attached_citations(answer: str, quote_end: int, quote_start: int) -> list[tuple[str, int]]:
    """The citation that belongs to one quote (fix 4).

    Rule: the first citation bracket after the quote, in the same paragraph. If there is
    none, the last one before the quote in the same paragraph ("[A p.3] says "...").
    Returns [] when the quote has no citation of its own."""
    para_end = answer.find("\n\n", quote_end)
    para_end = len(answer) if para_end == -1 else para_end
    para_start = answer.rfind("\n\n", 0, quote_start)
    para_start = 0 if para_start == -1 else para_start
    groups = citation_groups(answer)
    after = [cites for pos, cites in groups if quote_end <= pos < para_end]
    if after:
        return after[0]
    before = [cites for pos, cites in groups if para_start <= pos < quote_start]
    return before[-1] if before else []


# ─── Matching a quote against a page ──────────────────────────────────
def squash(text: str) -> str:
    """3.1's normalisation, then drop every space, so stray or missing spaces from the
    PDF no longer matter (fix 3)."""
    return ev.normalize(text).replace(" ", "")


def pieces(quote: str) -> list[str]:
    """Split a quote at ellipses and bracket edits (fixes 1 and 2). Each piece must be on
    the page, in order. Pieces under 3 words are dropped: "pass" proves nothing. If
    nothing is left, the whole quote with the brackets removed is used instead."""
    parts = [p for p in re.split(f"{ELLIPSIS_RE.pattern}|{EDIT_RE.pattern}", quote)]
    kept = [squash(p) for p in parts if len(ev.normalize(p).split()) >= MIN_PIECE_WORDS]
    if not kept:
        kept = [squash(EDIT_RE.sub("", ELLIPSIS_RE.sub(" ", quote)))]
    return [k for k in kept if k]


def on_page(quote: str, page: str) -> bool:
    """True when every piece of the quote appears on the page, in the quoted order."""
    text = squash(page)
    pos = 0
    for piece in pieces(quote):
        at = text.find(piece, pos)
        if at == -1:
            return False
        pos = at + len(piece)
    return True


def paper_pages(source: str) -> dict[int, str]:
    f = ev.TEXT_CACHE / f"{source}.txt"
    if not f.exists():
        return {}
    parts = re.split(r"\f?PAGE (\d+)\n", f.read_text(encoding="utf-8", errors="replace"))
    return {int(parts[i]): parts[i + 1] for i in range(1, len(parts) - 1, 2)}


# ─── The check ────────────────────────────────────────────────────────
def check_quote(answer: str, m: re.Match, retrieved_text: str = "") -> dict:
    """Outcome for one quote. Only "own_page" counts as verified.

      own_page         on the page cited for it                      verified
      other_cited_page on a page cited elsewhere in the answer        3.1 counted this
      wrong_page       elsewhere in the paper it cites                 model error
      in_context_only  in the retrieved text, not on any cited page   3.1 counted this
      not_found        none of the above
      no_citation      the quote has no citation of its own; checked against every page cited
    """
    quote = m.group(1)
    own = attached_citations(answer, m.end(), m.start())
    if own:
        if any(on_page(quote, ev.page_text(s, p)) for s, p in own):
            return {"quote": quote, "cited": own, "outcome": "own_page", "verified": True}
    others = [c for c in all_citations(answer) if c not in own]
    if not own:
        ok = any(on_page(quote, ev.page_text(s, p)) for s, p in others)
        return {"quote": quote, "cited": [], "outcome": "no_citation", "verified": ok}
    if any(on_page(quote, ev.page_text(s, p)) for s, p in others):
        return {"quote": quote, "cited": own, "outcome": "other_cited_page", "verified": False}
    if any(on_page(quote, t) for s, _ in own for t in paper_pages(s).values()):
        return {"quote": quote, "cited": own, "outcome": "wrong_page", "verified": False}
    if retrieved_text and on_page(quote, retrieved_text):
        return {"quote": quote, "cited": own, "outcome": "in_context_only", "verified": False}
    return {"quote": quote, "cited": own, "outcome": "not_found", "verified": False}


def check_answer(answer: str, retrieved_text: str = "") -> dict:
    """Quote verification for a whole answer, plus the per-quote detail."""
    detail = [check_quote(answer, m, retrieved_text) for m in QUOTE_RE.finditer(answer)]
    cited = [s for s, _ in all_citations(answer)]
    return {
        "n_quotes": len(detail),
        "quote_verification": (sum(d["verified"] for d in detail) / len(detail)) if detail else None,
        "n_citations": len(cited),
        "coverage": len(set(cited)),                      # distinct files cited, groups included
        "source_verification": (sum(s not in ev.MISATTRIBUTED for s in cited) / len(cited)) if cited else None,
        "cited_misattributed": ";".join(sorted({s for s in cited if s in ev.MISATTRIBUTED})),
        "quotes": detail,
    }


# ─── Validation against the 3.1 answers ───────────────────────────────
def old_check(rec: dict) -> dict:
    """The 3.1 checker, called as-is, for side-by-side comparison."""
    case = {"grading_notes": "", "expected": ""}
    m = ev.metrics_for(rec, case)
    return {"quote_verification": m["quote_verification"], "n_quotes": m["n_quotes"]}


def validate() -> None:
    """Re-check the 45 original 3.1 questions, run 1, old checker vs new. The 3.1 hand
    analysis of 2026-09-15 says what a correct checker should find."""
    answers = json.loads((HERE.parent / "checkpoint-3-1" / "checkpoint_3_1_answers.json").read_text())
    rows = list(csv.DictReader((HERE.parent / "checkpoint-3-1" / "checkpoint_3_1_results.csv").open()))
    passed = {r["id"] for r in rows if r["run"] == "1" and not r["variant_of"]
              and r["judge_prompt"] == "balanced" and r["verdict"] == "pass"}
    recs = [r for r in answers.values() if r["run"] == 1 and not r["variant_of"]]
    tally, clean_old, clean_new, flips = {}, [], [], []
    for rec in recs:
        ctx = " ".join(x["text"] for x in rec["retrieved"])
        new = check_answer(rec["answer"], ctx)
        old = old_check(rec)
        for d in new["quotes"]:
            tally[d["outcome"]] = tally.get(d["outcome"], 0) + 1
        if rec["id"] in passed:
            ok_src = ev.metrics_for(rec, {"grading_notes": "", "expected": ""})["source_verification"]
            if old["quote_verification"] == 1.0 and ok_src == 1.0:
                clean_old.append(rec["id"])
            if new["quote_verification"] == 1.0 and new["source_verification"] == 1.0:
                clean_new.append(rec["id"])
        if old["quote_verification"] != new["quote_verification"]:
            flips.append((rec["id"], old["quote_verification"], new["quote_verification"]))
    n_q = sum(tally.values())
    print(f"45 originals, run 1: {len(recs)} answers, {n_q} quotes")
    for k in ("own_page", "no_citation", "other_cited_page", "wrong_page", "in_context_only", "not_found"):
        print(f"  {k:<17} {tally.get(k, 0)}")
    print(f"\nPassed (balanced judge): {len(passed)}")
    print(f"  clean, old checker: {len(clean_old)}  {sorted(clean_old)}")
    print(f"  clean, new checker: {len(clean_new)}  {sorted(clean_new)}")
    print(f"  now clean: {sorted(set(clean_new) - set(clean_old))}")
    print(f"  no longer clean: {sorted(set(clean_old) - set(clean_new))}")
    print(f"\nAnswers whose quote score changed: {len(flips)}")
    for i, o, n in sorted(flips):
        print(f"  {i:<5} old {o}  new {n}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", action="store_true")
    if ap.parse_args().validate:
        validate()
