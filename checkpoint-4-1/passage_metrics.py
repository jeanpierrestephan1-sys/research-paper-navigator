r"""Checkpoint 4.1 - passage-level retrieval metrics beside the 3.1 paper-level ones.

Checkpoint 3.1 scored retrieval at the PAPER level: did any chunk from the right paper
come back? On that ruler hybrid reached 32/34 at k=6, and 100% on single-paper tasks,
so no retrieval change can show an improvement on it. It also counted Q2 and Q6 as
hits although the text holding the answer never came back and the model refused.

This scores questions at three levels:

  paper    any chunk from a listed paper            (3.1's ruler, reproduced exactly)
  page     a chunk from the page recorded for the question
  passage  a chunk that CONTAINS THE ANSWER - every key fact for its paper, or the
           verbatim sentence for quote questions

"page" is kept to show why it is not used: V1 and V2 got the answer back from a page
other than the recorded one and scored misses, while N4, C1 and C7 got the recorded page
but not the chunk with the answer. "passage" asks what the model needs: was it handed
text that contains the answer?

For cross-paper questions, first-hit counting hides the real problem, so coverage is
reported too: how many of the listed papers have an answer-bearing chunk in the top 6.

Two question sets, scored separately and never mixed:
  frozen45    the 3.1 test set, unchanged (34 have a source). Key facts frozen 09-15.
  multipaper  12 new cross-paper questions written for 4.1 (testInputs_multipaper.json).

Chunks never cross a page boundary (chunk_document in 2.1), so every chunk has one page.
The 2.1 retriever and the 3.1 test set are IMPORTED, not copied.

Run:
    python passage_metrics.py [--set frozen45|multipaper] --validate   # no API calls
    python passage_metrics.py [--set frozen45|multipaper]              # rank (cached) and score

Rankings are cached per set, so rescoring costs nothing. The first run of a set costs one
query embedding per question per retriever side.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CP31 = HERE.parent / "checkpoint-3-1"
sys.path.insert(0, str(HERE.parent / "checkpoint-2-1"))
sys.path.insert(0, str(CP31))
import capstone_checkpoint_2_1_baseline_retrieval as cp21  # noqa: E402
import capstone_checkpoint_3_1_evaluation as ev  # noqa: E402

K_CONTEXT = cp21.NUM_RETRIEVED      # 6 - what the model actually sees
K_POOL = cp21.CANDIDATE_POOL        # 20
DEPTH = 300

SETS = {
    "frozen45": {
        "cases": CP31 / "testInputs.json",
        "facts": HERE / "passage_key_facts.json",
        "rankings": HERE / "rankings_baseline.json",
        "results": HERE / "passage_metrics.json",
    },
    "multipaper": {
        "cases": HERE / "testInputs_multipaper.json",
        "facts": HERE / "passage_key_facts_multipaper.json",
        "rankings": HERE / "rankings_baseline_multipaper.json",
        "results": HERE / "passage_metrics_multipaper.json",
    },
}


def load_cases(cfg: dict) -> list[dict]:
    return [c for c in json.loads(cfg["cases"].read_text()) if ev.case_sources(c)]


def load_facts(cfg: dict) -> dict:
    return {k: v for k, v in json.loads(cfg["facts"].read_text()).items() if not k.startswith("_")}


def answer_bearing(chunk: dict, spec) -> bool:
    """True if this chunk contains the answer as defined for its paper."""
    text = " " + ev.normalize(chunk["text"]) + " "
    if isinstance(spec, dict):                      # quote: verbatim, spaces removed
        return ev.normalize(spec["quote"]).replace(" ", "") in text.replace(" ", "")
    return all(any(re.search(alt, text) for alt in group) for group in spec)


def hit_kind(chunk: dict, case: dict, facts: dict) -> dict:
    wanted = set(ev.case_sources(case))
    spec = facts[case["id"]].get(chunk["source"])
    return {
        "paper": chunk["source"] in {s for s, _ in wanted},
        "page": (chunk["source"], chunk["page"]) in wanted,
        "passage": spec is not None and answer_bearing(chunk, spec),
    }


# ─── validate the key facts BEFORE trusting them ─────────────────────
def validate(cfg: dict) -> None:
    """For every question and paper: which chunks in the WHOLE INDEX are answer-bearing?

    Zero means the key facts are wrong (or the answer straddles a chunk boundary) and no
    retriever could ever score a passage hit. The recorded page must be among them;
    if it is not, either the key facts or the recorded page is wrong.
    """
    facts, retr = load_facts(cfg), ev.get_retriever(verbose=False)
    by_source: dict[str, list[dict]] = {}
    for ch in retr._chunks:
        by_source.setdefault(ch["source"], []).append(ch)
    problems = 0
    print(f"KEY-FACT VALIDATION - answer-bearing chunks in the whole index\n{'=' * 78}")
    for c in load_cases(cfg):
        if c["id"] not in facts:
            print(f"{c['id']:<4} NO KEY FACTS"); problems += 1; continue
        recorded = {s: p for s, p in ev.case_sources(c)}
        missing = set(recorded) - set(facts[c["id"]])
        if missing:
            print(f"{c['id']:<4} NO KEY FACTS FOR {sorted(missing)}   <-- CHECK"); problems += 1
        for src, spec in facts[c["id"]].items():
            chunks = by_source.get(src, [])
            hits = [ch for ch in chunks if answer_bearing(ch, spec)]
            pages = sorted({ch["page"] for ch in hits})
            ok_rec = recorded.get(src) in pages
            flag = "" if hits and ok_rec else "   <-- CHECK"
            problems += bool(flag)
            print(f"{c['id']:<4} {src:<36} {len(hits):>3}/{len(chunks):<4} chunks  "
                  f"pages {pages[:10]}{'...' if len(pages) > 10 else ''}  "
                  f"recorded p{recorded.get(src)} {'in' if ok_rec else 'NOT in'}{flag}")
    print(f"\n{problems} to check")


# ─── rank once, score many times ─────────────────────────────────────
def rank(cfg: dict) -> dict:
    if cfg["rankings"].exists():
        return json.loads(cfg["rankings"].read_text())
    retr = ev.get_retriever(verbose=False)
    out = {}
    for c in load_cases(cfg):
        q = c["question"]
        out[c["id"]] = {
            "bm25": [k for k, _ in retr.bm25_topk(q, DEPTH)],
            "vector": [k for k, _ in retr.vector_topk(q, DEPTH)],
            "hybrid": [cp21.HybridChunkRetriever._key(ch) for ch, _ in retr.getTopK(q, K_POOL)],
        }
    cfg["rankings"].write_text(json.dumps(out))
    return out


def first_ranks(keys: list[str], case: dict, facts: dict, retr) -> dict:
    ranks = {"paper": None, "page": None, "passage": None}
    for n, key in enumerate(keys, 1):
        for level, hit in hit_kind(retr._by_key[key], case, facts).items():
            if hit and ranks[level] is None:
                ranks[level] = n
    return ranks


def papers_covered(keys: list[str], case: dict, facts: dict, retr, k: int) -> tuple[int, int]:
    """Multi-paper coverage: how many listed papers have an answer-bearing chunk in the top k."""
    listed = {s for s, _ in ev.case_sources(case)}
    got = {retr._by_key[key]["source"] for key in keys[:k]
           if hit_kind(retr._by_key[key], case, facts)["passage"]}
    return len(got & listed), len(listed)


def score(rankings: dict, label: str, cfg: dict) -> list[dict]:
    facts, retr, cases = load_facts(cfg), ev.get_retriever(verbose=False), load_cases(cfg)
    rows = []
    for c in cases:
        row = {"id": c["id"], "category": c["category"]}
        for name in ("bm25", "vector", "hybrid"):
            for level, r in first_ranks(rankings[c["id"]][name], c, facts, retr).items():
                row[f"{name}_{level}"] = r
        top6 = rankings[c["id"]]["hybrid"][:K_CONTEXT]
        row["hybrid_top6"] = [f"{retr._by_key[k]['source']}:p{retr._by_key[k]['page']}"
                              f"{'*' if hit_kind(retr._by_key[k], c, facts)['passage'] else ''}"
                              for k in top6]
        if c["category"] == "cross_paper":
            got, listed = papers_covered(rankings[c["id"]]["hybrid"], c, facts, retr, K_CONTEXT)
            row["hybrid_papers_covered@6"] = f"{got}/{listed}"
        rows.append(row)

    n = len(rows)
    print(f"RETRIEVAL METRICS [{label}] - {n} queries with a known source\n")
    print(f"{'retriever':<10}{'level':<9}{'recall@6':>12}{'recall@20':>12}{'MRR':>7}")
    print("-" * 50)
    for name in ("bm25", "vector", "hybrid"):
        for level in ("paper", "page", "passage"):
            rs = [r[f"{name}_{level}"] for r in rows]
            h6 = sum(1 for r in rs if r and r <= K_CONTEXT)
            h20 = sum(1 for r in rs if r and r <= K_POOL)
            mrr = sum(1.0 / r if r else 0.0 for r in rs) / n
            print(f"{name:<10}{level:<9}{h6:>5}/{n} {h6/n:>4.0%}{h20:>5}/{n} {h20/n:>4.0%}{mrr:>7.3f}")
        print()

    print(f"BY CATEGORY (hybrid, @6){'':<2}{'paper':>8}{'page':>8}{'passage':>9}")
    print("-" * 51)
    for cat in sorted({r["category"] for r in rows}):
        sub = [r for r in rows if r["category"] == cat]
        cell = lambda lv: f"{sum(1 for r in sub if r[f'hybrid_{lv}'] and r[f'hybrid_{lv}'] <= K_CONTEXT)}/{len(sub)}"
        print(f"{cat:<26}{cell('paper'):>8}{cell('page'):>8}{cell('passage'):>9}")

    print("\nPASSAGE MISSES AT 6 (hybrid)   * = answer-bearing chunk")
    for r in rows:
        if not (r["hybrid_passage"] and r["hybrid_passage"] <= K_CONTEXT):
            print(f"  {r['id']:<4} paper rank {r['hybrid_paper'] or '>20'}, passage rank {r['hybrid_passage'] or '>20'}")
            print(f"       top 6: {', '.join(x.replace('DBLP_', '') for x in r['hybrid_top6'])}")

    multi = [r for r in rows if "hybrid_papers_covered@6" in r]
    if multi:
        full = sum(1 for r in multi if r["hybrid_papers_covered@6"].split("/")[0] == r["hybrid_papers_covered@6"].split("/")[1])
        got = sum(int(r["hybrid_papers_covered@6"].split("/")[0]) for r in multi)
        listed = sum(int(r["hybrid_papers_covered@6"].split("/")[1]) for r in multi)
        print(f"\nMULTI-PAPER COVERAGE (hybrid @6): listed papers with an answer-bearing chunk")
        for r in multi:
            print(f"  {r['id']:<4} {r['hybrid_papers_covered@6']}")
        print(f"  every listed paper covered: {full}/{len(multi)} questions;  "
              f"papers covered: {got}/{listed}")
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=sorted(SETS), default="frozen45")
    ap.add_argument("--validate", action="store_true")
    args = ap.parse_args()
    cfg = SETS[args.set]
    if args.validate:
        validate(cfg)
        return
    rows = score(rank(cfg), f"baseline: 2.1 hybrid, {args.set}", cfg)

    if args.set == "frozen45":
        old = {r["id"]: r for r in json.loads((CP31 / "retrieval_metrics.json").read_text())}
        drift = [(r["id"], name, old[r["id"]][name], r[f"{name}_paper"])
                 for r in rows for name in ("bm25", "vector", "hybrid")
                 if old[r["id"]][name] != r[f"{name}_paper"]]
        print(f"\nDRIFT vs 3.1 retrieval_metrics.json (paper level): "
              f"{'none - same ruler' if not drift else drift}")
    cfg["results"].write_text(json.dumps(rows, indent=1))
    print(f"per-query ranks -> {cfg['results'].name}")


if __name__ == "__main__":
    main()
