r"""Capstone Checkpoint 4.1 — Advanced Retrieval Implementation.
Jupytext-style cell markers (# %% / # %% [markdown]) — runnable as a
plain script AND openable as cells in VS Code/PyCharm/Jupytext.

Completed from capstone_checkpoint_4_1_advanced_retrieval_starter.py. The starter's toy
demo is kept as given (run with --demo). Step 4b applies the technique to the real system.

THE PROBLEM (worksheet Steps 1-2)
A question that needs several papers gets one search and one list of six chunks. The paper
that best matches the question's words fills the list. M7 ("How does Stitch's approach to
library learning differ from DreamCoder's?") got six Stitch chunks and none from DreamCoder.

THE STRATEGY (worksheet Step 3): split, then share
  1. split   the model breaks the question into 2-4 sub-queries (starter's prompt, unchanged)
  2. search  each sub-query runs through the 2.1 hybrid retriever, unchanged - top 20 each
  3. turns   sub-queries take turns adding their best remaining chunk to the six
  4. share   no paper may hold more than 3 of the 6
The model still gets six chunks, the same as the baseline.

TWO VERSIONS, both reported
  v1  the limit of 3 per paper always applies. First run: cross-paper coverage rose, but
      4 single-paper questions (S1, S3, Q3, V2) lost their answer chunk to the limit.
  v2  the limit is lifted when every sub-query's best chunk comes from the same paper -
      the searches agree the question is about one paper. Added after seeing v1's
      results, and scored on the same questions, so it is a second try, not a clean test.

Why take turns instead of the starter's summed scores: in Lab 4.1, summing min-max scores
produced exact ties at 0.500, and the sort then gave one sub-query no slots at all.
Taking turns means every sub-query gets at least one chunk.

COST AND CACHING
Sub-queries and each sub-query's top 20 are saved per set (subqueries_advanced_<set>.json).
The first run of a set costs one LLM call per question plus one query embedding per
sub-query. Every later run is free and gives identical results.

Scoring uses passage_metrics.py and the frozen key facts, unchanged. The script refuses to
score if a frozen file has changed.

Run:
    python capstone_checkpoint_4_1_advanced_retrieval.py --demo              # starter demo, 2 LLM calls
    python capstone_checkpoint_4_1_advanced_retrieval.py --set multipaper    # 12 new cross-paper questions
    python capstone_checkpoint_4_1_advanced_retrieval.py --set frozen45      # the 34 sourced 3.1 questions
    python capstone_checkpoint_4_1_advanced_retrieval.py --set multipaper --show M9   # readable evidence
    python capstone_checkpoint_4_1_advanced_retrieval.py --answer    # 69 answers + 11 splits, saved
    python capstone_checkpoint_4_1_advanced_retrieval.py --judge     # 138 judge calls on saved answers
    python capstone_checkpoint_4_1_advanced_retrieval.py --report    # Step 5 answer tables, free
"""

# %% [markdown]
# # Capstone Checkpoint 4.1 — Advanced Retrieval Implementation
# **MO-LLM Module 4/Required Capstone Checkpoint (120 minutes)**
#
# Starter text for Steps 1-3 and 5-6 is unchanged apart from where noted.

# %%
from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import argparse
import csv
import hashlib
import json
import os
import re
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import networkx as nx
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import passage_metrics as pm  # noqa: E402  - imports the 2.1 retriever and 3.1 harness

cp21, ev = pm.cp21, pm.ev

# %%
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
LLM_MODEL = "openai/gpt-5.4-mini"  # Latest small OpenAI model, fast; covered by course credits
TEMPERATURE = 0.2
LOG_PATH = HERE / "checkpoint_4_1_responses.log"   # starter: Path.cwd()

# === SET THIS to the scenario you chose in Checkpoint 1.1 ===
SCENARIO = "research_papers"   # "research_papers" or "wikipedia"

DECOMPOSE_SYSTEM = (
    "You are a query decomposition assistant for a document retrieval system. "
    "Break the user's question into 2-4 focused sub-queries that together cover "
    "everything needed to answer it. Each sub-query should target a distinct aspect. "
    'Return ONLY a JSON array of strings, e.g., ["sub-query 1", "sub-query 2"].'
)

ANSWER_SYSTEM = (
    "You are a helpful assistant. Answer the question using only the provided "
    "documents, and quote from them where you can. If the documents do not contain "
    "the answer, say so rather than guessing."
)


# %%
def check_api_key() -> str:
    load_dotenv(HERE.parent.parent / ".env")   # added: the key lives at the project root
    load_dotenv()
    key = os.getenv("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError(
            "OPENROUTER_API_KEY not set. Grab a free key at "
            "https://openrouter.ai/keys, put it in a .env file next to this "
            "script, and rerun."
        )
    return key


def make_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=LLM_MODEL,
        temperature=TEMPERATURE,
        api_key=check_api_key(),
        base_url=OPENROUTER_BASE_URL,
    )


def log_response(label: str, prompt: str, response: str) -> None:
    ts = datetime.now().isoformat(timespec="seconds")
    entry = (
        f"[{ts}]  {label}  SCENARIO={SCENARIO}  MODEL={LLM_MODEL}\n"
        f"PROMPT:   {prompt}\n"
        f"RESPONSE: {response}\n"
        f"{'-' * 72}\n"
    )
    with LOG_PATH.open("a", encoding="utf-8") as fh:
        fh.write(entry)


# %% [markdown]
# ## A tiny sample corpus (stands in for your real one)

# %%
SAMPLE_DOCS = [
    {"id": "d1", "text": "Paper on retrieval-augmented generation: Grounding LLM answers in retrieved documents reduces hallucination.",
     "topics": ["RAG", "hallucination"], "links": ["d2", "d3"]},
    {"id": "d2", "text": "Study of hallucination in language models: Models fabricate specifics when they lack grounding.",
     "topics": ["hallucination"], "links": ["d1"]},
    {"id": "d3", "text": "Hybrid retrieval combines keyword and vector search to improve recall over either alone.",
     "topics": ["RAG", "retrieval"], "links": ["d1", "d4"]},
    {"id": "d4", "text": "Query decomposition breaks a complex question into sub-queries, improving multi-aspect retrieval.",
     "topics": ["retrieval", "decomposition"], "links": ["d3"]},
    {"id": "d5", "text": "Graph-based retrieval traverses relationships between documents to add related context.",
     "topics": ["retrieval", "graph"], "links": ["d4", "d6"]},
    {"id": "d6", "text": "Evaluation of RAG systems uses an LLM judge to score answers against grading notes.",
     "topics": ["evaluation", "RAG"], "links": ["d5"]},
]
DOC_BY_ID = {d["id"]: d for d in SAMPLE_DOCS}


def keyword_score(query: str, text: str) -> float:
    """A tiny, dependency-free relevance score: shared word count."""
    q = set(re.findall(r"[a-z0-9]+", query.lower()))
    t = set(re.findall(r"[a-z0-9]+", text.lower()))
    return float(len(q & t))


def baseline_retrieve(query: str, k: int = 3) -> list[tuple[str, float]]:
    """Single-pass retrieval: Score every doc once, take the top k. (id, score)."""
    scored = [(d["id"], keyword_score(query, d["text"])) for d in SAMPLE_DOCS]
    scored.sort(key=lambda x: x[1], reverse=True)
    return [(i, s) for i, s in scored[:k] if s > 0]


# %% [markdown]
# ## Step 2 — Multistep retrieval (provided, adapted from Lab 4.1)

# %%
def decompose_query(llm: ChatOpenAI, query: str) -> list[str]:
    messages = [SystemMessage(content=DECOMPOSE_SYSTEM), HumanMessage(content=query)]
    raw = llm.invoke(messages).content.strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        raw = raw[raw.find("["): raw.rfind("]") + 1] if "[" in raw else raw
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list) and all(isinstance(x, str) for x in parsed) and parsed:
            return parsed
    except (json.JSONDecodeError, ValueError):
        pass
    return [query]


def multistep_retrieve(llm: ChatOpenAI, query: str, k: int = 3) -> list[str]:
    sub_queries = decompose_query(llm, query)
    print(f"  decomposed into: {sub_queries}")
    score_map: dict[str, float] = {}
    for sq in sub_queries:
        for doc_id, score in baseline_retrieve(sq, 3 * k):
            score_map[doc_id] = score_map.get(doc_id, 0.0) + score
    ranked = sorted(score_map.items(), key=lambda kv: kv[1], reverse=True)
    return [doc_id for doc_id, _ in ranked[:k]]


# %% [markdown]
# ## Step 3 — Graph-based retrieval (provided, adapted from Lab 4.2)

# %%
def build_graph(docs: list[dict]) -> nx.DiGraph:
    G = nx.DiGraph()
    for d in docs:
        G.add_node(f"doc:{d['id']}", node_type="doc")
        for other in d["links"]:
            G.add_edge(f"doc:{d['id']}", f"doc:{other}", edge_type="links_to")
        for topic in d["topics"]:
            G.add_node(f"topic:{topic}", node_type="topic")
            G.add_edge(f"doc:{d['id']}", f"topic:{topic}", edge_type="relates_to")
    return G


def graph_retrieve(graph: nx.DiGraph, query: str, k: int = 3) -> dict[str, str]:
    """Return {doc_id: source_label}: baseline seeds plus their graph neighbors."""
    union: dict[str, str] = {}
    for doc_id, _ in baseline_retrieve(query, k):
        union[doc_id] = "seed"
    for seed in list(union):
        node = f"doc:{seed}"
        if not graph.has_node(node):
            continue
        for _, target, ed in graph.out_edges(node, data=True):
            if ed.get("edge_type") == "links_to":
                union.setdefault(target.removeprefix("doc:"), "linked")
            elif ed.get("edge_type") == "relates_to":
                for sib, _, ed2 in graph.in_edges(target, data=True):
                    if ed2.get("edge_type") == "relates_to":
                        union.setdefault(sib.removeprefix("doc:"), "topic")
    return union


# %% [markdown]
# ## Step 4 — Your advanced-retrieval plan

# %%
def my_advanced_plan() -> dict[str, Any]:
    return {
        "technique": "decomposition",
        "node_types": [],   # graph only - not used
        "edge_types": [],   # graph only - not used
        "test_queries": [
            # C2 and C8 from the 3.1 set: a needed paper never reached the six.
            "What different strategies do these papers use to combine language models "
            "with formal verification or symbolic reasoning?",
            "Where do these papers disagree about whether language models genuinely "
            "understand the code they produce?",
        ],
        "rationale": (
            "Questions that need several papers get one search, and one paper fills the "
            "six chunks. Splitting gives each part its own search, and a limit of three "
            "chunks per paper leaves room for the others."
        ),
    }


# %% [markdown]
# ## Step 4b — Split, then share, on the real system
#
# The 2.1 retriever is imported and called unchanged. Only what happens around it is new:
# split the question, search once per sub-query, then fill the six by taking turns.

# %%
K = cp21.NUM_RETRIEVED          # 6 - what the model sees, same as the baseline
POOL = cp21.CANDIDATE_POOL      # 20 - what each sub-query's search returns
PER_PAPER_CAP = 3               # set before any run; not tuned on results

FROZEN_MD5 = {
    "passage_key_facts.json": "aabdf037d7152afc41e2a21b39ba942b",
    "passage_key_facts_multipaper.json": "c49d35344ac4b6af54b6254e6d987743",
    "testInputs_multipaper.json": "034a5150cdf2d09ac0800e9d464a9f14",
}


def check_frozen() -> None:
    for name, want in FROZEN_MD5.items():
        got = hashlib.md5((HERE / name).read_bytes()).hexdigest()
        if got != want:
            raise SystemExit(f"{name} has changed (md5 {got}, frozen {want}). "
                             "Scores would not be comparable with the baseline.")


def build_cache(setname: str, cfg: dict, cases: list[dict] | None = None) -> dict:
    """Split each question and save every sub-query's top 20. Paid once per question."""
    path = HERE / f"subqueries_advanced_{setname}.json"
    cache = json.loads(path.read_text()) if path.exists() else {}
    cases = pm.load_cases(cfg) if cases is None else cases
    todo = [c for c in cases if cache.get(c["id"], {}).get("question") != c["question"]]
    if not todo:
        return cache
    llm, retr = make_llm(), ev.get_retriever(verbose=False)
    print(f"Splitting {len(todo)} questions: {len(todo)} LLM calls, one embedding per sub-query\n")
    for c in todo:
        subs = decompose_query(llm, c["question"])
        cache[c["id"]] = {
            "question": c["question"],
            "sub_queries": subs,
            "split_failed": subs == [c["question"]],
            "pools": [[cp21.HybridChunkRetriever._key(ch) for ch, _ in retr.getTopK(sq, POOL)]
                      for sq in subs],
        }
        log_response(f"DECOMPOSE {c['id']}", c["question"], json.dumps(subs))
        path.write_text(json.dumps(cache, indent=1))   # save as we go
        print(f"  {c['id']:<4} {subs}")
    return cache


def take_turns(pools: list[list[str]], paper_of, k: int = K, cap: int = PER_PAPER_CAP,
               with_origin: bool = False) -> list:
    """Sub-queries take turns adding their best remaining chunk to the list.

    A chunk is skipped if it is already in the list, or if its paper already holds `cap`
    chunks. Stops at k chunks, or when every sub-query's pool is used up.
    with_origin=True returns (chunk key, sub-query number) pairs instead of keys.
    """
    chosen: list[str] = []
    origin: list[int] = []
    per_paper: Counter = Counter()
    pos = [0] * len(pools)
    while len(chosen) < k:
        added = False
        for i, pool in enumerate(pools):
            while pos[i] < len(pool):
                key = pool[pos[i]]
                pos[i] += 1
                if key in chosen or per_paper[paper_of(key)] >= cap:
                    continue
                chosen.append(key)
                origin.append(i + 1)
                per_paper[paper_of(key)] += 1
                added = True
                break
            if len(chosen) == k:
                break
        if not added:
            break
    return list(zip(chosen, origin)) if with_origin else chosen


# %%
def label(key: str, case: dict, facts: dict, retr) -> str:
    ch = retr._by_key[key]
    star = "*" if pm.hit_kind(ch, case, facts)["passage"] else ""
    return f"{ch['source'].replace('DBLP_', '')}:p{ch['page']}{star}"


def score_six(keys: list[str], case: dict, facts: dict, retr) -> dict:
    ranks = pm.first_ranks(keys, case, facts, retr)
    row = {"paper": ranks["paper"], "passage": ranks["passage"],
           "top6": [label(k, case, facts, retr) for k in keys]}
    if case["category"] == "cross_paper":
        row["covered"] = list(pm.papers_covered(keys, case, facts, retr, K))
    return row


DESIGNS = ("v1", "v2")   # v1 = PER_PAPER_CAP always, kept for comparison; v2 = limit_for()


def limit_for(pools: list[list[str]], paper_of) -> int:
    """No paper may hold more than PER_PAPER_CAP of the six, unless every search's best
    chunk is from the same paper. Then the searches agree the question is about one paper,
    and the limit is lifted."""
    if len({paper_of(pool[0]) for pool in pools if pool}) <= 1:
        return K
    return PER_PAPER_CAP


def compare(setname: str, cfg: dict, cache: dict) -> None:
    facts, retr, cases = pm.load_facts(cfg), ev.get_retriever(verbose=False), pm.load_cases(cfg)
    baseline = json.loads(cfg["rankings"].read_text())
    paper_of = lambda key: retr._by_key[key]["source"]  # noqa: E731
    sides = ("baseline",) + DESIGNS

    rows = []
    for c in cases:
        pools = cache[c["id"]]["pools"]
        row = {"id": c["id"], "category": c["category"],
               "sub_queries": cache[c["id"]]["sub_queries"],
               "split_failed": cache[c["id"]]["split_failed"],
               "baseline": score_six(baseline[c["id"]]["hybrid"][:K], c, facts, retr)}
        for d in DESIGNS:
            cap = PER_PAPER_CAP if d == "v1" else limit_for(pools, paper_of)
            keys = take_turns(pools, paper_of, cap=cap)
            if len(keys) < K:
                print(f"  WARNING {c['id']} {d}: only {len(keys)} chunks - pools ran out under the cap")
            row[d] = score_six(keys, c, facts, retr) | {"cap": cap}
        rows.append(row)

    n = len(rows)
    hit = lambda r, side, lv: r[side][lv] is not None  # noqa: E731
    cells = lambda f: "".join(f"{f(s):>12}" for s in sides)  # noqa: E731
    print(f"\n{'=' * 80}\nBASELINE (2.1 hybrid, top 6) vs SPLIT, THEN SHARE - {setname}, {n} questions")
    print("Same six-chunk budget. * = chunk holds the answer.")
    print("v1 = limit of 3 per paper always.  v2 = limit lifted when every search's best "
          "chunk is from one paper.\n")
    print(f"{'metric':<44}{cells(lambda s: s)}")
    print("-" * 80)
    for lv, name in (("paper", "right paper in the six"), ("passage", "answer chunk in the six")):
        print(f"{name:<44}{cells(lambda s: f'{sum(hit(r, s, lv) for r in rows)}/{n}')}")
    mrr = lambda s: f"{sum(1 / r[s]['passage'] for r in rows if r[s]['passage']) / n:.3f}"  # noqa: E731
    print(f"{'answer chunk MRR within the six':<44}{cells(mrr)}")

    multi = [r for r in rows if "covered" in r["baseline"]]
    if multi:
        full = lambda s: f"{sum(r[s]['covered'][0] == r[s]['covered'][1] for r in multi)}/{len(multi)}"  # noqa: E731
        papers = lambda s: (f"{sum(r[s]['covered'][0] for r in multi)}/"  # noqa: E731
                            f"{sum(r[s]['covered'][1] for r in multi)}")
        print(f"{'cross-paper: every needed paper in the six':<44}{cells(full)}")
        print(f"{'cross-paper: needed papers in the six':<44}{cells(papers)}")

    lifted = sum(r["v2"]["cap"] == K for r in rows)
    print(f"\nv2 lifted the limit on {lifted} of {n} questions")

    if len({r["category"] for r in rows}) > 1:
        print(f"\n{'by category (answer chunk in the six)':<44}{cells(lambda s: s)}")
        for cat in sorted({r["category"] for r in rows}):
            sub = [r for r in rows if r["category"] == cat]
            count = lambda s, sub=sub: f"{sum(hit(r, s, 'passage') for r in sub)}/{len(sub)}"  # noqa: E731
            print(f"  {cat:<42}{cells(count)}")

    print("\nCHANGES vs baseline (answer chunk in the six, or cross-paper coverage)")
    for r in rows:
        b = r["baseline"]
        tags = []
        for d in DESIGNS:
            a = r[d]
            if hit(r, d, "passage") and not hit(r, "baseline", "passage"):
                tags.append(f"{d} GAINED")
            elif hit(r, "baseline", "passage") and not hit(r, d, "passage"):
                tags.append(f"{d} LOST")
            if "covered" in a and a["covered"] != b["covered"]:
                tags.append(f"{d} papers {b['covered'][0]}/{b['covered'][1]} -> {a['covered'][0]}/{a['covered'][1]}")
        if not tags and not r["split_failed"]:
            continue
        lift = "  (v2 lifted the limit)" if r["v2"]["cap"] == K else ""
        failed = "  (SPLIT FAILED - searched the whole question)" if r["split_failed"] else ""
        print(f"\n  {r['id']:<4} {'; '.join(tags) or 'no change'}{lift}{failed}")
        print(f"       sub-queries: {r['sub_queries']}")
        print(f"       baseline: {', '.join(b['top6'])}")
        print(f"       v1:       {', '.join(r['v1']['top6'])}")
        if r["v2"]["top6"] != r["v1"]["top6"]:
            print(f"       v2:       {', '.join(r['v2']['top6'])}")

    out = HERE / f"advanced_metrics_{setname}.json"
    out.write_text(json.dumps(rows, indent=1))
    print(f"\nper-question results -> {out.name}")


def show(setname: str, cfg: dict, cache: dict, qid: str) -> None:
    """Readable before/after for one question - worksheet evidence. Free: uses saved searches."""
    facts, retr = pm.load_facts(cfg), ev.get_retriever(verbose=False)
    case = next(c for c in pm.load_cases(cfg) if c["id"] == qid)
    paper_of = lambda key: retr._by_key[key]["source"]  # noqa: E731
    pools = cache[qid]["pools"]
    cap = limit_for(pools, paper_of)
    before = [(k, None) for k in json.loads(cfg["rankings"].read_text())[qid]["hybrid"][:K]]
    after = take_turns(pools, paper_of, cap=cap, with_origin=True)

    def title(key: str) -> str:
        t = " ".join(retr._by_key[key]["title"].split())
        return t if len(t) <= 60 else t[:57] + "..."

    print(f"\nQUESTION {qid}: {case['question']}")
    print("\nSEARCHES WRITTEN BY THE MODEL")
    for i, sq in enumerate(cache[qid]["sub_queries"], 1):
        print(f"  {i}. {sq}")
    print(f"\nLimit per paper: {'none - every search put the same paper first' if cap == K else cap}")
    for name, picks in (("BEFORE - one search, top six", before), ("AFTER - split, then share", after)):
        print(f"\n{name}")
        for n, (key, src) in enumerate(picks, 1):
            ch = retr._by_key[key]
            holds = "  <- holds the answer" if pm.hit_kind(ch, case, facts)["passage"] else ""
            frm = f"search {src}: " if src else ""
            print(f"  {n}. {frm}{title(key)} (p.{ch['page']}){holds}")
        if case["category"] == "cross_paper":
            got, listed = pm.papers_covered([k for k, _ in picks], case, facts, retr, K)
            print(f"  Needed papers with answer text: {got} of {listed}")


# %% [markdown]
# ## Step 4c — Answers, judged with the 3.1 harness unchanged
#
# The answer prompt (2.1 `answer()`), both grading prompts, the judge model and every
# deterministic metric (`metrics_for`) are imported from 3.1, not copied. Only the six
# chunks differ. Answers are generated once and saved; judging never regenerates them.
#
# What gets answered (one run each):
#   multipaper  baseline + updated   - no baseline answers existed for M1-M12
#   frozen45    updated only         - the baseline is 3.1's saved run 1 and run 2, same
#                                      retriever, prompt and model, so 3.1's two runs also
#                                      give the noise floor
# 69 answers, 11 extra splits (frozen45 questions with no source were never split), and
# 138 judge calls.

# %%
ANSWERS_41 = HERE / "checkpoint_4_1_answers.json"
RESULTS_41 = HERE / "checkpoint_4_1_results.csv"
JOBS = (("multipaper", "baseline"), ("multipaper", "updated"), ("frozen45", "updated"))


def all_cases(setname: str) -> list[dict]:
    """Every question in a set, including the 11 frozen45 questions with no source."""
    if setname == "frozen45":
        return ev.load_cases()
    return json.loads(pm.SETS[setname]["cases"].read_text())


def context_from(chunks: list[dict]) -> str:
    """Same format as HybridChunkRetriever.context() in 2.1, so the model sees the same layout."""
    blocks = [f"[{c['source']} p.{c['page']}] ({c['title'][:70]})\n{c['text']}" for c in chunks]
    return "\n\n---\n\n".join(blocks)


def generate_answers() -> None:
    retr, llm = ev.get_retriever(verbose=False), cp21.make_llm()
    paper_of = lambda key: retr._by_key[key]["source"]  # noqa: E731
    saved = json.loads(ANSWERS_41.read_text()) if ANSWERS_41.exists() else {}
    for setname, system in JOBS:
        cases = all_cases(setname)
        cache = build_cache(setname, pm.SETS[setname], cases) if system == "updated" else {}
        print(f"\nAnswering {setname} with the {system} system ({len(cases)} questions)")
        for c in cases:
            key = f"{c['id']}|{system}|run1"
            if key in saved:
                continue
            if system == "baseline":
                context, hits = retr.context(c["question"], K)
            else:
                pools = cache[c["id"]]["pools"]
                picks = take_turns(pools, paper_of, cap=limit_for(pools, paper_of))
                hits = [(retr._by_key[k], None) for k in picks]
                context = context_from([h[0] for h in hits])
            ans = cp21.answer(llm, c["question"], context)
            saved[key] = {
                "id": c["id"], "run": 1, "system": system, "set": setname,
                "question": c["question"], "category": c["category"], "variant_of": None,
                "answer": ans,
                "retrieved": [{"source": h[0]["source"], "page": h[0]["page"],
                               "score": h[1], "text": h[0]["text"]} for h in hits],
                "ts": datetime.now().isoformat(timespec="seconds"),
            }
            ANSWERS_41.write_text(json.dumps(saved, indent=1))   # save as we go
            print(f"  {c['id']:<4} {len(ans):>5} chars  {len({h[0]['source'] for h in hits})} papers")
    print(f"\n{len(saved)} answers saved -> {ANSWERS_41.name}")


def judge_answers() -> None:
    saved = json.loads(ANSWERS_41.read_text())
    cases = {c["id"]: c for s, _ in JOBS for c in all_cases(s)}
    rows = list(csv.DictReader(RESULTS_41.open())) if RESULTS_41.exists() else []
    done = {(r["id"], r["system"], r["judge_prompt"]) for r in rows}
    judge = ev.make_judge()
    for rec in saved.values():
        case = cases[rec["id"]]
        for pname, ptext in ev.JUDGE_PROMPTS.items():
            if (rec["id"], rec["system"], pname) in done:
                continue
            # Same message and verdict parsing as 3.1 judge_saved()
            msg = [SystemMessage(content="Reply with exactly one word: pass or fail."),
                   HumanMessage(content=ptext.format(response=rec["answer"],
                                                     grading_notes=case["grading_notes"]))]
            verdict = "pass" if "pass" in judge.invoke(msg).content.strip().lower() else "fail"
            rows.append({"system": rec["system"], "set": rec["set"], **ev.metrics_for(rec, case),
                         "judge_prompt": pname, "judge_model": ev.JUDGE_MODEL, "verdict": verdict})
            print(f"  {rec['id']:<4} {rec['system']:<8} {pname:<9} {verdict}")
            with RESULTS_41.open("w", newline="") as fh:        # save as we go
                w = csv.DictWriter(fh, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows(rows)
    print(f"\n{len(rows)} rows -> {RESULTS_41.name}")


def needed_cited(answer: str, case: dict) -> tuple[int, int]:
    """How many of a question's listed papers the answer cites at least once."""
    cited = {s for s, _ in ev.CITE_RE.findall(answer)}
    listed = {s for s, _ in ev.case_sources(case)}
    return len(cited & listed), len(listed)


def report_answers() -> None:
    rows = list(csv.DictReader(RESULTS_41.open()))
    old = [r for r in csv.DictReader((ev.RESULTS).open()) if not r["variant_of"]]
    new_ans = json.loads(ANSWERS_41.read_text())
    old_ans = json.loads(ev.ANSWERS.read_text())
    cases = {c["id"]: c for s, _ in JOBS for c in all_cases(s)}

    def verdicts(src, prompt, **match):
        return {r["id"]: r["verdict"] for r in src if r["judge_prompt"] == prompt
                and all(r[k] == v for k, v in match.items())}

    def quotes(src, **match):
        sub = [r for r in src if r["judge_prompt"] == "balanced" and all(r[k] == v for k, v in match.items())]
        n = sum(int(r["n_quotes"]) for r in sub)
        ok = sum(round(float(r["quote_verification"]) * int(r["n_quotes"]))
                 for r in sub if r["quote_verification"] not in ("", "None"))
        return f"{ok}/{n}"

    def cited(answers: dict, ids: list[str], suffix: str) -> tuple[str, str]:
        pairs = [needed_cited(answers[f"{i}{suffix}"]["answer"], cases[i]) for i in ids]
        return (f"{sum(a == b for a, b in pairs)}/{len(pairs)}",
                f"{sum(a for a, _ in pairs)}/{sum(b for _, b in pairs)}")

    pc = lambda v: f"{sum(x == 'pass' for x in v.values())}/{len(v)}"  # noqa: E731

    print(f"\n{'=' * 80}\nANSWERS - 12 NEW CROSS-PAPER QUESTIONS (one run each)\n{'=' * 80}")
    m_ids = [c["id"] for c in all_cases("multipaper")]
    print(f"{'metric':<46}{'baseline':>12}{'updated':>12}")
    for prompt in ("balanced", "strict"):
        b = verdicts(rows, prompt, set="multipaper", system="baseline")
        u = verdicts(rows, prompt, set="multipaper", system="updated")
        print(f"{'grader pass, ' + prompt:<46}{pc(b):>12}{pc(u):>12}")
    bq, uq = cited(new_ans, m_ids, "|baseline|run1"), cited(new_ans, m_ids, "|updated|run1")
    print(f"{'answer cites every needed paper':<46}{bq[0]:>12}{uq[0]:>12}")
    print(f"{'needed papers cited':<46}{bq[1]:>12}{uq[1]:>12}")
    print(f"{'quotes found on the cited page':<46}"
          f"{quotes(rows, set='multipaper', system='baseline'):>12}{quotes(rows, set='multipaper', system='updated'):>12}")

    print(f"\n{'=' * 80}\nANSWERS - 45 QUESTIONS FROM 3.1 (baseline = 3.1 runs 1 and 2)\n{'=' * 80}")
    print(f"{'metric':<40}{'3.1 run 1':>12}{'3.1 run 2':>12}{'updated':>12}")
    for prompt in ("balanced", "strict"):
        r1, r2 = verdicts(old, prompt, run="1"), verdicts(old, prompt, run="2")
        u = verdicts(rows, prompt, set="frozen45", system="updated")
        print(f"{'grader pass, ' + prompt:<40}{pc(r1):>12}{pc(r2):>12}{pc(u):>12}")
    for cat in sorted({c["category"] for c in all_cases("frozen45")}):
        r1 = {i: v for i, v in verdicts(old, "balanced", run="1").items() if cases[i]["category"] == cat}
        r2 = {i: v for i, v in verdicts(old, "balanced", run="2").items() if cases[i]["category"] == cat}
        u = {i: v for i, v in verdicts(rows, "balanced", set="frozen45", system="updated").items()
             if cases[i]["category"] == cat}
        print(f"{'  balanced, ' + cat:<40}{pc(r1):>12}{pc(r2):>12}{pc(u):>12}")
    c_ids = [c["id"] for c in all_cases("frozen45") if c["category"] == "cross_paper"]
    o1, o2, uq = cited(old_ans, c_ids, "|run1"), cited(old_ans, c_ids, "|run2"), cited(new_ans, c_ids, "|updated|run1")
    print(f"{'C1-C8 needed papers cited':<40}{o1[1]:>12}{o2[1]:>12}{uq[1]:>12}")
    print(f"{'quotes found on the cited page':<40}"
          f"{quotes(old, run='1'):>12}{quotes(old, run='2'):>12}{quotes(rows, set='frozen45', system='updated'):>12}")

    print("\nVERDICT CHANGES BEYOND NOISE (balanced): updated differs from BOTH 3.1 runs")
    r1, r2 = verdicts(old, "balanced", run="1"), verdicts(old, "balanced", run="2")
    u = verdicts(rows, "balanced", set="frozen45", system="updated")
    for i in sorted(u):
        if r1.get(i) == r2.get(i) != u[i]:
            print(f"  {i:<4} {r1[i]} -> {u[i]}")
    flips = [i for i in r1 if r1[i] != r2.get(i)]
    print(f"  (3.1 noise: {len(flips)} of {len(r1)} flipped between its own two runs: {flips})")

    print("\nMULTIPAPER VERDICT CHANGES (balanced, one run each - no noise floor)")
    b = verdicts(rows, "balanced", set="multipaper", system="baseline")
    u = verdicts(rows, "balanced", set="multipaper", system="updated")
    for i in m_ids:
        nb = needed_cited(new_ans[f"{i}|baseline|run1"]["answer"], cases[i])
        nu = needed_cited(new_ans[f"{i}|updated|run1"]["answer"], cases[i])
        flag = "  <-" if b[i] != u[i] or nb != nu else ""
        print(f"  {i:<4} {b[i]} -> {u[i]}   needed papers cited {nb[0]}/{nb[1]} -> {nu[0]}/{nu[1]}{flag}")


# %% [markdown]
# ## Step 5 — Run baseline vs. advanced and capture the evidence (starter demo)

# %%
def answer_from_docs(llm: ChatOpenAI, query: str, doc_ids: list[str]) -> str:
    context = "\n\n".join(f"[{i}] {DOC_BY_ID[i]['text']}" for i in doc_ids if i in DOC_BY_ID)
    messages = [
        SystemMessage(content=ANSWER_SYSTEM),
        HumanMessage(content=f"Documents:\n{context}\n\nQuestion: {query}"),
    ]
    return llm.invoke(messages).content


def run_demo() -> None:
    llm = make_llm()
    graph = build_graph(SAMPLE_DOCS)
    query = "How does breaking a question into parts and following document links help retrieval?"

    print("=" * 72)
    print(f"Checkpoint 4.1 demo — scenario: {SCENARIO}\nQuery: {query}\n")

    base = [i for i, _ in baseline_retrieve(query, 3)]
    print(f"BASELINE (single-pass) retrieved: {base}")

    multi = multistep_retrieve(llm, query, 3)
    print(f"MULTI-STEP retrieved:             {multi}")

    g = graph_retrieve(graph, query, 3)
    print(f"GRAPH retrieved (with sources):   {g}")

    answer = answer_from_docs(llm, query, list(g.keys()))
    print(f"\nGraph-augmented answer:\n{answer}\n")
    log_response("GRAPH", query, answer)

    try:
        plan = my_advanced_plan()
        print("Your advanced-retrieval plan:")
        print(json.dumps(plan, indent=2))
    except NotImplementedError as e:
        print(f"[my_advanced_plan not done yet] {e}")

    print("=" * 72)


# %%
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true", help="run the starter's toy demo (2 LLM calls)")
    ap.add_argument("--set", choices=sorted(pm.SETS), default="multipaper")
    ap.add_argument("--show", metavar="ID", help="readable before/after for one question, e.g. M9")
    ap.add_argument("--answer", action="store_true", help="generate and save answers (69 LLM calls)")
    ap.add_argument("--judge", action="store_true", help="judge saved answers (138 judge calls)")
    ap.add_argument("--report", action="store_true", help="answer-level tables for Step 5 (free)")
    args = ap.parse_args()
    if args.demo:
        run_demo()
        return
    check_frozen()
    if args.answer or args.judge or args.report:
        if args.answer:
            generate_answers()
        if args.judge:
            judge_answers()
        if args.report:
            report_answers()
        return
    cfg = pm.SETS[args.set]
    cache = build_cache(args.set, cfg)
    if args.show:
        show(args.set, cfg, cache, args.show)
        return
    compare(args.set, cfg, cache)


if __name__ == "__main__":
    main()
