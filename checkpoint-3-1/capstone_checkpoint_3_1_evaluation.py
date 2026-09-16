r"""Capstone Checkpoint 3.1 - Evaluation Infrastructure and Baseline Diagnosis.

Completed from capstone_checkpoint_3_1_evaluation_starter.py (kept alongside, untouched).

The starter evaluates a six-sentence demo corpus with a single pass/fail judge. That
demonstrates the workflow but produces no evidence about the capstone system, so this
file evaluates the REAL Checkpoint 2.1 baseline: hybrid BM25 + vector retrieval over
the 123-paper Research Papers index.

The 2.1 retriever is IMPORTED, not reimplemented. That is deliberate. Every number
here has to describe the system as built in Checkpoint 2.1, and a copy-pasted
retriever would silently drift from it.

Subcommands, in the order they are meant to be run:

    python capstone_checkpoint_3_1_evaluation.py --validate
        Seven checks on the TEST SET itself. Six are pure string work over the local
        text cache; check 2 (retrievability floor) embeds each query, so the command
        costs ~45 embedding calls and nothing else. Run this first:
        Lab 3.1 produced a metric that read 12% where the truth was 50%, entirely
        because the answer key was wrong. Validate the ruler before measuring.

    python capstone_checkpoint_3_1_evaluation.py --variants
        Generate 3 paraphrases for each of the 20 flagged cases -> testInputs_variants.json.
        These are LLM-written and MUST be reviewed by hand before use; Lab 3.2 showed
        lexical drift detectors flag 26 rows to find 1 real drift.

    python capstone_checkpoint_3_1_evaluation.py --answer [--variants-file F] [--run N]
        Generate answers ONCE and save them. Judging never regenerates answers.
        Lab 3.1's independent-judge experiment was confounded exactly this way.

    python capstone_checkpoint_3_1_evaluation.py --judge
        Judge the SAVED answers under both grading prompts. Retrieval and generation
        are byte-identical across the two, so any movement is caused by the ruler.

    python capstone_checkpoint_3_1_evaluation.py --report
        Compute all ten metrics and print the tables for worksheet Steps 4 and 5.
"""

# %%
from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import argparse
import csv
import json
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
CP21 = HERE.parent / "checkpoint-2-1"
sys.path.insert(0, str(CP21))

# The Checkpoint 2.1 system, imported unchanged. This is the system under test.
import capstone_checkpoint_2_1_baseline_retrieval as cp21  # noqa: E402

from langchain_core.messages import HumanMessage, SystemMessage  # noqa: E402
from langchain_openai import ChatOpenAI  # noqa: E402

# %%
TEST_SET = HERE / "testInputs.json"
VARIANTS_RAW = HERE / "testInputs_variants.json"
VARIANTS_CLEAN = HERE / "testInputs_variants_clean.json"
ANSWERS = HERE / "checkpoint_3_1_answers.json"
RESULTS = HERE / "checkpoint_3_1_results.csv"
TEXT_CACHE = cp21.TEXT_CACHE

JUDGE_MODEL = "openai/gpt-oss-120b"   # NOT the answering model. See make_judge().
PARAPHRASE_MODEL = "openai/gpt-5.4-mini"

# Files whose contents do not match the paper their filename names. Established by two
# independent detectors (arXiv stamp year vs filename year; filename first-author
# surname vs page-1 authors) and confirmed at the PDF via PDFKit, not only in the text
# cache. 1,029 of 8,092 chunks - 12.7% of the index - carry a false source label.
#
# This list is a FLOOR, not a total: the surname test could only be applied to 85 of
# 153 files, because the rest carry no parseable author surname in the filename.
MISATTRIBUTED = {
    "DBLP_conf_fmcad_AlurBJMRSSSTU13", "DBLP_conf_icra_CamposISK19",
    "DBLP_conf_icse_JeonQFFS16", "DBLP_conf_nips_Inala0PPB0RS20",
    "DBLP_conf_oopsla_YessenovXS11", "DBLP_conf_pldi_ChaudhuriS10",
    "DBLP_conf_pldi_SinghGS13", "DBLP_conf_pldi_Solar-LezamaATBSS07",
    "DBLP_conf_pldi_Solar-LezamaJB08", "DBLP_conf_popl_YangYS12",
    "DBLP_conf_ppopp_KurajSP22", "DBLP_journals_cacm_AlurSFS18",
    "DBLP_journals_cacm_WangZKS16", "DBLP_journals_ftpl_ChaudhuriEPSSY21",
    "DBLP_journals_ftpl_CheungS16", "DBLP_journals_sttt_Solar-Lezama13",
}

# Words that appear in grading notes because of how grading notes are WRITTEN, not
# because of what they are about. Left in, they inflate the relevance score by matching
# on scaffolding. Inherited from Lab 3.1.
GRADING_BOILERPLATE = {
    "must", "state", "states", "stating", "answer", "answers", "correct", "should",
    "paper", "papers", "collection", "excerpt", "excerpts", "page", "citation",
    "reproduce", "verbatim", "decline", "declines", "fail", "pass", "least",
    "distinct", "different", "example", "for", "such", "that", "with", "from",
}

_GRADING_PREAMBLE = "You are grading a retrieval-augmented answer against reference grading notes.\n"
_GRADING_INPUTS = "Response: {response}\n\nGrading Notes: {grading_notes}\n\nVerdict (pass/fail):"

# Two prompts, one metric. The system, corpus, questions and ANSWERS are identical
# across them - answers are generated once and saved. So any movement in the score is
# caused by the ruler alone. In Lab 3.1 that swing was 89% -> 40%.
JUDGE_PROMPTS = {
    "balanced": (
        _GRADING_PREAMBLE
        + "Return 'pass' if the response is factually consistent with the grading notes "
        "and captures their main point(s), even if it omits minor details or is worded "
        "differently. Return 'fail' if it contradicts the notes, is unsupported by them, "
        "or misses the central point.\n" + _GRADING_INPUTS
    ),
    "strict": (
        _GRADING_PREAMBLE
        + "Return 'pass' ONLY if the response covers every key point stated in the "
        "grading notes, with no omissions. Any missing detail, however minor, is a "
        "'fail'.\n" + _GRADING_INPUTS
    ),
}

PARAPHRASE_SYSTEM = (
    "You rewrite research questions for a robustness test. Given ONE question, produce "
    "THREE different paraphrases of it.\n\n"
    "Rules:\n"
    "1. Each paraphrase must ask for EXACTLY the same information as the original.\n"
    "2. The three must differ from each other and from the original in wording and "
    "sentence structure. Identical or near-identical outputs make the test useless.\n"
    "3. KEEP any name that identifies what is being asked about - paper titles, system "
    "names, benchmark names. Removing them changes the question rather than rewording "
    "it, and leaves a question nobody could answer.\n"
    "4. Vary the surrounding vocabulary and phrasing, not the thing being asked.\n\n"
    "Reply with exactly three lines, one paraphrase per line, no numbering, no commentary."
)

# %% [markdown]
# ## Normalization - why quote verification needs it
#
# Q8's target sentence is stored in the cache as "recently- proposed" and "analy- sis":
# the PDF broke those words across lines and the extractor preserved the break. A naive
# substring match rejects a perfectly correct quotation. Normalizing is not cosmetic;
# without it the attribution metric reports fabrication where there is none.

# %%
def normalize(text: str) -> str:
    """Fold a string to a form that survives PDF extraction artifacts."""
    t = unicodedata.normalize("NFKD", text)
    t = t.replace("“", '"').replace("”", '"')
    t = t.replace("‘", "'").replace("’", "'")
    t = t.replace("–", "-").replace("—", "-").replace("−", "-")
    t = re.sub(r"-\s+", "", t)          # rejoin hyphenated line breaks
    t = re.sub(r"[^a-z0-9]+", " ", t.lower())
    return re.sub(r"\s+", " ", t).strip()


def content_words(text: str) -> set[str]:
    toks = re.findall(r"[a-z0-9]{3,}", text.lower())
    return {t for t in toks
            if t not in cp21._STOPWORDS and t not in GRADING_BOILERPLATE}


def page_text(source: str, page: int) -> str:
    """Exact text of one page of one paper, from the cache the index was built from."""
    f = TEXT_CACHE / f"{source}.txt"
    if not f.exists():
        return ""
    parts = re.split(r"\f?PAGE (\d+)\n", f.read_text(encoding="utf-8", errors="replace"))
    for i in range(1, len(parts) - 1, 2):
        if int(parts[i]) == page:
            return parts[i + 1]
    return ""


def case_sources(c: dict) -> list[tuple[str, int]]:
    """All (source, page) pairs that legitimately support a case.

    Cross-paper cases have several. Scoring their grading notes against ONE page
    always looks poorly grounded - that false alarm is what flagged C5 and C7 on the
    first validation run. N4 carries two pages on purpose: its claim is severed by a
    page boundary, which is the thing that case exists to test.
    """
    if c.get("sources"):
        return [(x["source"], x["page"]) for x in c["sources"]]
    if c.get("source"):
        return [(c["source"], c["page"])]
    return []


def load_cases(variants_file: Path | None = None) -> list[dict]:
    cases = json.loads(TEST_SET.read_text(encoding="utf-8"))
    for c in cases:
        c["variant_of"] = None
    if variants_file and variants_file.exists():
        cases += json.loads(variants_file.read_text(encoding="utf-8"))
    return cases


# %% [markdown]
# ## The retriever, built once
#
# Chunks and index come from Checkpoint 2.1 untouched. Building the chunk list is
# local work over the text cache; the Chroma index already exists, so nothing is
# re-embedded and this costs nothing.

# %%
_RETRIEVER = None


def get_retriever(verbose: bool = False):
    global _RETRIEVER
    if _RETRIEVER is None:
        chunks = cp21.build_chunks(verbose=verbose)
        db = cp21.build_or_load_db(chunks)
        _RETRIEVER = cp21.HybridChunkRetriever(chunks, db)
    return _RETRIEVER


def make_judge() -> ChatOpenAI:
    """The judge is a DIFFERENT model from the one that writes the answers.

    Word the justification carefully in the writeup: this removes a NAMED weakness.
    It is not evidence that self-grading bias is small. Lab 3.1's attempt to measure
    that was confounded, because re-judging also regenerated the answers - the judge
    may have been grading different text. Self-grading bias here is UNMEASURED.
    """
    return ChatOpenAI(model=JUDGE_MODEL, temperature=0,
                      api_key=cp21.check_api_key(), base_url=cp21.OPENROUTER_BASE_URL)


# %% [markdown]
# ## Step A - validate the test set BEFORE trusting it
#
# Lab 3.1's retrieval metric reported 12% where hand-checking put the truth at 50%.
# Nothing was wrong with the system; the answer key named one supporting document when
# several existed. A wrong key silently scores every answer against a false standard,
# so the key gets checked first. All seven checks are deterministic - no model, no cost.

# %%
def validate() -> None:
    cases = load_cases()
    print(f"TEST SET VALIDATION - {len(cases)} cases\n" + "=" * 78)

    # 1. Answer-key grounding. Do the grading notes describe text that is actually in
    #    the named source at the named page? This is the 12%-vs-50% check.
    print("\n[1] ANSWER-KEY GROUNDING - grading-note content words present in the cited page")
    weak = []
    for c in cases:
        srcs = case_sources(c)
        if not srcs:
            continue
        pg = normalize(" ".join(page_text(s_, p_) for s_, p_ in srcs))
        want = content_words(c["grading_notes"])
        if not want:
            continue
        frac = sum(1 for w in want if w in pg) / len(want)
        if frac < 0.50:
            weak.append((frac, c["id"], f"{len(srcs)} source(s)", ""))
    if weak:
        for f, i, s, p in sorted(weak):
            print(f"    LOW {f:.0%}  {i:<4} {s} p.{p}")
        print(f"    -> {len(weak)} case(s) under 50%. Inspect before trusting the key.")
    else:
        print("    all sourced cases >= 50% grounded in their cited page")

    # 2. Retrievability floor. Where does the target paper actually rank, before any
    #    answering happens? This separates a HARD question from an IMPOSSIBLE one.
    #    At 2.1 the paraphrase target sat at BM25 #91 and outside the vector top 300,
    #    with CANDIDATE_POOL = 20. That question could never have passed at any k.
    #    NOTE: this is the one check in validate() that calls an API - one query
    #    embedding per case. Everything else here is local.
    print(f"\n[2] RETRIEVABILITY FLOOR - target rank vs pool size ({cp21.CANDIDATE_POOL})")
    retr = get_retriever()
    unreachable = []
    for c in cases:
        srcs = {s_ for s_, _ in case_sources(c)}
        if not srcs:
            continue
        bm = retr.bm25_topk(c["question"], 300)
        vec = retr.vector_topk(c["question"], 300)
        def rank_of(hits):
            """Best rank of ANY acceptable source - a cross-paper case is reachable
            if it can reach one of its several valid papers."""
            for n, (key, _) in enumerate(hits, 1):
                if retr._by_key[key]["source"] in srcs:
                    return n
            return None
        rb, rv = rank_of(bm), rank_of(vec)
        best = min([r for r in (rb, rv) if r], default=None)
        flag = ""
        if best is None or best > cp21.CANDIDATE_POOL:
            flag = "  <-- OUTSIDE THE POOL: cannot pass at any k"
            unreachable.append(c["id"])
        print(f"    {c['id']:<4} bm25 #{str(rb or '>300'):<5} vector #{str(rv or '>300'):<5}{flag}")
    print(f"    -> {len(unreachable)} case(s) unreachable by construction: {unreachable}")

    # 3. Answer leakage, WITH a minimum-length guard. Without the guard a three-word
    #    question like "What is NeSyCat?" scores 100% for a structural reason - its
    #    only content word is the thing being asked about. Same failure shape as the
    #    12%-vs-50% metric: a confident number whose denominator was not what the
    #    designer assumed.
    print("\n[3] ANSWER LEAKAGE - question content words appearing in grading notes (>=5 words only)")
    leak = []
    for c in cases:
        q = content_words(c["question"])
        if len(q) < 5:
            continue
        g = content_words(c["grading_notes"])
        leak.append((len(q & g) / len(q), c["id"]))
    for r, i in sorted(leak, reverse=True)[:5]:
        print(f"    {r:.0%}  {i}")
    print(f"    -> {len(leak)} cases measurable, mean {sum(r for r, _ in leak)/max(len(leak),1):.0%}; "
          f"{len(cases)-len(leak)} too short to measure")

    # 4. Prediction-bias audit. Did the person who wrote this set build it to fail?
    print("\n[4] PREDICTION BIAS - expected outcomes, recorded before the run")
    for k, v in Counter(c["expected"] for c in cases).most_common():
        print(f"    {v:>3}  ({v/len(cases):>4.0%})  {k}")

    # 5. Source diversity. How much of the corpus does this set actually exercise?
    print("\n[5] SOURCE DIVERSITY")
    srcs = [c["source"] for c in cases if c.get("source")]
    print(f"    {len(set(srcs))} distinct papers targeted of 123 indexed ({len(set(srcs))/123:.0%})")
    top = Counter(srcs).most_common(3)
    print(f"    most-targeted: {', '.join(f'{s} x{n}' for s, n in top)}")

    # 6. Corrupted-source membership. Every source_verification case must target a
    #    misattributed file; no other category may.
    print("\n[6] CORRUPTED-SOURCE MEMBERSHIP")
    bad = []
    for c in cases:
        s = c.get("source")
        if not s:
            continue
        want = c["category"] == "source_verification"
        if (s in MISATTRIBUTED) != want:
            bad.append((c["id"], c["category"], s))
    print(f"    {'OK' if not bad else 'MISMATCH: ' + str(bad)}")

    # 7. Duplicate questions. Lab 3.2 found 8 near-duplicate variants, one byte-identical.
    print("\n[7] DUPLICATE QUESTIONS (Jaccard > 0.8)")
    dupes = []
    toks = {c["id"]: content_words(c["question"]) for c in cases}
    ids = list(toks)
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            a, b = toks[ids[i]], toks[ids[j]]
            if a and b and len(a & b) / len(a | b) > 0.8:
                dupes.append((ids[i], ids[j]))
    print(f"    {'none' if not dupes else dupes}")
    print("\n" + "=" * 78)


# %% [markdown]
# ## Step B - paraphrase variants
#
# Robustness delta needs THE SAME question asked two ways. Two different sets of
# questions would measure how hard the questions were, not how much phrasing mattered.

# %%
def make_variants(n: int = 3) -> None:
    """Generate n paraphrases per flagged case, all in ONE call per case.

    The first attempt generated each variant in a separate call at temperature 0.7,
    with no knowledge of its siblings. Four cases came back with three byte-identical
    copies, and one produced three copies of the ORIGINAL. Independent samples from the
    same prompt converge; asking for three at once forces them apart.

    Nothing here replaces the hand review. Lab 3.2's lesson was that lexical detectors
    flagged 26 rows to find 1 real drift - deciding whether two questions still mean the
    same thing is a judgement, and lexical tools are the wrong instrument for it. The
    duplicate check below only removes the failures obvious enough to catch for free.
    """
    llm = ChatOpenAI(model=PARAPHRASE_MODEL, temperature=0.9,
                     api_key=cp21.check_api_key(), base_url=cp21.OPENROUTER_BASE_URL)
    out, dropped = [], []
    targets = [c for c in load_cases() if c["generate_variants"]]
    print(f"Generating {n} variants for each of {len(targets)} cases...\n")
    for c in targets:
        msg = [SystemMessage(content=PARAPHRASE_SYSTEM),
               HumanMessage(content=c["question"])]
        lines = [l.strip(" -*\t") for l in llm.invoke(msg).content.strip().splitlines()]
        lines = [l for l in lines if len(l) > 15][:n]
        seen = {normalize(c["question"])}
        print(f"  {c['id']}  ORIGINAL: {c['question']}")
        for k, line in enumerate(lines, 1):
            nl = normalize(line)
            if nl in seen:
                dropped.append((f"{c['id']}v{k}", line))
                print(f"      v{k} DROPPED (duplicate): {line}")
                continue
            seen.add(nl)
            v = dict(c)
            v["id"] = f"{c['id']}v{k}"
            v["question"] = line
            v["variant_of"] = c["id"]
            v["generate_variants"] = False
            out.append(v)
            print(f"      v{k} {line}")
        print()
    VARIANTS_RAW.write_text(json.dumps(out, indent=1), encoding="utf-8")

    # Side-by-side review sheet. The variants file alone cannot be reviewed, because
    # judging drift requires seeing the original next to it.
    review = HERE / "variants_review.csv"
    by_id = {c["id"]: c for c in load_cases()}
    with review.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["variant_id", "original_question", "variant_question",
                    "same_question?", "reviewer_note"])
        for v in out:
            w.writerow([v["id"], by_id[v["variant_of"]]["question"], v["question"], "", ""])
    print(f"{len(out)} variants -> {VARIANTS_RAW.name}   "
          f"({len(dropped)} auto-dropped as duplicates)")
    print(f"Side-by-side review sheet -> {review.name}")
    print("\nOpen the review sheet, put n in 'same_question?' for any variant that")
    print("drifted, then save the survivors as", VARIANTS_CLEAN.name)


def apply_review() -> None:
    """Build the clean variant set from the marked-up review sheet.

    Anything marked 'n' (or 'no') in the same_question? column is dropped. The gap
    between the raw file and the clean one is the drift-review evidence: it records
    how many LLM-written paraphrases silently changed the question, which is a
    limitation of the robustness metric rather than a result about the system.
    """
    review = HERE / "variants_review.csv"
    if not review.exists():
        print(f"No {review.name}. Run --variants first.")
        return
    rejected = set()
    for row in csv.DictReader(review.open(encoding="utf-8")):
        if row.get("same_question?", "").strip().lower() in ("n", "no", "0", "false"):
            rejected.add(row["variant_id"])
    raw = json.loads(VARIANTS_RAW.read_text(encoding="utf-8"))
    kept = [v for v in raw if v["id"] not in rejected]
    VARIANTS_CLEAN.write_text(json.dumps(kept, indent=1), encoding="utf-8")
    print(f"raw variants     : {len(raw)}")
    print(f"rejected by hand : {len(rejected)}  {sorted(rejected)}")
    print(f"kept             : {len(kept)}  -> {VARIANTS_CLEAN.name}")
    if raw:
        print(f"\nDrift rate: {len(rejected)}/{len(raw)} = {len(rejected)/len(raw):.0%} of "
              f"LLM-written paraphrases changed the question.")
        print("That figure is a limitation of the robustness metric, and belongs in Step 7.")


# %% [markdown]
# ## Step C - generate answers ONCE
#
# Answers are saved so the judge never regenerates them. Lab 3.1's independent-judge
# result was confounded because re-judging also re-answered, so the disagreement may
# have been the judge grading different text.

# %%
def generate_answers(variants_file: Path | None = None, run: int = 1) -> None:
    retr = get_retriever(verbose=True)
    llm = cp21.make_llm()
    cases = load_cases(variants_file)
    saved = json.loads(ANSWERS.read_text(encoding="utf-8")) if ANSWERS.exists() else {}
    print(f"\nAnswering {len(cases)} cases (run {run})...\n")
    for i, c in enumerate(cases, 1):
        context, hits = retr.context(c["question"], cp21.NUM_RETRIEVED)
        ans = cp21.answer(llm, c["question"], context)
        saved[f"{c['id']}|run{run}"] = {
            "id": c["id"], "run": run, "question": c["question"],
            "category": c["category"], "variant_of": c.get("variant_of"),
            "answer": ans,
            "retrieved": [{"source": h[0]["source"], "page": h[0]["page"],
                           "score": h[1], "text": h[0]["text"]} for h in hits],
            "ts": datetime.now().isoformat(timespec="seconds"),
        }
        print(f"  [{i:>3}/{len(cases)}] {c['id']:<7} {len(ans):>5} chars  "
              f"{len({h[0]['source'] for h in hits})} papers")
        ANSWERS.write_text(json.dumps(saved, indent=1), encoding="utf-8")
    print(f"\n{len(saved)} answers saved -> {ANSWERS.name}")


# %% [markdown]
# ## Step D - judge the SAVED answers under both prompts

# %%
def judge_saved() -> None:
    saved = json.loads(ANSWERS.read_text(encoding="utf-8"))
    cases = {c["id"]: c for c in load_cases(VARIANTS_CLEAN if VARIANTS_CLEAN.exists() else VARIANTS_RAW)}
    llm = make_judge()
    rows = []
    print(f"Judging {len(saved)} saved answers under {len(JUDGE_PROMPTS)} prompts...\n")
    for key, rec in saved.items():
        case = cases.get(rec["id"])
        if not case:
            continue
        for pname, ptext in JUDGE_PROMPTS.items():
            msg = [SystemMessage(content="Reply with exactly one word: pass or fail."),
                   HumanMessage(content=ptext.format(response=rec["answer"],
                                                     grading_notes=case["grading_notes"]))]
            verdict = "pass" if "pass" in llm.invoke(msg).content.strip().lower() else "fail"
            rows.append({**metrics_for(rec, case),
                         "judge_prompt": pname, "judge_model": JUDGE_MODEL,
                         "verdict": verdict})
            print(f"  {rec['id']:<7} run{rec['run']} {pname:<9} {verdict}")
    write_results(rows)


# %% [markdown]
# ## The deterministic metrics
#
# Six of the ten need no model at all. When the judge is the least trustworthy part of
# the setup, a suite that mostly does not depend on it is worth having.

# %%
CITE_RE = re.compile(r"\[([A-Za-z0-9_.\-]+)\s+p\.?\s*(\d+)\]")
QUOTE_RE = re.compile(r"[\"“]([^\"”]{25,})[\"”]")
YEAR_RE = re.compile(r"\b(19[89]\d|20[0-2]\d)\b")


def metrics_for(rec: dict, case: dict) -> dict:
    retrieved_text = " ".join(r["text"] for r in rec["retrieved"])
    norm_retrieved = normalize(retrieved_text)

    # RELEVANCE. Share of the grading notes' content words present in the retrieved
    # excerpts. Lexical, so it matches words rather than meaning: it cannot tell
    # paraphrase from absence. The 0.55 threshold from Lab 3.1 was chosen by looking at
    # the 8 rows it was tested on - treat it as a guess to recalibrate, not a boundary.
    want = content_words(case["grading_notes"])
    have = content_words(retrieved_text)
    relevance = len(want & have) / len(want) if want else 0.0

    cites = CITE_RE.findall(rec["answer"])
    cited_sources = [c[0] for c in cites]

    # COVERAGE. Distinct source files cited. Counts FILES, not works: 9 near-duplicate
    # pairs survive exact-hash dedup, so one paper can appear under two names.
    coverage = len(set(cited_sources))
    retrieved_papers = len({r["source"] for r in rec["retrieved"]})
    context_used = coverage / retrieved_papers if retrieved_papers else 0.0

    # QUOTE VERIFICATION. Does the quoted string appear on the page cited?
    quotes = QUOTE_RE.findall(rec["answer"])
    verified = 0
    for q in quotes:
        nq = normalize(q)
        if not nq:
            continue
        if any(nq in normalize(page_text(s, int(p))) for s, p in cites):
            verified += 1
        elif nq in norm_retrieved:
            verified += 1          # present in context but cite mismatched
    quote_verification = verified / len(quotes) if quotes else None

    # SOURCE VERIFICATION. Does the cited file contain the paper its name claims?
    # The judge cannot see this: it reads the answer and the grading notes, never the
    # source label. A fluent, accurately-quoted answer citing the wrong paper passes.
    sound = [s for s in cited_sources if s not in MISATTRIBUTED]
    source_verification = len(sound) / len(cited_sources) if cited_sources else None

    # DATE-CLAIM SUPPORT. When the answer asserts a year, is that year anywhere in the
    # retrieved excerpts? Measures hallucination directly rather than inferring it.
    # Chunk metadata carries source, title and page - there is no date field, so the
    # publication date of a chunk's own paper is available in 0% of cases.
    claimed = set(YEAR_RE.findall(rec["answer"]))
    supported = {y for y in claimed if y in retrieved_text}
    date_claim_support = len(supported) / len(claimed) if claimed else None

    return {
        "id": rec["id"], "run": rec["run"], "category": rec["category"],
        "variant_of": rec.get("variant_of"), "expected": case["expected"],
        "question": rec["question"],
        "relevance": round(relevance, 3),
        "coverage": coverage,
        "context_used": round(context_used, 3),
        "quote_verification": quote_verification,
        "source_verification": source_verification,
        "date_claim_support": date_claim_support,
        "n_quotes": len(quotes), "n_citations": len(cites),
        "cited_misattributed": ";".join(s for s in cited_sources if s in MISATTRIBUTED),
        "answer_chars": len(rec["answer"]),
    }


def write_results(rows: list[dict]) -> None:
    if not rows:
        print("no rows")
        return
    cols = list(rows[0])
    with RESULTS.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"\n{len(rows)} rows -> {RESULTS.name}")


# %% [markdown]
# ## Step E - report
#
# Rows are joined on (id, run, judge_prompt). NEVER on row position: Lab 3.2 joined
# result CSVs by row index, RAGAS returned rows in completion order, 0 of 53 aligned,
# and it produced a confident and entirely fictional analysis.

# %%
def report() -> None:
    if not RESULTS.exists():
        print("No results yet. Run --judge first.")
        return
    rows = list(csv.DictReader(RESULTS.open(encoding="utf-8")))
    f = lambda r, k: float(r[k]) if r.get(k) not in (None, "", "None") else None

    print("\n" + "=" * 78 + "\nACCURACY BY CATEGORY AND GRADING PROMPT\n" + "=" * 78)
    cats = sorted({r["category"] for r in rows})
    print(f"{'category':<22}{'balanced':>12}{'strict':>12}{'disagree':>12}")
    for cat in cats:
        sub = [r for r in rows if r["category"] == cat and not r["variant_of"]
               and r["run"] == "1"]
        bal = [r for r in sub if r["judge_prompt"] == "balanced"]
        strc = {r["id"]: r["verdict"] for r in sub if r["judge_prompt"] == "strict"}
        if not bal:
            continue
        pb = sum(r["verdict"] == "pass" for r in bal) / len(bal)
        ps = sum(v == "pass" for v in strc.values()) / len(strc) if strc else 0
        dis = sum(1 for r in bal if strc.get(r["id"]) != r["verdict"])
        print(f"{cat:<22}{pb:>11.0%}{ps:>12.0%}{dis:>9}/{len(bal)}")

    print("\n" + "=" * 78 + "\nDETERMINISTIC METRICS (judge-independent)\n" + "=" * 78)
    for cat in cats:
        sub = [r for r in rows if r["category"] == cat
               and r["judge_prompt"] == "balanced" and r["run"] == "1"]
        if not sub:
            continue
        rel = [f(r, "relevance") for r in sub if f(r, "relevance") is not None]
        cov = [f(r, "coverage") for r in sub if f(r, "coverage") is not None]
        qv = [f(r, "quote_verification") for r in sub if f(r, "quote_verification") is not None]
        sv = [f(r, "source_verification") for r in sub if f(r, "source_verification") is not None]
        ds = [f(r, "date_claim_support") for r in sub if f(r, "date_claim_support") is not None]
        m = lambda xs: f"{sum(xs)/len(xs):.0%}" if xs else "  -"
        print(f"{cat:<22} relevance {m(rel):>5}   quote-verif {m(qv):>5}   "
              f"source-verif {m(sv):>5}   date-support {m(ds):>5}   "
              f"coverage {sum(cov)/len(cov):.1f}" if cov else "")

    print("\n" + "=" * 78 + "\nFALSE CITATIONS THE JUDGE PASSED\n" + "=" * 78)
    bad = [r for r in rows
           if r["judge_prompt"] == "balanced" and r["verdict"] == "pass"
           and r["cited_misattributed"] and r["run"] == "1"]
    for r in bad:
        print(f"  {r['id']:<7} PASSED, citing {r['cited_misattributed']}")
    print(f"  -> {len(bad)} answer(s) judged correct while citing a file whose contents")
    print("     do not match its name. The judge cannot see source labels.")

    print("\n" + "=" * 78 + "\nROBUSTNESS (originals vs paraphrases)\n" + "=" * 78)
    for pname in ("balanced", "strict"):
        has_variants = {x["variant_of"] for x in rows if x["variant_of"]}
        orig = [r for r in rows if r["judge_prompt"] == pname and not r["variant_of"]
                and r["run"] == "1" and r["id"] in has_variants]
        var = [r for r in rows if r["judge_prompt"] == pname and r["variant_of"]
               and r["run"] == "1"]
        if not orig or not var:
            continue
        po = sum(r["verdict"] == "pass" for r in orig) / len(orig)
        pv = sum(r["verdict"] == "pass" for r in var) / len(var)
        print(f"  {pname:<9} originals {po:.0%} ({len(orig)})   "
              f"variants {pv:.0%} ({len(var)})   delta {100*(po-pv):+.1f} pts")

    print("\n" + "=" * 78 + "\nNOISE FLOOR - same 45 originals answered twice\n" + "=" * 78)
    for pname in ("balanced", "strict"):
        r1 = {r["id"]: r["verdict"] for r in rows
              if r["judge_prompt"] == pname and r["run"] == "1" and not r["variant_of"]}
        r2 = {r["id"]: r["verdict"] for r in rows
              if r["judge_prompt"] == pname and r["run"] == "2" and not r["variant_of"]}
        both = set(r1) & set(r2)
        if not both:
            continue
        flipped = sorted(i for i in both if r1[i] != r2[i])
        p1 = sum(r1[i] == "pass" for i in both) / len(both)
        p2 = sum(r2[i] == "pass" for i in both) / len(both)
        print(f"  {pname:<9} run1 {p1:.0%}   run2 {p2:.0%}   "
              f"{len(flipped)} of {len(both)} flipped ({len(flipped)/len(both)*100:.1f} pts)")
        if flipped:
            print(f"            flipped: {flipped}")
    print("  Anything smaller than this is not a measurement. The judge is at")
    print("  temperature 0, so all of this variance comes from generation at 0.2.")

    print("\n" + "=" * 78 + "\nPREDICTED vs ACTUAL\n" + "=" * 78)
    wrong = []
    for r in rows:
        if r["judge_prompt"] != "balanced" or r["variant_of"] or r["run"] != "1":
            continue
        exp, got = r["expected"], r["verdict"]
        if exp == "pass" and got == "fail":
            wrong.append((r["id"], exp, got))
        elif exp == "fail" and got == "pass":
            wrong.append((r["id"], exp, got))
    print(f"  {len(wrong)} case(s) where the pre-registered prediction was wrong:")
    for i, e, g in wrong:
        print(f"    {i:<7} predicted {e}, got {g}")
    print("  A wrong prediction is a finding, not an error - it is what pre-registering is for.\n")


# %%
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--variants", action="store_true")
    ap.add_argument("--apply-review", action="store_true")
    ap.add_argument("--answer", action="store_true")
    ap.add_argument("--judge", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--variants-file", type=str, default=None)
    ap.add_argument("--run", type=int, default=1)
    a = ap.parse_args()
    vf = Path(a.variants_file) if a.variants_file else None
    if a.validate:
        validate()
    elif a.variants:
        make_variants()
    elif a.apply_review:
        apply_review()
    elif a.answer:
        generate_answers(vf, a.run)
    elif a.judge:
        judge_saved()
    elif a.report:
        report()
    else:
        ap.print_help()
