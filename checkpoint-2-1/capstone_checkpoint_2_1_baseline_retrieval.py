r"""Capstone Checkpoint 2.1 — Retrieval Strategy Design and Baseline Implementation.

Completed from capstone_checkpoint_2_1_baseline_retrieval_starter.py (the starter
is kept alongside, untouched).

The starter runs a shared-word-count retriever over six hardcoded sentences. That
demonstrates the workflow but produces no evidence about this corpus, so this file
implements the real baseline the worksheet asks for: hybrid BM25 + vector retrieval
over the 153-PDF Research Papers corpus, chunked, with page-level citations.

Run:
    python capstone_checkpoint_2_1_baseline_retrieval.py            # all queries + LLM answers
    python capstone_checkpoint_2_1_baseline_retrieval.py --stats    # corpus + index stats, no API calls
    python capstone_checkpoint_2_1_baseline_retrieval.py --fuse "<query>"   # fusion arithmetic, no LLM
    python capstone_checkpoint_2_1_baseline_retrieval.py --compare "<query>" # bm25 vs vector vs hybrid

Corpus text is read from the extracted cache, NOT the PDFs:
    ../../data/capstone-corpus/text-cache/*.txt
Rebuild it with:
    swift extract_all.swift ResearchPapers text-cache
"""

# %% [markdown]
# # Capstone Checkpoint 2.1 — baseline retrieval over the Research Paper corpus
#
# Checkpoint 1.1 established the before-picture: asked about specific papers with no
# retrieval, the model fabricated quotations that appear nowhere in the source, and
# contradicted itself across runs. This checkpoint adds the retrieval layer.

# %%
from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import hashlib
import os
import re
import shutil
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from rank_bm25 import BM25Okapi

# %%
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
LLM_MODEL = "openai/gpt-5.4-mini"
EMBEDDING_MODEL = "openai/text-embedding-3-small"
TEMPERATURE = 0.2

SCENARIO = "research_papers"

HERE = Path(__file__).resolve().parent
TEXT_CACHE = HERE.parent.parent / "data" / "capstone-corpus" / "text-cache"
PAPER_INDEX = HERE.parent.parent / "data" / "capstone-corpus" / "paper-index.tsv"
CHROMA_DIR = HERE / "chroma_db"
LOG_PATH = HERE / "checkpoint_2_1_retrieval.log"

# Chunking. Papers run to hundreds of pages, so unlike the Module 2 labs — where an
# email was stored whole — a document here cannot be a retrieval unit. CHUNK_CHARS is
# a compromise: large enough to hold a claim plus the sentences that qualify it, small
# enough that four of them fit in a prompt without crowding out the question.
CHUNK_CHARS = 1400
CHUNK_OVERLAP = 200        # so a claim split across a boundary survives in one piece

NUM_RETRIEVED = 6          # chunks sent to the model
CANDIDATE_POOL = 20        # candidates pulled from EACH retriever before fusion
WEIGHT_BM25 = 0.5
WEIGHT_VECTOR = 0.5
EMBED_BATCH = 500

ANSWER_SYSTEM = (
    "You are a research assistant answering questions about a corpus of computer "
    "science papers. Answer using ONLY the provided excerpts. Quote directly from "
    "them to support each claim, and name the source of every quotation using the "
    "[source p.N] label attached to the excerpt. If the excerpts do not contain the "
    "answer, say so plainly rather than guessing, and say what they do contain "
    "instead. Never reconstruct a quotation from memory."
)

_STOPWORDS = {
    "a", "an", "the", "and", "but", "or", "nor", "so", "yet", "for",
    "in", "on", "at", "to", "of", "by", "with", "from", "into", "onto", "upon",
    "about", "above", "below", "between", "through", "during", "before", "after",
    "under", "over", "around", "along", "across", "is", "are", "was", "were",
    "be", "been", "being", "have", "has", "had", "do", "does", "did",
    "i", "we", "you", "he", "she", "it", "they", "me", "us", "him", "her", "them",
    "my", "our", "your", "his", "its", "their", "this", "that", "these", "those",
    "as", "if", "up", "out", "not", "no",
    # Added after Module 2. The Lab 2.2 finding was that a query's META-vocabulary —
    # the words used to talk ABOUT the corpus rather than about its subject — is
    # itself a retrieval target, and competes with the real question for context
    # slots. In the email corpus, a one-line file reading "No Emails" scored highest
    # of anything in the lab on the query "what are the emails saying about a
    # detonation?". The equivalent words here are these.
    "paper", "papers", "author", "authors", "collection", "corpus", "document",
    "documents", "say", "says", "saying", "what", "which", "how", "many", "does",
}


def tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in _STOPWORDS]


def check_api_key() -> str:
    load_dotenv(HERE.parent.parent / ".env")
    load_dotenv()
    key = os.getenv("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY is not set. Put it in a .env file and rerun.")
    return key


def make_llm() -> ChatOpenAI:
    return ChatOpenAI(model=LLM_MODEL, temperature=TEMPERATURE,
                      api_key=check_api_key(), base_url=OPENROUTER_BASE_URL)


def get_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(model=EMBEDDING_MODEL, api_key=check_api_key(),
                            base_url=OPENROUTER_BASE_URL)


def log(label: str, text: str) -> None:
    ts = datetime.now().isoformat(timespec="seconds")
    with LOG_PATH.open("a", encoding="utf-8") as fh:
        fh.write(f"[{ts}] {label}\n{text}\n{'-' * 72}\n")


# %% [markdown]
# ## Step 1 — Decide what is actually in the corpus
#
# The folder holds 153 PDFs, but it does not hold 153 papers. Two problems have to be
# resolved before anything is indexed, and both are findings in their own right.

# %%
def load_titles() -> dict[str, str]:
    titles = {}
    if PAPER_INDEX.exists():
        for line in PAPER_INDEX.read_text(encoding="utf-8").splitlines():
            if "\t" in line:
                fname, title = line.split("\t", 1)
                titles[fname.replace(".pdf", ".txt")] = title.strip()
    return titles


def is_proceedings_volume(text: str) -> bool:
    """Detect a conference proceedings volume rather than a single paper.

    Detected from CONTENT, not from the filename, because the filename lies. Four of
    the five largest files in this corpus carry author-style names
    (DBLP_conf_cav_BraggFRS21, DBLP_conf_tacas_InalaPQLS17, DBLP_conf_cav_GaoKDRSAK19,
    DBLP_conf_cav_KongSG18) and are entire LNCS volumes. A hand-built index of this
    corpus flagged 14 volumes and missed DBLP_conf_snapl_2015 — the complete 347-page
    SNAPL'15 LIPIcs volume — which is exactly the kind of omission a content test
    catches and a filename test does not.

    They matter because a volume is not citable. A Research Paper Navigator answers
    "which paper says this", and a chunk drawn from page 300 of a proceedings volume
    belongs to some paper whose identity the chunk does not record. Worse, the five
    largest files hold 45% of all text in the corpus, so leaving them in would let
    unattributable material dominate the index.

    First attempt at this test looked for LNCS/LIPIcs series markers plus "(Eds.)",
    and wrongly caught 10 real papers — including Solar-Lezama's "The Sketching
    Approach to Program Synthesis". A paper published in LNCS carries its own series
    citation on page one ("...(Eds.): CAV 2011, LNCS 6806, pp. 24-40"), so the series
    name is evidence of nothing. What only a volume carries is the SERIES FRONT
    MATTER: the masthead ("Commenced Publication in 1973"), the editorial board, or a
    title page that says "Edited by". Those three plus a size test agree exactly with
    a hand-verified list of 15 — no false positives, nothing missed.
    """
    head = text[:4000].lower()
    return ("commenced publication" in head        # LNCS masthead
            or "editorial board" in head           # series board listing
            or "edited by" in head                 # volume title page
            or len(text) > 400_000)                # largest real paper here is 280k


def build_manifest(verbose: bool = True) -> list[dict]:
    """Return the documents to index, after removing volumes and duplicate texts."""
    titles = load_titles()
    files = sorted(f for f in os.listdir(TEXT_CACHE) if f.endswith(".txt"))

    records, volumes = [], []
    for f in files:
        text = (TEXT_CACHE / f).read_text(encoding="utf-8", errors="replace")
        rec = {"file": f, "text": text, "title": titles.get(f, f), "chars": len(text)}
        (volumes if is_proceedings_volume(text) else records).append(rec)

    # Exact-duplicate texts. This corpus pairs arXiv preprints with their published
    # conference versions under different filenames but byte-identical extracted text.
    # Left in, a query hits both and burns two context slots on one document, which
    # silently halves the effective breadth of every answer.
    by_hash: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_hash[hashlib.md5(r["text"].encode("utf-8", "replace")).hexdigest()].append(r)

    kept, dropped_dupes = [], []
    for group in by_hash.values():
        group.sort(key=lambda r: r["file"])   # deterministic: keep the first name
        kept.append(group[0])
        dropped_dupes.extend(group[1:])
        group[0]["duplicate_of"] = [g["file"] for g in group[1:]]
    kept.sort(key=lambda r: r["file"])

    if verbose:
        total = sum(len(open(TEXT_CACHE / f, encoding="utf-8", errors="replace").read())
                    for f in files)
        vol_chars = sum(r["chars"] for r in volumes)
        dup_chars = sum(r["chars"] for r in dropped_dupes)
        print(f"Corpus: {len(files)} files, {total:,} chars")
        print(f"  - {len(volumes):>3} proceedings volumes  ({vol_chars:,} chars, "
              f"{vol_chars/total:.0%} of the corpus)")
        print(f"  - {len(dropped_dupes):>3} duplicate copies    ({dup_chars:,} chars, "
              f"{dup_chars/total:.0%})")
        print(f"  = {len(kept):>3} papers indexed       "
              f"({sum(r['chars'] for r in kept):,} chars)")
    return kept


# %% [markdown]
# ## Step 2 — Chunking
#
# The Module 2 labs never exercised this: an email is ~800 characters and was stored
# whole. A paper here averages 120,000. Chunking is the decision that most shapes
# retrieval quality in this checkpoint.

# %%
def chunk_document(rec: dict) -> list[dict]:
    """Split one paper into overlapping chunks that each know their page number.

    Page-aware on purpose. The extractor emits a form-feed + 'PAGE n' marker per page,
    so a chunk can carry the page it came from and the model can cite '[file p.7]'.
    Checkpoint 1.1's failure was fabricated quotations; a citation that names a page
    is what makes a quotation checkable, which is the whole point of the scenario.

    Chunks never span a page boundary. That costs some recall for claims split across
    a page break, and buys citations that are exact rather than approximate.
    """
    chunks = []
    pages = re.split(r"\f?PAGE (\d+)\n", rec["text"])
    # re.split with one group yields [pre, num, text, num, text, ...]
    for i in range(1, len(pages) - 1, 2):
        page_no, body = int(pages[i]), pages[i + 1]
        body = re.sub(r"[ \t]+", " ", body).strip()
        if len(body) < 120:      # skip near-empty pages (figures, dividers)
            continue
        start = 0
        while start < len(body):
            piece = body[start:start + CHUNK_CHARS]
            if len(piece) >= 120:
                chunks.append({
                    "text": piece,
                    "source": rec["file"].replace(".txt", ""),
                    "title": rec["title"],
                    "page": page_no,
                })
            if start + CHUNK_CHARS >= len(body):
                break
            start += CHUNK_CHARS - CHUNK_OVERLAP
    return chunks


def build_chunks(verbose: bool = True) -> list[dict]:
    manifest = build_manifest(verbose=verbose)
    chunks = []
    for rec in manifest:
        chunks.extend(chunk_document(rec))
    if verbose:
        n_chars = sum(len(c["text"]) for c in chunks)
        print(f"Chunking: {len(chunks):,} chunks, {n_chars:,} chars "
              f"(~{n_chars//4:,} tokens, ~${n_chars/4/1_000_000*0.02:.2f} to embed)")
    return chunks


# %% [markdown]
# ## Step 3 — The retrievers
#
# BM25 and vector search over the same chunk list, fused as in Lab 2.2.

# %%
def _normalize(scores: list[float], invert: bool = False) -> list[float]:
    """Min-max scale to [0,1]; invert flips a distance into a similarity.

    Relative to the candidate pool, not absolute — so the best candidate always scores
    1.0 whether the query was answered well or not at all. Lab 2.2 showed the cost:
    when one retriever has no real signal, min-max manufactures one and hands it half
    the vote.
    """
    if not scores:
        return []
    lo, hi = min(scores), max(scores)
    if hi == lo:
        return [0.5] * len(scores)
    scaled = [(s - lo) / (hi - lo) for s in scores]
    return [1.0 - s for s in scaled] if invert else scaled


class HybridChunkRetriever:
    def __init__(self, chunks: list[dict], db: Chroma):
        self._chunks = chunks
        self._by_key = {self._key(c): c for c in chunks}
        self._keys = [self._key(c) for c in chunks]
        self._bm25 = BM25Okapi([tokenize(c["text"]) for c in chunks])
        self._db = db

    @staticmethod
    def _key(c: dict) -> str:
        return f"{c['source']}#p{c['page']}#{hashlib.md5(c['text'][:200].encode()).hexdigest()[:8]}"

    def bm25_topk(self, query: str, k: int) -> list[tuple[str, float]]:
        scores = self._bm25.get_scores(tokenize(query))
        top = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [(self._keys[i], float(scores[i])) for i in top]

    def vector_topk(self, query: str, k: int) -> list[tuple[str, float]]:
        hits = self._db.similarity_search_with_score(query, k=k)
        return [(d.metadata["key"], float(s)) for d, s in hits]

    def getTopK(self, query: str, k: int) -> list[tuple[dict, float]]:
        bm = self.bm25_topk(query, CANDIDATE_POOL)
        vec = self.vector_topk(query, CANDIDATE_POOL)
        bm_n = _normalize([s for _, s in bm])
        vec_n = _normalize([s for _, s in vec], invert=True)
        fused: dict[str, float] = defaultdict(float)
        for (key, _), n in zip(bm, bm_n):
            fused[key] += WEIGHT_BM25 * n
        for (key, _), n in zip(vec, vec_n):
            fused[key] += WEIGHT_VECTOR * n
        ranked = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)[:k]
        return [(self._by_key[key], score) for key, score in ranked if key in self._by_key]

    def context(self, query: str, k: int = NUM_RETRIEVED) -> tuple[str, list[tuple[dict, float]]]:
        hits = self.getTopK(query, k)
        blocks = [f"[{c['source']} p.{c['page']}] ({c['title'][:70]})\n{c['text']}"
                  for c, _ in hits]
        return "\n\n---\n\n".join(blocks), hits


# %%
def build_or_load_db(chunks: list[dict]) -> Chroma:
    if CHROMA_DIR.is_dir() and any(CHROMA_DIR.iterdir()):
        print(f"Loading existing vector DB from {CHROMA_DIR.name}/")
        return Chroma(persist_directory=str(CHROMA_DIR), embedding_function=get_embeddings())

    print("Building vector DB (first run — embedding the chunks)...")
    docs = [Document(page_content=c["text"],
                     metadata={"key": HybridChunkRetriever._key(c), "source": c["source"],
                               "page": c["page"], "title": c["title"][:120]})
            for c in chunks]
    # Stage and rename, so an interrupted build cannot leave a partial index that the
    # load branch above would silently accept forever.
    staging = Path(str(CHROMA_DIR) + ".building")
    if staging.is_dir():
        shutil.rmtree(staging)
    emb = get_embeddings()
    db = None
    for start in range(0, len(docs), EMBED_BATCH):
        batch = docs[start:start + EMBED_BATCH]
        if db is None:
            db = Chroma.from_documents(batch, emb, persist_directory=str(staging))
        else:
            db.add_documents(batch)
        print(f"  embedded {min(start + EMBED_BATCH, len(docs)):,}/{len(docs):,}", flush=True)
    del db
    os.rename(staging, CHROMA_DIR)
    print(f"  Indexed {len(docs):,} chunks into {CHROMA_DIR.name}/")
    return Chroma(persist_directory=str(CHROMA_DIR), embedding_function=get_embeddings())


# %% [markdown]
# ## Step 4 — Representative queries
#
# Chosen to exercise the four behaviors the Research Paper Navigator scenario claims:
# answer about one paper, compare across papers, identify change over time, and quote
# from source. Two of them reuse prompts from Checkpoint 1.1, where the un-retrieved
# model fabricated the answer — so the before/after is measured on identical wording.

# %%
def my_representative_queries() -> list[str]:
    return [
        # 1. PRECISE / single paper. Checkpoint 1.1: the model invented "program,
        #    action, and native" for PAN. Ground truth is in conf_nips_li2025encompass.
        "In the ENCOMPASS paper, what does PAN stand for and what two concerns does "
        "the programming model separate?",

        # 2. VERBATIM QUOTATION. Checkpoint 1.1: fabricated the sentence twice and
        #    refused twice. Tests whether retrieval makes exact quotation possible.
        "Quote the first sentence of the abstract of 'The Sketching Approach to "
        "Program Synthesis'.",

        # 3. CROSS-PAPER COMPARISON. Needs several documents, not one.
        "How do different papers in this collection define the program synthesis "
        "problem, and where do their definitions disagree?",

        # 4. PARAPHRASE / vocabulary mismatch. Deliberately avoids the corpus's own
        #    terms ("sketch", "partial program", "synthesis") to test whether the
        #    vector half earns its place, as it did on restart/resumption in Lab 2.1.
        "Which work lets a programmer leave blanks in an unfinished program for a "
        "machine to fill in automatically?",

        # 5. TREND OVER TIME. The scenario claims this; it is also the query type most
        #    likely to expose the counting and aggregation limit found in Module 1.
        "How did the stated motivation for program synthesis change between the "
        "earliest and the most recent work in this collection?",
    ]


# %%
def answer(llm: ChatOpenAI, query: str, context: str) -> str:
    messages = [SystemMessage(content=ANSWER_SYSTEM),
                HumanMessage(content=f"Excerpts:\n{context}\n\nQuestion: {query}")]
    return llm.invoke(messages).content


# %% [markdown]
# ## Step 5 — Run and capture the evidence

# %%
def show_hits(hits: list[tuple[dict, float]]) -> str:
    lines = []
    for rank, (c, score) in enumerate(hits, 1):
        lines.append(f"  #{rank} {score:.3f}  {c['source']} p.{c['page']}  — {c['title'][:64]}")
    return "\n".join(lines)


def run(k: int = NUM_RETRIEVED) -> None:
    """Run every representative query at a given context size.

    k is exposed because it is the single most misleading knob in this system. In the
    Module 2 labs, asking "how many emails discuss the hurricane" returned "three" —
    which was not a fact about the corpus but a count of how many retrieved documents
    happened to be relevant. The model had described its own context window and
    presented it as a property of the collection. Any claim about context size in the
    writeup should be measured with this flag rather than asserted.
    """
    chunks = build_chunks()
    db = build_or_load_db(chunks)
    retr = HybridChunkRetriever(chunks, db)
    llm = make_llm()
    print(f"\nCheckpoint 2.1 — hybrid retrieval  |  scenario: {SCENARIO}")
    print(f"weights bm25={WEIGHT_BM25} vector={WEIGHT_VECTOR}, pool={CANDIDATE_POOL}, "
          f"k={k}\n")
    for i, query in enumerate(my_representative_queries(), 1):
        context, hits = retr.context(query, k)
        print("=" * 78)
        print(f"QUERY {i} (k={k}): {query}\n")
        print("retrieved:")
        print(show_hits(hits))
        ans = answer(llm, query, context)
        print(f"\nanswer:\n{ans}\n")
        log(f"QUERY {i} (k={k}): {query}",
            f"retrieved:\n{show_hits(hits)}\n\nanswer:\n{ans}")
    print("=" * 78)
    print(f"Done. Full transcript in {LOG_PATH.name}")


def compare(query: str) -> None:
    """Same query through BM25 alone, vector alone, and the fusion. No LLM call."""
    chunks = build_chunks(verbose=False)
    db = build_or_load_db(chunks)
    retr = HybridChunkRetriever(chunks, db)
    by_key = retr._by_key

    W = 54

    def fmt(key):
        c = by_key[key]
        return f"{c['source'][:30]:<30} p.{c['page']:<4}"

    bm = retr.bm25_topk(query, NUM_RETRIEVED)
    vec = retr.vector_topk(query, NUM_RETRIEVED)
    hyb = retr.getTopK(query, NUM_RETRIEVED)
    print(f'\nQuery: "{query}"\n')
    print(f"{'BM25 only (score)':<{W}}{'VECTOR only (distance)':<{W}}{'HYBRID (fused)'}")
    print("-" * (W * 2 + 44))
    for i in range(NUM_RETRIEVED):
        a = f"{fmt(bm[i][0])} {bm[i][1]:7.2f}" if i < len(bm) else ""
        b = f"{fmt(vec[i][0])} {vec[i][1]:7.3f}" if i < len(vec) else ""
        c = (f"{hyb[i][0]['source'][:30]:<30} p.{hyb[i][0]['page']:<4} {hyb[i][1]:7.3f}"
             if i < len(hyb) else "")
        print(f"{a:<{W}}{b:<{W}}{c}")
    bm_src = {by_key[k]['source'] for k, _ in bm}
    vec_src = {by_key[k]['source'] for k, _ in vec}
    print(f"\npapers: bm25={len(bm_src)}  vector={len(vec_src)}  "
          f"overlap={len(bm_src & vec_src)}  ({sorted(bm_src & vec_src)})\n")


def fuse_probe(query: str) -> None:
    """Show the fusion arithmetic for one query. No LLM call."""
    chunks = build_chunks(verbose=False)
    db = build_or_load_db(chunks)
    retr = HybridChunkRetriever(chunks, db)
    bm = retr.bm25_topk(query, CANDIDATE_POOL)
    vec = retr.vector_topk(query, CANDIDATE_POOL)
    bm_n = _normalize([s for _, s in bm])
    vec_n = _normalize([s for _, s in vec], invert=True)
    bmd = {k: (i + 1, n) for i, ((k, _), n) in enumerate(zip(bm, bm_n))}
    vecd = {k: (i + 1, n) for i, ((k, _), n) in enumerate(zip(vec, vec_n))}
    rows = []
    for key in set(bmd) | set(vecd):
        b = WEIGHT_BM25 * bmd[key][1] if key in bmd else 0.0
        v = WEIGHT_VECTOR * vecd[key][1] if key in vecd else 0.0
        rows.append((b + v, key, b, v))
    rows.sort(key=lambda r: (-r[0], r[1]))
    print(f'\nQuery: "{query}"')
    print(f"  {len(set(bmd) & set(vecd))} of {len(rows)} candidates found by BOTH\n")
    print(f"  {'fused':>6} {'bm25':>10} {'vector':>10}  {'both':<5} source")
    print(f"  {'-'*6} {'-'*10} {'-'*10}  {'-'*5} {'-'*50}")
    for pos, (total, key, b, v) in enumerate(rows):
        c = retr._by_key[key]
        bc = f"#{bmd[key][0]:<2} {b:.3f}" if key in bmd else "  --     "
        vc = f"#{vecd[key][0]:<2} {v:.3f}" if key in vecd else "  --     "
        mark = "<<" if pos < NUM_RETRIEVED else "  "
        print(f"  {total:6.3f} {bc:>10} {vc:>10}  {'yes' if key in bmd and key in vecd else '':<5} "
              f"{c['source'][:40]:<40} p.{c['page']:<4} {mark}")
    print(f"\n  '<<' marks the {NUM_RETRIEVED} chunks sent to the model.\n")


def stats() -> None:
    chunks = build_chunks()
    per_paper = defaultdict(int)
    for c in chunks:
        per_paper[c["source"]] += 1
    counts = sorted(per_paper.values())
    print(f"\nchunks per paper: min {counts[0]}, median {counts[len(counts)//2]}, "
          f"max {counts[-1]}, papers {len(counts)}")
    print("\nmost-chunked papers (these dominate any broad query):")
    for src, n in sorted(per_paper.items(), key=lambda kv: -kv[1])[:8]:
        print(f"  {n:>5} chunks  {src}")


if __name__ == "__main__":
    argv = sys.argv[1:]
    if argv and argv[0] == "--stats":
        stats()
    elif argv and argv[0] == "--fuse":
        fuse_probe(argv[1])
    elif argv and argv[0] == "--compare":
        compare(argv[1])
    elif argv and argv[0] == "--k":
        run(int(argv[1]))
    else:
        run()
