r"""Capstone Checkpoint 5.1 — Designing and Evaluating an Agent-Based RAG System.
Jupytext-style cell markers (# %% / # %% [markdown]) — runnable as a
plain script AND openable as cells in VS Code/PyCharm/Jupytext.

The starter's cells are kept as they were, with its one TODO (my_agent_plan) filled in.
The starter ran its toy demo on import; that now runs only with --demo. The real agent,
over the Research Paper Navigator corpus, is added after Step 4 below.

    python capstone_checkpoint_5_1_agent_rag.py --demo      starter's toy demo (paid: ~4 calls)
    python capstone_checkpoint_5_1_agent_rag.py --plan      print my_agent_plan()
    python capstone_checkpoint_5_1_agent_rag.py --check     free: paper lookup + frozen files
    python capstone_checkpoint_5_1_agent_rag.py --answer    answer the test cases (paid; cached)
    python capstone_checkpoint_5_1_agent_rag.py --judge     judge saved answers (paid; cached)
    python capstone_checkpoint_5_1_agent_rag.py --rescore   free: recompute checker columns
    python capstone_checkpoint_5_1_agent_rag.py --report    comparison table (free)
    python capstone_checkpoint_5_1_agent_rag.py --show M7 agent_v1 1   one run, step by step
"""

# %% [markdown]
# # Capstone Checkpoint 5.1 — Designing and Evaluating an Agent-Based RAG System
# **MO-LLM Module 5 -  Required Capstone Checkpoint (120 minutes)**
#
# ## What this checkpoint is
#
# This is the final build step of your capstone. You will turn the retrieval system you've
# developed into an **agent-based RAG system**. Instead of a fixed pipeline, an agent
# decides at each step whether it has enough information, what to retrieve next, which
# tool to use, or whether to answer. This activity allows you to apply the Module 5 labs (Lab 5.1's agentic
# retriever and Lab 5.2's tool-using agent) to your capstone scenario.
#
# The graded deliverable is your completed Capstone Checkpoint 5.1 worksheet. This script is a
# runnable demonstration of an agentic loop on a tiny sample corpus so you can see the
# decision-making before adapting it to your full system.
#
# **Learning outcomes (Module 5):**
# 1. Build an agent-based system that integrates retrieval and external tools.
# 2. Use system prompts to guide agent behavior and decision-making.
# 3. Design workflows that coordinate retrieval, reasoning, and tool use in a RAG system.
# 4. Evaluate an agentic workflow against a fixed retrieval pipeline.
# 5. Design workflows that coordinate retrieval, reasoning, and tool use within a RAG system.
# 6. Build an agent-based system that integrates retrieval and external tools to complete user tasks.

# %% [markdown]
# ## Step 1 — Keep your capstone scenario
#
# Use the **same scenario** you chose in Checkpoint 1.1 and have built on since.
#
# | Scenario | Corpus | Useful agent tools/actions |
# |---|---|---|
# | **Research Paper Navigator** | ~150 research-paper PDFs | semantic search; "find papers by author/year"; "follow citations"; ask a clarifying question |
# | **Wikipedia Retrieval Engine** | ~2,400 Wikipedia HTML articles | semantic search; "find by category"; "follow links"; ask a clarifying question |
#
# An agent shines when one query isn't enough — it can search, look at what it found,
# and decide to search again (or use a different tool) before answering.

# %% [markdown]
# ## Setup (~5 min)
#
# 1. **Python 3.11 or 3.12**
# 2. `pip install langchain-openai langchain-core python-dotenv`
# 3. Get a free OpenRouter key at <https://openrouter.ai/keys>. The labs use the paid
#    `openai/gpt-5.4-mini` model, which is covered by the course credits.
# 4. Create a `.env` file next to this script: `OPENROUTER_API_KEY=sk-or-v1-...`
#
# This runs on a tiny built-in sample corpus, so you do not need to prepare your own
# dataset. It still requires an OpenRouter API key to run the LLM (it is not offline or
# free of API calls). The built-in corpus is deliberately minimal: It only demonstrates
# the agent control flow, not retrieval quality, and it is unrelated to your capstone:
# Howerver, your design should target your own capstone corpus. Your full agent is what you describe
# in the written submission.

# %%
from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

# %%
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
LLM_MODEL = "openai/gpt-5.4-mini"
TEMPERATURE = 0.2
MAX_STEPS = 3
HERE = Path(__file__).resolve().parent
LOG_PATH = HERE / "checkpoint_5_1_agent.log"   # starter: Path.cwd()

# === SET THIS to the scenario you chose in Checkpoint 1.1 ===
SCENARIO = "research_papers"   # "research_papers" or "wikipedia"

DECIDE_SYSTEM = (
    "You are an agent retrieving from a small document collection. Given the question, "
    "the queries already run, and the documents found so far, decide what to do next. "
    'Respond with ONLY a JSON object: {"done": true|false, "new_queries": ["..."], '
    '"reasoning": "..."}. Set done=true when you have enough to answer; otherwise give '
    "1-2 new_queries targeting what is still missing (do not repeat past queries)."
)
ANSWER_SYSTEM = (
    "You are a helpful assistant. Answer the question using ONLY the provided documents, "
    "quoting where you can. If they do not contain the answer, say so."
)


# %%
def check_api_key() -> str:
    load_dotenv()
    key = os.getenv("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError(
            "OPENROUTER_API_KEY not set. Grab a free key at https://openrouter.ai/keys, "
            "put it in a .env file next to this script, and rerun."
        )
    return key


def make_llm() -> ChatOpenAI:
    return ChatOpenAI(model=LLM_MODEL, temperature=TEMPERATURE,
                      api_key=check_api_key(), base_url=OPENROUTER_BASE_URL)


def log(label: str, text: str) -> None:
    ts = datetime.now().isoformat(timespec="seconds")
    with LOG_PATH.open("a", encoding="utf-8") as fh:
        fh.write(f"[{ts}] {label}\n{text}\n{'-' * 72}\n")


# %% [markdown]
# ## A tiny sample corpus + a keyword retriever (provided)

# %%
SAMPLE_DOCS = [
    {"id": "d1", "text": "Program synthesis generates programs from a specification, such as input-output examples."},
    {"id": "d2", "text": "The sketching approach lets a programmer leave holes in a program for a synthesizer to fill."},
    {"id": "d3", "text": "Retrieval-augmented generation grounds a model's answers in retrieved documents to reduce hallucination."},
    {"id": "d4", "text": "An agentic retriever decides at each step whether it has enough information or should search again."},
    {"id": "d5", "text": "A tool-using agent chooses among actions — search, look up by date, ask the user — to complete a task."},
    {"id": "d6", "text": "Evaluating an agent compares its answers and cost against a fixed single-pass pipeline."},
]
DOC_BY_ID = {d["id"]: d for d in SAMPLE_DOCS}


def retrieve(query: str, k: int = 2) -> list[str]:
    q = set(re.findall(r"[a-z0-9]+", query.lower()))
    scored = [(d["id"], len(q & set(re.findall(r"[a-z0-9]+", d["text"].lower())))) for d in SAMPLE_DOCS]
    scored.sort(key=lambda x: x[1], reverse=True)
    return [doc_id for doc_id, s in scored[:k] if s > 0]


def decide(llm: ChatOpenAI, question: str, collected: dict[str, str], executed: list[str]) -> dict:
    docs = "\n".join(f"[{i}] {DOC_BY_ID[i]['text']}" for i in collected) or "(none yet)"
    user = f"Question: {question}\n\nQueries run: {executed or '(none)'}\n\nDocuments so far:\n{docs}"
    raw = llm.invoke([SystemMessage(content=DECIDE_SYSTEM), HumanMessage(content=user)]).content.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0]
    try:
        d = json.loads(raw)
        return {"done": bool(d.get("done", True)), "new_queries": d.get("new_queries", []) or [],
                "reasoning": d.get("reasoning", "")}
    except (json.JSONDecodeError, ValueError):
        return {"done": True, "new_queries": [], "reasoning": "parse-fail -> stop"}


def agentic_answer(llm: ChatOpenAI, question: str) -> str:
    """Provided: a minimal agentic loop — retrieve, decide whether to continue, repeat."""
    collected: dict[str, str] = {}
    executed: list[str] = []
    pending = [question]
    for step in range(MAX_STEPS):
        for q in pending:
            for doc_id in retrieve(q):
                collected[doc_id] = DOC_BY_ID[doc_id]["text"]
            executed.append(q)
        d = decide(llm, question, collected, executed)
        print(f"  step {step + 1}: have {sorted(collected)}  -> done={d['done']}  ({d['reasoning'][:60]})")
        if d["done"] or not d["new_queries"]:
            break
        pending = d["new_queries"]
    context = "\n\n".join(f"[{i}] {collected[i]}" for i in collected)
    return llm.invoke([SystemMessage(content=ANSWER_SYSTEM),
                       HumanMessage(content=f"Documents:\n{context}\n\nQuestion: {question}")]).content


# %% [markdown]
# ## Step 2 — Your agent design (TODO)
#
# Design the agent you will build for **your** scenario. Return a dictionary with the keys
# below — this is the plan you implement in your real system and describe in the report.

# %%
def my_agent_plan() -> dict[str, Any]:
    """Return your agent design for your chosen scenario.

    TODO — your turn. Return a dictionary with these keys:
      - "tools": list[str]      — the actions/tools your agent can take (e.g.,
                                  ["semantic_search", "find_by_author", "follow_citation",
                                   "clarify", "answer"]).
      - "stop_condition": str   — how the agent decides it has enough to answer.
      - "system_prompt_idea": str — one or two sentences on how you'll instruct the agent
                                  to choose actions (this is what "guiding agent behavior
                                  with system prompts" means).
      - "test_tasks": list[str] — 2-3 questions for your scenario that need more than one
                                  retrieval step (so the agentic loop earns its keep).

    Example (illustrative — replace with your own scenario):
        return {
            "tools": ["semantic_search", "find_by_author", "clarify", "answer"],
            "stop_condition": "the retrieved passages cover every part of the question",
            "system_prompt_idea": "Search first; clarify only if the request is ambiguous; "
                                  "answer once the retrieved text supports a grounded reply.",
            "test_tasks": [
                "What changed between the v1 and v2 proposals, and who approved it?",
                "Summarize the budget decisions discussed across Q1 and Q2.",
            ],
        }

    """
    cases = {c["id"]: c for c in json.loads(CASES_FILE.read_text(encoding="utf-8"))}
    return {
        # search_all is the 2.1 hybrid search. search_in_paper is the 4.1 Step 7 design,
        # never built there. Left out: clarify (no one to answer during testing),
        # find-by-year (filename years name the venue, not the work), follow-citations
        # (no citation data exists in the index).
        "tools": ["search_all", "search_in_paper", "answer"],
        "stop_condition": (
            "The planner chooses 'answer' when it judges the excerpts enough. A hard cap "
            f"of {AGENT_MAX_STEPS} tool calls forces an answer, and so does an unreadable reply."
        ),
        "system_prompt_idea": (
            "Lab 5.2's action menu, adapted to papers: search the whole collection, or "
            "search inside one named paper, or answer. Papers are often named by their "
            "system rather than their title, so the prompt says to try alternate names."
        ),
        "test_tasks": [cases[i]["question"] for i in TEST_IDS],
    }


# %% [markdown]
# ## Step 3 — Run the agentic loop and capture the evidence
#
# Runs the provided agentic loop on a multistep question (watch it retrieve, decide,
# and retrieve again), then prints your plan. Reproduce this in your real system for the
# report and compare it against your fixed Checkpoint 2.1 retriever on the same task.

# %%
def run() -> None:
    llm = make_llm()
    question = "How does an agentic retriever differ from a fixed pipeline, and how is it evaluated?"
    print(f"Checkpoint 5.1 — agentic RAG demo  |  scenario: {SCENARIO}")
    print(f"Question: {question}\n")
    answer = agentic_answer(llm, question)
    print(f"\nAgent answer:\n{answer}\n")
    log("AGENTIC", f"Q: {question}\nA: {answer}")
    try:
        print("Your agent plan:")
        print(json.dumps(my_agent_plan(), indent=2))
    except NotImplementedError as e:
        print(f"[my_agent_plan not done yet] {e}")
    print("=" * 72)
    print("Done. Build this agent for your real system and compare it to your 2.1 baseline.")


# %% [markdown]
# ## Step 4 — Your written submission (the graded deliverable)
#
# The deliverable is a **written submission** (suggested length: 500-750 words). Cover the following:
#
# 1. **System overview**: State your scenario and the retrieval system you built (2.1–4.1).
# 2. **Agent design**: Describe the actions/tools your agent can take, and how a **system prompt**
#    guides it to choose among them (retrieve, use a tool, clarify, or answer).
# 3. **Workflow**: Explain how retrieval, reasoning, and tool use are coordinated across steps,
#    and how the agent decides it has enough information to answer.
# 4. **Evaluation**: Compare the agent-based system against your fixed Checkpoint 2.1/3.1
#    pipeline on a few representative tasks. Did multi-step decision-making improve the
#    answers? At what cost (extra model calls, latency)?
# 5. **Reflection**: When an agentic approach adds value vs. when a simpler pipeline is
#    better, and the limitations/trade-offs of increasingly autonomous systems.
#
# Include evidence (sample tasks, the agent's step-by-step decisions from a log, before/
# after comparison).


# %% [markdown]
# ## The real agent — Research Paper Navigator
#
# Everything below runs on the capstone corpus, not the toy one above. The design is
# Lab 5.2's: a PLAN node asks the model for one action as JSON, plain code runs that
# action, and control returns to PLAN until it chooses "answer" or hits the step cap.
#
#     question ─► PLAN ─► search_all ───────┐
#                  ▲  ─► search_in_paper ───┤  (each adds excerpts, then back to PLAN)
#                  └────────────────────────┘
#                  PLAN ─► answer ─► END
#
# The answer is written with 2.1's ANSWER_SYSTEM and 2.1's message format, so any
# difference from the baseline comes from which excerpts the agent gathered.

# %%
import argparse
import csv
import hashlib
import sys
import time
from typing import TypedDict

from langgraph.graph import END, StateGraph
from rank_bm25 import BM25Okapi

sys.path.insert(0, str(HERE.parent / "checkpoint-2-1"))
sys.path.insert(0, str(HERE.parent / "checkpoint-3-1"))
sys.path.insert(0, str(HERE))
import capstone_checkpoint_2_1_baseline_retrieval as cp21  # noqa: E402  - search, answer prompt
import capstone_checkpoint_3_1_evaluation as ev            # noqa: E402  - retriever, judge
import quote_check as qc                                    # noqa: E402  - fixed quote checker

CASES_FILE = HERE.parent / "checkpoint-4-1" / "testInputs_multipaper.json"
CASES_MD5 = "034a5150cdf2d09ac0800e9d464a9f14"   # frozen at 4.1; refuse to score if it changes
TEST_IDS = ("M5", "M7", "M9")
RUNS = (1, 2)                                    # every system answers every question twice
ANSWERS = HERE / "checkpoint_5_1_answers.json"
RESULTS = HERE / "checkpoint_5_1_results.csv"
TRACE_LOG = HERE / "checkpoint_5_1_traces.log"

# Set before any run and not tuned on results.
K = cp21.NUM_RETRIEVED     # 6: the baseline's context, and what one search_all adds
PAPER_K = 3                # what one search_in_paper adds: 4.1's per-paper cap, reused
AGENT_MAX_STEPS = 5        # tool calls per question: Lab 5.2's cap (the starter's toy MAX_STEPS stays 3)
PLANNER_SHOWS = 20         # excerpts shown to the planner: Lab 5.2's cap
GENERIC = {"the", "system", "approach", "method", "work", "et", "al", "tool", "framework"}


def check_frozen() -> None:
    got = hashlib.md5(CASES_FILE.read_bytes()).hexdigest()
    if got != CASES_MD5:
        raise SystemExit(f"{CASES_FILE.name} has changed (md5 {got}, frozen {CASES_MD5}).")


def load_cases() -> dict[str, dict]:
    check_frozen()
    return {c["id"]: c for c in json.loads(CASES_FILE.read_text(encoding="utf-8"))}


def block(c: dict) -> str:
    """One excerpt, labelled exactly as the 2.1 baseline labels it."""
    return f"[{c['source']} p.{c['page']}] ({c['title'][:70]})\n{c['text']}"


# %% [markdown]
# ### The two search tools
#
# `search_all` is the 2.1 hybrid search, unchanged. `search_in_paper` runs the same
# BM25 + vector fusion on one paper's excerpts only. It first has to turn a name like
# "Stitch" into a file: title first, then page 1, then a plain search as a last resort.

# %%
class PaperTools:
    def __init__(self, retr):
        self.retr = retr
        self.titles: dict[str, str] = {}
        self.page1: dict[str, str] = {}
        self.by_source: dict[str, list[dict]] = {}
        for c in retr._chunks:
            self.titles.setdefault(c["source"], c["title"])
            self.by_source.setdefault(c["source"], []).append(c)
            if c["page"] == 1:
                self.page1[c["source"]] = self.page1.get(c["source"], "") + " " + c["text"]
        self._bm25: dict[str, BM25Okapi] = {}

    def search_all(self, query: str, k: int = K) -> list[dict]:
        return [c for c, _ in self.retr.getTopK(query, k)]

    def resolve(self, name: str) -> tuple[str | None, str]:
        """Name from the planner -> (source file, how it was found)."""
        words = [w for w in re.findall(r"[a-z0-9]+", name.lower())
                 if w not in cp21._STOPWORDS and w not in GENERIC and len(w) > 1]
        if not words:
            return None, "no usable words in the name"

        def toks(text: str) -> list[str]:
            return re.findall(r"[a-z0-9]+", text.lower())

        def hits(src: str) -> int:
            t = toks(self.page1.get(src, ""))
            return sum(t.count(w) for w in words)

        in_title = [s for s in self.titles if set(words) <= set(toks(self.titles[s]))]
        if in_title:
            return max(sorted(in_title), key=hits), "title"
        on_page1 = [s for s in self.page1 if set(words) <= set(toks(self.page1[s]))]
        if on_page1:
            return max(sorted(on_page1), key=hits), "page 1"
        top = self.retr.getTopK(name, 1)
        return (top[0][0]["source"], "search") if top else (None, "not found")

    def search_in_paper(self, source: str, query: str, k: int = PAPER_K) -> list[dict]:
        """2.1's fusion (min-max, 50/50) over one paper's excerpts."""
        chunks = self.by_source[source]
        if source not in self._bm25:
            self._bm25[source] = BM25Okapi([cp21.tokenize(c["text"]) for c in chunks])
        scores = self._bm25[source].get_scores(cp21.tokenize(query))
        bm = sorted(range(len(chunks)), key=lambda i: scores[i], reverse=True)[:cp21.CANDIDATE_POOL]
        vec = self.retr._db.similarity_search_with_score(
            query, k=cp21.CANDIDATE_POOL, filter={"source": source})
        fused: dict[str, float] = {}
        for i, n in zip(bm, cp21._normalize([scores[i] for i in bm])):
            key = self.retr._key(chunks[i])
            fused[key] = fused.get(key, 0.0) + cp21.WEIGHT_BM25 * n
        for (d, _), n in zip(vec, cp21._normalize([s for _, s in vec], invert=True)):
            key = d.metadata["key"]
            fused[key] = fused.get(key, 0.0) + cp21.WEIGHT_VECTOR * n
        ranked = sorted(fused, key=fused.get, reverse=True)[:k]
        return [self.retr._by_key[key] for key in ranked if key in self.retr._by_key]


# %% [markdown]
# ### The planner's instructions (version 1 — the course's prompt, adapted)
#
# Lab 5.2's PLAN_SYSTEM with its tools swapped for this corpus's. The input the planner
# sees is also built the Lab 5.2 way. This is the "initial system" of worksheet Step 2;
# the Step 3 refinement is chosen after it has been evaluated, not before.

# %%
PLAN_SYSTEM_V1 = """You are a research agent for a collection of computer science research \
papers. You help the user find information using a search engine over excerpts of those papers.

You are given the user's question (and prior conversation), the searches already executed,
and the excerpts retrieved so far. Decide what to do next and respond with a JSON object ONLY:
{
  "action": "search_all" | "search_in_paper" | "answer",
  "queries": ["query1", "query2"],   // 1-3 NEW queries; for search_all and search_in_paper; don't repeat executed ones
  "paper": "paper name",             // only for action=="search_in_paper": the paper's title or its system's name
  "reasoning": "brief explanation"
}

Guidelines:
- "search_all": you need more information via semantic search across the whole collection.
- "search_in_paper": you need text from one specific paper. Searches only inside that paper.
- "answer": you have enough information to answer.
Papers are often referred to by the name of their system rather than their title — searching \
alternate names can help."""


def planner_input_v1(state: dict) -> str:
    """Assembled as Lab 5.2's plan_node assembles it (batch mode: no prior conversation)."""
    excerpts = "\n\n---\n\n".join(block(c) for c in list(state["chunks"].values())[:PLANNER_SHOWS])
    executed = "\n".join(f"- {q}" for q in state["executed"]) or "(none)"
    return (f"Prior conversation:\n(none)\n\n"
            f"Interactions so far:\n{state['question']}\n\n"
            f"Queries already executed:\n{executed}\n\n"
            f"Excerpts retrieved ({len(state['chunks'])} total):\n\n{excerpts}")


AGENTS = {
    "agent_v1": {"plan_system": PLAN_SYSTEM_V1, "planner_input": planner_input_v1},
}


# %% [markdown]
# ### The refinement (version 2) — chosen AFTER the v1 evaluation
#
# Step 2 found v1 stopped after one broad search in 6 of 6 runs and never used
# search_in_paper; on M7 it answered with zero excerpts from DreamCoder's own paper.
# Three changes, all to the planner, each from a Module 5 lab finding. The answer
# prompt and both tools are untouched, so the baseline comparison stays fair.
#
#   1. The question is labelled "The question to answer" (Lab 5.2: the planner acted
#      on the previous question because the current one sat unlabelled).
#   2. Every search reports which papers its excerpts came from, and a by-paper tally
#      is shown (Lab 5.1: the judge saw only what it found, never what was missing).
#   3. A rule: before answering, check every paper the question names against the
#      tally; a named paper with no excerpts of its own gets a search_in_paper first.

# %%
PLAN_SYSTEM_V2 = """You are a research agent for a collection of computer science research \
papers. You help the user find information using a search engine over excerpts of those papers.

You are given the question to answer, the searches already run and what each found, a tally
of excerpts by paper, and the excerpts themselves. Decide what to do next and respond with a
JSON object ONLY:
{
  "action": "search_all" | "search_in_paper" | "answer",
  "queries": ["query1", "query2"],   // 1-3 NEW queries; for search_all and search_in_paper; don't repeat executed ones
  "paper": "paper name",             // only for action=="search_in_paper": the paper's title or its system's name
  "reasoning": "brief explanation"
}

Guidelines:
- "search_all": you need more information via semantic search across the whole collection.
- "search_in_paper": you need text from one specific paper. Searches only inside that paper.
- "answer": you have enough information to answer.
- Before choosing "answer", list every paper the question names and check each against the
  by-paper tally. A named paper with NO excerpts from its own file is missing — use
  "search_in_paper" to get its text first. One paper's description of another paper is not
  the other paper's text.
- Do not repeat a search that found nothing new.
Papers are often referred to by the name of their system rather than their title — searching \
alternate names can help."""


def planner_input_v2(state: dict) -> str:
    """v1's input, restructured: the question labelled, each search's yield shown, and a
    by-paper tally so a missing paper is visible as an absence."""
    tally: dict[tuple[str, str], int] = {}
    for c in state["chunks"].values():
        k = (c["source"], c["title"][:60])
        tally[k] = tally.get(k, 0) + 1
    tally_lines = "\n".join(f"- {t} ({s}): {n} excerpt(s)"
                            for (s, t), n in sorted(tally.items(), key=lambda x: -x[1])) or "(none)"
    searches = []
    for t in state["trace"]:
        if "tool" not in t:
            continue
        found: dict[str, int] = {}
        for np_ in t["new"]:
            src = np_.rsplit(" p.", 1)[0]
            found[src] = found.get(src, 0) + 1
        desc = ", ".join(f"{s}({n})" for s, n in found.items()) or "nothing new"
        label = t["query"] if t["tool"] == "search_all" else f"in {t.get('paper', '?')}: {t['query']}"
        searches.append(f"- [{t['tool']}] {label!r} -> {desc}")
    excerpts = "\n\n---\n\n".join(block(c) for c in list(state["chunks"].values())[:PLANNER_SHOWS])
    return (f"The question to answer:\n{state['question']}\n\n"
            f"Searches already run, and what each found:\n" + ("\n".join(searches) or "(none)") +
            f"\n\nExcerpts gathered so far, by paper:\n{tally_lines}\n\n"
            f"The excerpts ({len(state['chunks'])} total; first {PLANNER_SHOWS} shown):\n\n{excerpts}")


AGENTS["agent_v2"] = {"plan_system": PLAN_SYSTEM_V2, "planner_input": planner_input_v2}


# %% [markdown]
# ### The agent graph

# %%
class _AgentState(TypedDict):
    question: str
    executed: list          # searches run, as the planner sees them
    chunks: dict            # excerpt key -> excerpt, in the order found; deduped for free
    steps: int              # tool calls so far; the cap counts these
    next_action: str
    pending_queries: list
    pending_paper: str
    trace: list             # every decision and what each tool returned
    answer: str


class PaperAgent:
    def __init__(self, tools: PaperTools, llm: ChatOpenAI, version: str = "agent_v1",
                 verbose: bool = True):
        self.tools, self.llm, self.version, self.verbose = tools, llm, version, verbose
        self.plan_system = AGENTS[version]["plan_system"]
        self.planner_input = AGENTS[version]["planner_input"]
        self.meter: dict = {}
        self._graph = self._build_graph()

    def _say(self, text: str) -> None:
        if self.verbose:
            print(text)
        with TRACE_LOG.open("a", encoding="utf-8") as fh:
            fh.write(text + "\n")

    def _invoke(self, messages: list, kind: str) -> str:
        resp = self.llm.invoke(messages)
        self.meter[f"{kind}_calls"] += 1
        usage = getattr(resp, "usage_metadata", None) or {}
        self.meter["in_tokens"] += usage.get("input_tokens", 0)
        self.meter["out_tokens"] += usage.get("output_tokens", 0)
        return resp.content if hasattr(resp, "content") else str(resp)

    def _build_graph(self):
        def add(state: dict, found: list[dict]) -> tuple[dict, list[str]]:
            chunks, new = dict(state["chunks"]), []
            for c in found:
                key = self.tools.retr._key(c)
                if key not in chunks:
                    chunks[key] = c
                    new.append(f"{c['source']} p.{c['page']}")
            return chunks, new

        def search_all_node(state: _AgentState) -> dict:
            chunks, executed, trace = dict(state["chunks"]), list(state["executed"]), list(state["trace"])
            for q in state["pending_queries"]:
                self.meter["embed_calls"] += 1
                chunks, new = add({"chunks": chunks}, self.tools.search_all(q))
                executed.append(q)
                trace.append({"tool": "search_all", "query": q, "new": new})
                self._say(f"  [search_all] {q!r} -> {len(new)} new excerpt(s)")
            return {"chunks": chunks, "executed": executed, "trace": trace,
                    "pending_queries": [], "steps": state["steps"] + 1}

        def search_in_paper_node(state: _AgentState) -> dict:
            chunks, executed, trace = dict(state["chunks"]), list(state["executed"]), list(state["trace"])
            name = state["pending_paper"]
            source, how = self.tools.resolve(name)
            if how == "search":
                self.meter["embed_calls"] += 1
            title = self.tools.titles.get(source, "")[:60] if source else ""
            self._say(f"  [search_in_paper] {name!r} -> {source} ({how}) {title!r}")
            for q in state["pending_queries"]:
                tag = f"PAPER:{name} :: {q}"
                if source:
                    self.meter["embed_calls"] += 1
                    chunks, new = add({"chunks": chunks}, self.tools.search_in_paper(source, q))
                else:
                    new = []
                executed.append(tag)
                trace.append({"tool": "search_in_paper", "paper": name, "resolved": source,
                              "how": how, "query": q, "new": new})
                self._say(f"      {q!r} -> {len(new)} new excerpt(s)")
            return {"chunks": chunks, "executed": executed, "trace": trace,
                    "pending_queries": [], "pending_paper": "", "steps": state["steps"] + 1}

        def plan_node(state: _AgentState) -> dict:
            trace = list(state["trace"])
            if state["steps"] >= AGENT_MAX_STEPS:
                self._say("  [plan] step cap reached — forcing answer")
                trace.append({"plan": "answer", "reasoning": "step cap reached (no model call)"})
                return {"next_action": "answer", "pending_queries": [], "pending_paper": "", "trace": trace}
            raw = self._invoke([SystemMessage(content=self.plan_system),
                                HumanMessage(content=self.planner_input(state))], "planner").strip()
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0]
            try:
                result = json.loads(raw)
                action = result.get("action", "answer")
            except (json.JSONDecodeError, ValueError):
                action, result = "answer", {"reasoning": f"unreadable reply: {raw[:80]!r}"}
            queries = result.get("queries", []) or []
            paper = result.get("paper", "") or ""
            self._say(f"  [plan] action={action} paper={paper!r} queries={queries}\n"
                      f"         reasoning={result.get('reasoning', '')!r}")
            trace.append({"plan": action, "queries": queries, "paper": paper,
                          "reasoning": result.get("reasoning", "")})
            if action == "search_all":
                return {"next_action": "search_all", "pending_queries": queries, "pending_paper": "", "trace": trace}
            if action == "search_in_paper":
                return {"next_action": "search_in_paper", "pending_queries": queries,
                        "pending_paper": paper, "trace": trace}
            return {"next_action": "answer", "pending_queries": [], "pending_paper": "", "trace": trace}

        def answer_node(state: _AgentState) -> dict:
            # 2.1's answer prompt and message format, with every excerpt gathered.
            context = "\n\n---\n\n".join(block(c) for c in state["chunks"].values())
            ans = self._invoke([SystemMessage(content=cp21.ANSWER_SYSTEM),
                                HumanMessage(content=f"Excerpts:\n{context}\n\nQuestion: {state['question']}")],
                               "answer")
            self._say(f"  [answer] from {len(state['chunks'])} excerpt(s), {state['steps']} tool call(s)")
            return {"answer": ans}

        def route(state: _AgentState) -> str:
            # As in Lab 5.2: an action missing what it needs falls through to answer.
            a = state["next_action"]
            if a == "search_all" and state["pending_queries"]:
                return "search_all"
            if a == "search_in_paper" and state["pending_paper"] and state["pending_queries"]:
                return "search_in_paper"
            return "answer"

        g = StateGraph(_AgentState)
        g.add_node("plan", plan_node)
        g.add_node("search_all", search_all_node)
        g.add_node("search_in_paper", search_in_paper_node)
        g.add_node("answer", answer_node)
        g.set_entry_point("plan")
        g.add_edge("search_all", "plan")
        g.add_edge("search_in_paper", "plan")
        g.add_conditional_edges("plan", route, {"search_all": "search_all",
                                                "search_in_paper": "search_in_paper",
                                                "answer": "answer"})
        g.add_edge("answer", END)
        return g.compile()

    def run(self, question: str) -> dict:
        self.meter = {"planner_calls": 0, "answer_calls": 0, "embed_calls": 0,
                      "in_tokens": 0, "out_tokens": 0}
        t0 = time.perf_counter()
        final = self._graph.invoke({
            "question": question, "executed": [], "chunks": {}, "steps": 0,
            "next_action": "plan", "pending_queries": [], "pending_paper": "",
            "trace": [], "answer": "",
        }, {"recursion_limit": 4 * AGENT_MAX_STEPS + 6})
        return {"answer": final["answer"], "chunks": list(final["chunks"].values()),
                "trace": final["trace"], "tool_calls": final["steps"],
                "seconds": round(time.perf_counter() - t0, 2), **self.meter}


def baseline_run(tools: PaperTools, llm: ChatOpenAI, question: str, k: int = K) -> dict:
    """The fixed 2.1 pipeline: one search, top k excerpts, one answer."""
    t0 = time.perf_counter()
    hits = tools.search_all(question, k)
    context = "\n\n---\n\n".join(block(c) for c in hits)
    resp = llm.invoke([SystemMessage(content=cp21.ANSWER_SYSTEM),
                       HumanMessage(content=f"Excerpts:\n{context}\n\nQuestion: {question}")])
    usage = getattr(resp, "usage_metadata", None) or {}
    return {"answer": resp.content, "chunks": hits, "trace": [], "tool_calls": 1,
            "seconds": round(time.perf_counter() - t0, 2), "planner_calls": 0,
            "answer_calls": 1, "embed_calls": 1, "in_tokens": usage.get("input_tokens", 0),
            "out_tokens": usage.get("output_tokens", 0)}


# %% [markdown]
# ### Evaluation
#
# Systems, each run twice per question:
#   baseline          2.1 as submitted: one search, 6 excerpts
#   agent_v1          the agent above
#   baseline_matched  2.1 given as many excerpts as agent_v1 gathered in that run. Not asked
#                     for by the worksheet. It separates "the agent chose well" from "the
#                     agent simply read more" — Lab 4.2's graph lost to plain search once
#                     the two were given the same budget.

# %%
BASE_SYSTEMS = ("baseline", "agent_v1", "baseline_matched")
MATCHED_TO = {"baseline_matched": "agent_v1", "baseline_matched_v2": "agent_v2"}


def save(saved: dict) -> None:
    ANSWERS.write_text(json.dumps(saved, indent=1), encoding="utf-8")


def generate(systems: tuple[str, ...] = BASE_SYSTEMS) -> None:
    cases = load_cases()
    retr, llm = ev.get_retriever(verbose=False), cp21.make_llm()
    tools = PaperTools(retr)
    saved = json.loads(ANSWERS.read_text()) if ANSWERS.exists() else {}
    for qid in TEST_IDS:
        q = cases[qid]["question"]
        for run_no in RUNS:
            for system in systems:
                key = f"{qid}|{system}|run{run_no}"
                if key in saved:
                    continue
                header = f"\n=== {key} ===\n{q}"
                print(header)
                with TRACE_LOG.open("a", encoding="utf-8") as fh:
                    fh.write(header + "\n")
                if system == "baseline":
                    out = baseline_run(tools, llm, q)
                elif system.startswith("baseline_matched"):
                    agent_key = f"{qid}|{MATCHED_TO[system]}|run{run_no}"
                    n = len(saved[agent_key]["retrieved"])
                    out = baseline_run(tools, llm, q, k=n)
                    print(f"  matched to {agent_key}: {n} excerpts")
                else:
                    out = PaperAgent(tools, llm, system).run(q)
                saved[key] = {
                    "id": qid, "run": run_no, "system": system, "question": q,
                    "category": cases[qid]["category"], "variant_of": None,
                    "answer": out["answer"],
                    "retrieved": [{"source": c["source"], "page": c["page"], "text": c["text"]}
                                  for c in out["chunks"]],
                    "trace": out["trace"],
                    **{k: out[k] for k in ("tool_calls", "seconds", "planner_calls", "answer_calls",
                                           "embed_calls", "in_tokens", "out_tokens")},
                    "ts": datetime.now().isoformat(timespec="seconds"),
                }
                save(saved)   # save as we go
                print(f"  -> {len(out['chunks'])} excerpts, {out['planner_calls']} planner + "
                      f"{out['answer_calls']} answer call(s), {out['seconds']}s")


def metrics_row(rec: dict, case: dict) -> dict:
    chk = qc.check_answer(rec["answer"], " ".join(r["text"] for r in rec["retrieved"]))
    needed = [s["source"] for s in case["sources"]]
    per_paper = [sum(r["source"] == n for r in rec["retrieved"]) for n in needed]
    cited = {s for s, _ in qc.all_citations(rec["answer"])}
    return {
        "id": rec["id"], "system": rec["system"], "run": rec["run"],
        "needed_in_context": sum(n > 0 for n in per_paper), "needed_total": len(needed),
        "needed_excerpts": "/".join(map(str, per_paper)),
        "needed_cited": sum(n in cited for n in needed),
        "quote_verification": chk["quote_verification"], "n_quotes": chk["n_quotes"],
        "source_verification": chk["source_verification"],
        "cited_misattributed": chk["cited_misattributed"],
        "n_excerpts": len(rec["retrieved"]),
        "context_chars": sum(len(r["text"]) for r in rec["retrieved"]),
        "tool_calls": rec["tool_calls"], "planner_calls": rec["planner_calls"],
        "answer_calls": rec["answer_calls"], "embed_calls": rec["embed_calls"],
        "in_tokens": rec["in_tokens"], "out_tokens": rec["out_tokens"], "seconds": rec["seconds"],
        "answer_chars": len(rec["answer"]),
    }


def judge() -> None:
    """Same judge, prompts and message as 3.1 and 4.1: a different model from the answerer."""
    cases = load_cases()
    saved = json.loads(ANSWERS.read_text())
    rows = list(csv.DictReader(RESULTS.open())) if RESULTS.exists() else []
    done = {(r["id"], r["system"], str(r["run"]), r["judge_prompt"]) for r in rows}
    jm = ev.make_judge()
    for rec in saved.values():
        for pname, ptext in ev.JUDGE_PROMPTS.items():
            if (rec["id"], rec["system"], str(rec["run"]), pname) in done:
                continue
            msg = [SystemMessage(content="Reply with exactly one word: pass or fail."),
                   HumanMessage(content=ptext.format(response=rec["answer"],
                                                     grading_notes=cases[rec["id"]]["grading_notes"]))]
            verdict = "pass" if "pass" in jm.invoke(msg).content.strip().lower() else "fail"
            rows.append({**metrics_row(rec, cases[rec["id"]]), "judge_prompt": pname,
                         "judge_model": ev.JUDGE_MODEL, "verdict": verdict})
            print(f"  {rec['id']:<4} {rec['system']:<17} run{rec['run']} {pname:<9} {verdict}")
            with RESULTS.open("w", newline="", encoding="utf-8") as fh:   # save as we go
                w = csv.DictWriter(fh, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows(rows)
    print(f"\n{len(rows)} rows -> {RESULTS.name}")


def report() -> None:
    """Built from the results file only, never from memory or the session log."""
    rows = list(csv.DictReader(RESULTS.open()))
    systems = [s for s in ("baseline", "agent_v1", "baseline_matched", "agent_v2",
                           "baseline_matched_v2") if any(r["system"] == s for r in rows)]
    print(f"{'case':<5}{'system':<20}{'balanced':<10}{'strict':<9}{'needed in ctx':<15}"
          f"{'cited':<10}{'quotes ok':<12}{'excerpts':<10}{'calls':<7}{'tokens in':<12}{'secs':<6}")
    for qid in TEST_IDS:
        for s in systems:
            rs = [r for r in rows if r["id"] == qid and r["system"] == s]
            if not rs:
                continue
            by_run = {}
            for r in rs:
                by_run.setdefault(r["run"], {})[r["judge_prompt"]] = r
            runs = [by_run[k] for k in sorted(by_run)]
            one = [next(iter(v.values())) for v in runs]
            verdict = lambda p: " ".join(v[p]["verdict"][0].upper() for v in runs if p in v)  # noqa: E731
            qv = [float(r["quote_verification"]) for r in one if r["quote_verification"] not in ("", "None")]
            print(f"{qid:<5}{s:<20}{verdict('balanced'):<10}{verdict('strict'):<9}"
                  f"{' '.join(r['needed_excerpts'] for r in one):<15}"
                  f"{' '.join(r['needed_cited'] + '/' + r['needed_total'] for r in one):<10}"
                  f"{' '.join(f'{x:.2f}' for x in qv):<12}"
                  f"{' '.join(r['n_excerpts'] for r in one):<10}"
                  f"{' '.join(str(int(r['planner_calls']) + int(r['answer_calls'])) for r in one):<7}"
                  f"{' '.join(r['in_tokens'] for r in one):<12}"
                  f"{' '.join(r['seconds'] for r in one)}")
    print("\nPer run, left to right. P/F = judge verdict. 'needed in ctx' = excerpts from each "
          "needed paper. 'cited' = needed papers cited. calls = model calls (planner + answer).")


def rescore() -> None:
    """Free: recompute the checker columns from the saved answers (e.g. after a checker
    fix). Judge verdicts are kept exactly as they were; no model is called."""
    cases = load_cases()
    saved = json.loads(ANSWERS.read_text())
    rows = list(csv.DictReader(RESULTS.open()))
    for r in rows:
        rec = saved[f"{r['id']}|{r['system']}|run{r['run']}"]
        r.update({k: v for k, v in metrics_row(rec, cases[r["id"]]).items()})
    with RESULTS.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} rows rescored -> {RESULTS.name} (verdicts unchanged)")


def show(qid: str, system: str, run_no: int) -> None:
    rec = json.loads(ANSWERS.read_text())[f"{qid}|{system}|run{run_no}"]
    print(f"{qid} {system} run{run_no}\n{rec['question']}\n")
    for t in rec["trace"]:
        print("  ", json.dumps(t, ensure_ascii=False)[:300])
    print(f"\nExcerpts sent to the answer ({len(rec['retrieved'])}):")
    for r in rec["retrieved"]:
        print(f"   {r['source']} p.{r['page']}")
    print(f"\nAnswer:\n{rec['answer']}")


def check() -> None:
    """Free: frozen test set, and which file each likely paper name resolves to."""
    cases = load_cases()
    print(f"{CASES_FILE.name}: md5 matches the 4.1 freeze")
    tools = PaperTools(ev.get_retriever(verbose=False))
    for qid in TEST_IDS:
        print(f"{qid} needs: {[s['source'] for s in cases[qid]['sources']]}")
    for name in ("DreamCoder", "Stitch", "JoinABLe", "Fusion 360 Gallery", "Synquid", "Lifty",
                 "Liquid Information Flow Control", "Program Synthesis from Polymorphic Refinement Types"):
        src, how = tools.resolve(name)
        print(f"  {name!r:<55} -> {src} ({how})  {tools.titles.get(src, '')[:50]!r}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--answer", nargs="*", metavar="SYSTEM")
    ap.add_argument("--judge", action="store_true")
    ap.add_argument("--rescore", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--show", nargs=3, metavar=("ID", "SYSTEM", "RUN"))
    a = ap.parse_args()
    if a.demo:
        run()
    if a.plan:
        print(json.dumps(my_agent_plan(), indent=2))
    if a.check:
        check()
    if a.answer is not None:
        generate(tuple(a.answer) or BASE_SYSTEMS)
    if a.judge:
        judge()
    if a.rescore:
        rescore()
    if a.report:
        report()
    if a.show:
        show(a.show[0], a.show[1], int(a.show[2]))
