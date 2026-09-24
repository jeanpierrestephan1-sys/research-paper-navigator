# Research Paper Navigator

Capstone project for the MIT RAG program. A retrieval-augmented LLM system that
answers questions about a collection of computer science research papers and
supports its answers with direct quotations from the source documents.

## Scenario

Research Paper Navigator. The system must:

- Answer questions about individual papers
- Compare ideas across multiple papers
- Identify trends or changes over time
- Quote directly from sources to support its answers

## Purpose

A general LLM answers questions about research papers confidently but without
grounding. It cannot quote accurately, and when its training knowledge is thin
it fabricates. This project adds retrieval so that every answer is traceable to
a specific document.

## Structure

One folder per capstone checkpoint. Each holds the course starter file
(untouched), the completed solution file, the submitted worksheet, and the
result files that the worksheet's numbers come from.

- `checkpoint-1-1/` — the LLM with no retrieval. It fabricated quotations,
  so retrieval is needed.
- `checkpoint-2-1/` — the baseline search. Each paper is split into chunks of
  1,400 characters that never cross a page. A keyword search (BM25) and a vector
  search each return 20 chunks; their scores are combined and the top 6 go to the
  model with a `[file p.N]` label on each. The vector index is built here and
  reused by every later checkpoint.
- `checkpoint-3-1/` — the evaluation harness. 45 test questions with grading
  notes (`testInputs.json`), a separate grader model, and checks that need no
  model: does each quote appear on the page it cites, is the cited file the paper
  its name claims, is each year in the answer present in the retrieved text.
- `checkpoint-4-1/` — advanced retrieval for questions that need two or three
  papers. The question is split into sub-queries; the sub-queries take turns
  filling the six chunks, with no paper allowed more than three. 12 new
  multi-paper questions (`testInputs_multipaper.json`).
- `checkpoint-5-1/` — the agent. A planner picks one action at a time (search
  the whole collection, search inside one named paper, or answer) until it
  answers or reaches 5 tool calls. Includes `quote_check.py`, a corrected
  version of the 3.1 quote check.

## Setup

Python 3.12 and an OpenRouter API key.

```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file in the repo root containing:

```
OPENROUTER_API_KEY=your-key-here
```

## Corpus

The research paper corpus is **not** included here. It is course-provided
material — 153 PDFs, roughly 500 MB, of third-party copyrighted papers.
Download it from the course platform and store it outside this repository.

The code expects this layout, one level above the repo:

```
../data/capstone-corpus/
    ResearchPapers/      the 153 PDFs from the course
    text-cache/          one .txt per PDF, with a "PAGE n" line before each page
    paper-index.tsv      one line per PDF: filename, tab, title
```

`text-cache/` is built from the PDFs with `extract_all.swift` (Apple PDFKit, so
macOS), and `paper-index.tsv` with `index.swift`. Both scripts live beside the
corpus, outside this repository. The checkpoints read only the text cache and
the index; they never open a PDF.

The vector index (`checkpoint-2-1/chroma_db/`) is built on the first 2.1 run and
is not committed. Building it embeds every chunk, which costs credits. Every
later checkpoint imports the 2.1 code and reuses that index.

## Running a checkpoint

Run each checkpoint from inside its own folder. Every checkpoint saves its
model output to files as it goes, so a rerun skips anything already done and
the `--report` commands cost nothing.

```
cd checkpoint-1-1
python capstone_checkpoint_1_1_baseline.py

cd checkpoint-2-1
python capstone_checkpoint_2_1_baseline_retrieval.py --stats      # index stats, no API calls
python capstone_checkpoint_2_1_baseline_retrieval.py              # builds the index on first run

cd checkpoint-3-1
python capstone_checkpoint_3_1_evaluation.py --validate           # check the test set first
python capstone_checkpoint_3_1_evaluation.py --answer
python capstone_checkpoint_3_1_evaluation.py --judge
python capstone_checkpoint_3_1_evaluation.py --report

cd checkpoint-4-1
python capstone_checkpoint_4_1_advanced_retrieval.py --show M7    # before/after for one question
python capstone_checkpoint_4_1_advanced_retrieval.py --answer
python capstone_checkpoint_4_1_advanced_retrieval.py --judge
python capstone_checkpoint_4_1_advanced_retrieval.py --report

cd checkpoint-5-1
python capstone_checkpoint_5_1_agent_rag.py --check               # free: frozen test set, paper lookup
python capstone_checkpoint_5_1_agent_rag.py --answer
python capstone_checkpoint_5_1_agent_rag.py --judge
python capstone_checkpoint_5_1_agent_rag.py --report
python capstone_checkpoint_5_1_agent_rag.py --show M7 agent_v2 1  # one run, step by step
python quote_check.py --validate                                  # the fixed quote check against the 3.1 answers
```

Every checkpoint's solution file has a docstring at the top that lists its
commands and what each one costs.
