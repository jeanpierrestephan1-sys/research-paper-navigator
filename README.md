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

## Asking the Navigator a question

The finished system is `checkpoint-7-1/navigator.py`. After Setup and Corpus
below:

```
cd checkpoint-7-1
python navigator.py --check                                   # offline self-test, no API calls
python navigator.py "What is MANTRA, and what does it validate?"
python navigator.py --chat                                    # one question after another
```

Each answer ends with a line showing the path it took (`cache`, `direct` or
`agent`), the model calls made, the seconds taken and the cost. If a quotation
in the answer is not on the page it cites, the answer says so underneath.

How a question is answered:

1. **Sanitize.** `& < >` are escaped and any line that starts with an excerpt
   label such as `[file p.4]` is removed, so pasted text cannot pose as a
   retrieved excerpt.
2. **Cache.** A question close enough to one answered before (cosine
   similarity 0.85 or more) gets the stored answer.
3. **Route.** A small model (`gpt-5-nano`) decides whether the question needs
   text from two or more papers, or a trend over time.
4. **Retrieve.** One paper: one hybrid search (keyword + vector) for 6
   excerpts. Several papers: the 5.1 agent plans its own searches across the
   collection or inside a named paper, up to 5 rounds.
5. **Answer.** `gpt-5.4-mini` answers from the excerpts only. Excerpts and
   question sit in separately labelled sections, and the prompt says only the
   excerpts are sources.
6. **Check.** Every quotation is checked against the page it cites. Only
   answers whose quotations were all found are stored in the cache.

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
- `checkpoint-6-1/` — the security, reliability and cost audit. Nothing is
  changed here; the 5.1 agent and the 2.1 pipeline are probed as they are. Two
  probes: a forged excerpt pasted into the question in the system's own
  `[file p.N]` format, and a question whose answer comes from a file named for
  the wrong paper. Calls, tokens, seconds and cost per question are re-derived
  from the 60 answers saved in 5.1. `rerun-2026-09-30/` holds the same probes
  run again a day later.
- `checkpoint-7-1/` — production readiness, and the finished system. Four
  changes, each aimed at a 6.1 finding: the question is kept apart from the
  excerpts (sanitizer and labelled prompt sections); every quotation is checked
  before the answer is returned; the 16 files named for the wrong paper are
  relabeled (`relabeled_files.json`: 15 with the real paper's name from page 1,
  one page of charts marked unidentified); and a router plus a semantic cache keep the agent for questions that
  need it. `capstone_checkpoint_7_1_evaluation.py` holds the before-and-after
  evidence: the 6.1 attacks and two new ones, and all 57 test questions run
  twice through the new system, the first pass graded by the 3.1 judge.
- `tools/` — the three Swift scripts that build the text cache and the paper
  index from the PDFs, plus a copy of `paper-index.tsv`. See Corpus below.

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

That key is the only setting the code reads from the environment. Every model
is called through OpenRouter:

| Model | Used for |
|---|---|
| `openai/gpt-5.4-mini` | answers, and the agent's planner (every checkpoint from 2.1) |
| `openai/gpt-5-nano` | the 7.1 router |
| `openai/text-embedding-3-small` | the vector index, and the 7.1 cache |
| `openai/gpt-oss-120b` | the judge in 3.1 onward; a different model from the one that answers |

At OpenRouter prices on 2026-10-06, an answer costs well under a cent on the
direct path and about a cent on the agent path; 7.1's evaluation prints the
measured figures. The 7.1 cache is a local folder, `checkpoint-7-1/semantic_cache/`,
created on first use and not committed. Delete it to start with an empty cache.

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

`text-cache/` is built from the PDFs with `tools/extract_all.swift`, and
`paper-index.tsv` with `tools/index.swift`. Both use Apple PDFKit, so they need
macOS. A copy of `paper-index.tsv` is kept in `tools/`, so only the text cache
has to be built. With `$REPO` set to the path of this repository:

```
cd ../data/capstone-corpus
swift $REPO/tools/extract_all.swift ResearchPapers text-cache
cp $REPO/tools/paper-index.tsv .        # or rebuild it: swift $REPO/tools/index.swift ResearchPapers > paper-index.tsv
```

`tools/pdftext.swift` prints one PDF's text and was used for spot checks, such
as comparing a file's first page against its name in 3.1. The checkpoints read
only the text cache and the index; they never open a PDF.

The vector index (`checkpoint-2-1/chroma_db/`) is built on the first 2.1 run and
is not committed. Building it embeds every chunk, which costs credits. Every
later checkpoint imports the 2.1 code and reuses that index.

## Running a checkpoint

Run each checkpoint from inside its own folder. Every checkpoint saves its
model output to files as it goes, so a rerun skips anything already done and
the `--report` commands cost nothing.

```
cd checkpoint-1-1
python capstone_checkpoint_1_1_baseline.py                        # runs the questions; this one has no free mode

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

cd checkpoint-6-1
python capstone_checkpoint_6_1_audit.py cost                      # free: cost, calls and latency from the 5.1 answers
python capstone_checkpoint_6_1_audit.py report                    # free: the saved probe answers and their checks
python capstone_checkpoint_6_1_audit.py probes                    # runs the 10 probe answers, skipping any already saved
cd rerun-2026-09-30 && python rerun_audit.py report               # the next-day re-run, same commands

cd checkpoint-7-1
python navigator.py --check                                       # free: filters tested on the exact 6.1 attack text
python capstone_checkpoint_7_1_evaluation.py report               # free: before and after, from the saved files
python capstone_checkpoint_7_1_evaluation.py threshold            # embeddings only: sets the cache threshold
python capstone_checkpoint_7_1_evaluation.py router               # picks the router prompt, on the 3.1 paraphrases
python capstone_checkpoint_7_1_evaluation.py probes               # 28 attack answers, skipping any already saved
python capstone_checkpoint_7_1_evaluation.py suite                # 57 questions, twice (second pass uses the cache)
python capstone_checkpoint_7_1_evaluation.py judge                # grades the first pass with the 3.1 judge
python capstone_checkpoint_7_1_production_ready.py                # the course's demo script; calls the model on start
```

Every checkpoint's solution file has a docstring at the top that lists its
commands and what each one costs.

## Known limitations

- **The corpus has defects the code works around, not fixes.** 16 files hold a
  different paper from the one their name says. Checkpoint 7.1 relabels them from
  a hand-made list (`checkpoint-7-1/relabeled_files.json`); a 17th would go
  unnoticed. Near-duplicate copies (for example a preprint beside its published
  version, found by hand in 3.1) remain in the index.
- **The quote check flags; it does not remove.** A quotation that is not on the
  page it cites is listed under the answer, but stays in it. The check is strict:
  it also flags a real quotation with a reference number silently left out, and
  a sentence the answer repeats only to say it is not in the papers.
- **The router adds about 6 seconds to each new question** (median 6.5 s in the
  router test) at the reasoning setting that routed best. A faster setting (`reasoning_effort="minimal"`) takes about 1
  second but misrouted 3 of 59 paraphrased questions in testing.
- **Questions are answered one at a time.** `--chat` does not remember earlier
  turns, and the cache is a local folder for one user.
- **The evidence is small.** Attacks were run twice per path, and the 57 test
  questions graded once by a model grader; two identical runs of the 3.1 test set
  flipped 2 of 45 verdicts on the balanced grader and 6 on the strict one. Treat
  differences of that size as noise.
- **Costs depend on OpenRouter.** Prices in the scripts were read on the dates in
  their comments, and model access depends on the key's allowlist.
- **The text cache needs macOS** (Apple PDFKit) to build from the PDFs.
