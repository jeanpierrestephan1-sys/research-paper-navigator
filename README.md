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

One folder per capstone checkpoint. Each holds that checkpoint's solution file
and worksheet.

- `checkpoint-1-1/` — baseline evaluation of the LLM without retrieval

## Setup

Python 3.12 and an OpenRouter API key.

```
python -m venv .venv
source .venv/bin/activate
pip install langchain-openai langchain-core python-dotenv
```

Create a `.env` file containing:

```
OPENROUTER_API_KEY=your-key-here
```

## Running a checkpoint

Each checkpoint runs from inside its own folder, which is where its output log
is written.

```
cd checkpoint-1-1
python capstone_checkpoint_1_1_baseline.py
```

## Corpus

The research paper corpus is **not** included here. It is course-provided
material — 153 PDFs, roughly 500 MB, of third-party copyrighted papers.
Download it from the course platform and store it outside this repository.

