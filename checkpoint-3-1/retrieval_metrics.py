r"""Checkpoint 3.1 - retrieval-only evaluation.

Everything in the main harness is END-TO-END: question -> chunks -> answer -> verdict.
When an answer fails, that cannot say which stage broke. This stops at retrieval and
asks one question: given the query, did the correct paper come back?

Ground truth is the `source` / `sources` field already recorded on each test case, so
no new labelling is needed. No LLM is involved; the only API calls are the query
embeddings for the vector side.

The pair that matters is recall@6 vs recall@20:
  in the pool but not the top 6  -> a RANKING problem, fixable by reranking
  outside the pool entirely      -> a RETRIEVAL problem, no value of k rescues it
Checkpoint 2.1's paraphrase failure was the second kind. This measures it.
"""
import json, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "checkpoint-2-1"))
import capstone_checkpoint_2_1_baseline_retrieval as cp21
import capstone_checkpoint_3_1_evaluation as ev

K_CONTEXT = cp21.NUM_RETRIEVED      # 6 - what the model actually sees
K_POOL = cp21.CANDIDATE_POOL        # 20 - the ceiling on what fusion can select from
DEPTH = 300


def first_rank(hits, retr, wanted: set[str]) -> int | None:
    for n, (key, _) in enumerate(hits, 1):
        if retr._by_key[key]["source"] in wanted:
            return n
    return None


def main() -> None:
    cases = [c for c in json.loads((HERE / "testInputs.json").read_text())
             if ev.case_sources(c)]
    retr = ev.get_retriever(verbose=False)
    out = []
    for c in cases:
        wanted = {s for s, _ in ev.case_sources(c)}
        ranks = {
            "bm25":   first_rank(retr.bm25_topk(c["question"], DEPTH), retr, wanted),
            "vector": first_rank(retr.vector_topk(c["question"], DEPTH), retr, wanted),
        }
        hyb = retr.getTopK(c["question"], K_POOL)
        ranks["hybrid"] = next((n for n, (ch, _) in enumerate(hyb, 1)
                                if ch["source"] in wanted), None)
        out.append({"id": c["id"], "category": c["category"], **ranks})

    def agg(name):
        rs = [r[name] for r in out]
        rr = [1.0 / r if r else 0.0 for r in rs]
        return (sum(1 for r in rs if r and r <= K_CONTEXT) / len(rs),
                sum(1 for r in rs if r and r <= K_POOL) / len(rs),
                sum(rr) / len(rr))

    print(f"RETRIEVAL METRICS  -  {len(out)} queries with a known correct source\n")
    print(f"{'retriever':<12}{'recall@6':>10}{'recall@20':>11}{'MRR':>8}")
    print("-" * 41)
    for name in ("bm25", "vector", "hybrid"):
        r6, r20, mrr = agg(name)
        print(f"{name:<12}{r6:>9.0%}{r20:>10.0%}{mrr:>8.2f}")

    print(f"\nBY CATEGORY (hybrid)\n{'category':<22}{'recall@6':>10}{'recall@20':>11}{'MRR':>8}{'n':>4}")
    print("-" * 55)
    cats = sorted({r["category"] for r in out})
    for cat in cats:
        sub = [r["hybrid"] for r in out if r["category"] == cat]
        rr = [1.0 / r if r else 0.0 for r in sub]
        print(f"{cat:<22}{sum(1 for r in sub if r and r<=K_CONTEXT)/len(sub):>9.0%}"
              f"{sum(1 for r in sub if r and r<=K_POOL)/len(sub):>10.0%}"
              f"{sum(rr)/len(rr):>8.2f}{len(sub):>4}")

    ranking = [r["id"] for r in out
               if r["hybrid"] and K_CONTEXT < r["hybrid"] <= K_POOL]
    missing = [r["id"] for r in out if not r["hybrid"] or r["hybrid"] > K_POOL]
    print(f"\nRANKING problem  (in the pool, outside the top 6): {ranking or 'none'}")
    print(f"RETRIEVAL problem (outside the pool entirely)     : {missing or 'none'}")

    print(f"\nWHERE THE HALVES DISAGREE (rank gap > 20)")
    for r in out:
        b, v = r["bm25"], r["vector"]
        if b and v and abs(b - v) > 20:
            win = "vector" if v < b else "bm25"
            print(f"  {r['id']:<5} bm25 #{b:<4} vector #{v:<5} -> {win} carries it")
        elif bool(b) != bool(v):
            print(f"  {r['id']:<5} bm25 #{b or '>300':<4} vector #{v or '>300':<5} -> one side only")

    (HERE / "retrieval_metrics.json").write_text(json.dumps(out, indent=1))
    print(f"\nper-query ranks -> retrieval_metrics.json")


if __name__ == "__main__":
    main()
