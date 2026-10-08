"""Compare BM25-only, vector-only and hybrid retrieval on a labelled question set.

Metrics: Hit@k (is a chunk from the correct document in the top k?) and MRR
(mean reciprocal rank of the first correct-document chunk).
Also checks the guardrail on out-of-scope questions (no LLM call needed).
Usage:  python -m eval.run_eval [--docs data/docs] [--k 3]
"""
import argparse
import json

from unibot.bot import is_out_of_scope
from unibot.chunking import build_chunks, load_documents
from unibot.retrieval import Retriever


def evaluate(retriever, questions, mode, k):
    hits, rr = 0, 0.0
    for q in questions:
        chunks, _ = retriever.search(q["question"], mode, k)
        sources = [c["source"] for c in chunks]
        if q["source"] in sources:
            hits += 1
            rr += 1.0 / (sources.index(q["source"]) + 1)
    n = len(questions)
    return hits / n, rr / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", default="data/docs")
    ap.add_argument("--k", type=int, default=3)
    args = ap.parse_args()

    chunks = build_chunks(load_documents(args.docs))
    retriever = Retriever(chunks, persist_dir=".chroma_eval")
    questions = json.load(open("eval/questions.json"))
    print(f"{len(chunks)} chunks | {len(questions)} labelled questions | k={args.k}\n")
    print(f"{'mode':<10} {'Hit@'+str(args.k):>8} {'MRR':>8}")
    for mode in ("bm25", "vector", "hybrid"):
        hit, mrr = evaluate(retriever, questions, mode, args.k)
        print(f"{mode:<10} {hit:>8.2f} {mrr:>8.2f}")

    oos = json.load(open("eval/out_of_scope.json"))
    declined = sum(is_out_of_scope(retriever.search(q, "hybrid", 1)[1]) for q in oos)
    print(f"\nGuardrail declined {declined}/{len(oos)} out-of-scope questions (before any LLM call)")


if __name__ == "__main__":
    main()
