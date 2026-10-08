"""Dependency-free tests for chunking, fusion, reranking and the guardrail."""
from unibot.bot import build_prompt, is_out_of_scope
from unibot.chunking import chunk_text
from unibot.retrieval import reciprocal_rank_fusion, rerank, tokenize


def test_chunk_overlap_and_coverage():
    text = " ".join(str(i) for i in range(100))
    chunks = chunk_text(text, size=40, overlap=10)
    assert chunks[0].split()[-10:] == chunks[1].split()[:10]
    assert chunks[-1].split()[-1] == "99"


def test_short_text_single_chunk():
    assert chunk_text("hello world", 120, 30) == ["hello world"]


def test_rrf_prefers_docs_ranked_well_by_both():
    scores = reciprocal_rank_fusion([["a", "b", "c"], ["b", "a", "d"]])
    assert max(scores, key=scores.get) in {"a", "b"} and scores["a"] > scores["c"] > scores["d"] - 1


def test_rerank_uses_both_signals():
    order = rerank({"a": 9, "b": 1}, {"a": 0.2, "b": 0.9}, ["a", "b"], w_vec=0.8)
    assert order[0] == "b"


def test_guardrail_threshold():
    assert is_out_of_scope({"top_vector_similarity": 0.1}, 0.3)
    assert not is_out_of_scope({"top_vector_similarity": 0.6}, 0.3)


def test_prompt_numbers_passages():
    p = build_prompt("Q?", [{"source": "x.md", "text": "T1"}, {"source": "y.md", "text": "T2"}])
    assert "[1] (source: x.md)" in p and "[2] (source: y.md)" in p


def test_tokenize():
    assert tokenize("Hello, World-1!") == ["hello", "world", "1"]
