"""BM25, vector, and hybrid retrieval over the same chunk list."""
import re
import venv

TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return TOKEN.findall(text.lower())


def reciprocal_rank_fusion(rankings: list[list[str]], k: int = 60) -> dict[str, float]:
    """Combine several ranked id lists. Score = sum of 1 / (k + rank)."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return scores


def _minmax(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return {}
    lo, hi = min(scores.values()), max(scores.values())
    return {k: (v - lo) / (hi - lo) if hi > lo else 1.0 for k, v in scores.items()}


def rerank(bm25_scores: dict[str, float], vec_scores: dict[str, float],
           candidates: list[str], w_vec: float = 0.6) -> list[str]:
    """Re-order fused candidates by a weighted blend of normalised BM25 and vector scores."""
    b, v = _minmax(bm25_scores), _minmax(vec_scores)
    blended = {c: w_vec * v.get(c, 0.0) + (1 - w_vec) * b.get(c, 0.0) for c in candidates}
    return sorted(candidates, key=lambda c: blended[c], reverse=True)


class Retriever:
    def __init__(self, chunks: list[dict], persist_dir: str = ".chroma", collection: str = "unibot"):
        
        import chromadb
        from rank_bm25 import BM25Okapi

        self.chunks = chunks
        self.by_id = {c["id"]: c for c in chunks}
        self.bm25 = BM25Okapi([tokenize(c["text"]) for c in chunks])

        client = chromadb.PersistentClient(path=persist_dir)
        client.delete_collection(collection) if collection in [c.name for c in client.list_collections()] else None
        self.col = client.create_collection(collection, metadata={"hnsw:space": "cosine"})
        self.col.add(ids=[c["id"] for c in chunks], documents=[c["text"] for c in chunks],
                     metadatas=[{"source": c["source"]} for c in chunks])

    # Each search returns (ids ranked best-first, {id: raw score}) for `n` results.
    def bm25_search(self, query: str, n: int = 20):
        scores = self.bm25.get_scores(tokenize(query))
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:n]
        return [self.chunks[i]["id"] for i in order], {self.chunks[i]["id"]: float(scores[i]) for i in order}

    def vector_search(self, query: str, n: int = 20):
        res = self.col.query(query_texts=[query], n_results=min(n, len(self.chunks)))
        ids = res["ids"][0]
        sims = {i: 1.0 - d for i, d in zip(ids, res["distances"][0])}  # cosine similarity
        return ids, sims

    def search(self, query: str, mode: str = "hybrid", k: int = 4):
        """mode: 'bm25' | 'vector' | 'hybrid'. Returns (chunks, diagnostics)."""
        b_ids, b_sc = self.bm25_search(query)
        v_ids, v_sc = self.vector_search(query)
        if mode == "bm25":
            ids = b_ids
        elif mode == "vector":
            ids = v_ids
        elif mode == "hybrid":
            fused = reciprocal_rank_fusion([b_ids, v_ids])
            candidates = sorted(fused, key=fused.get, reverse=True)[:20]
            ids = rerank(b_sc, v_sc, candidates)
        else:
            raise ValueError(f"unknown mode: {mode}")
        diag = {"top_vector_similarity": max(v_sc.values(), default=0.0),
                "top_bm25_score": max(b_sc.values(), default=0.0)}
        return [self.by_id[i] for i in ids[:k]], diag
