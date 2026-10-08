"""Question answering with citations and an out-of-scope guardrail."""
import os

from .retrieval import Retriever

MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
MIN_SIMILARITY = float(os.environ.get("UNIBOT_MIN_SIMILARITY", "0.30"))
DECLINE = ("I can only answer questions about the university documents I've been given, "
           "and I couldn't find anything relevant to that. Try rephrasing, or ask the relevant office.")

SYSTEM = (
    "You are UniBot, a university assistant. Answer ONLY from the numbered context passages. "
    "Cite every claim with its passage number like [1]. If the passages do not contain the answer, "
    "reply exactly: NOT_IN_CONTEXT. Do not use outside knowledge."
)


def is_out_of_scope(diag: dict, min_similarity: float = MIN_SIMILARITY) -> bool:
    """Guardrail 1 (before calling the LLM): nothing in the corpus is semantically close."""
    return diag["top_vector_similarity"] < min_similarity


def build_prompt(question: str, chunks: list[dict]) -> str:
    ctx = "\n\n".join(f"[{i}] (source: {c['source']})\n{c['text']}" for i, c in enumerate(chunks, 1))
    return f"Context passages:\n{ctx}\n\nQuestion: {question}"


class UniBot:
    def __init__(self, retriever: Retriever, mode: str = "hybrid", k: int = 4):
        from groq import Groq  # needs GROQ_API_KEY in the environment
        self.retriever, self.mode, self.k = retriever, mode, k
        self.client = Groq(api_key=os.environ["GROQ_API_KEY"])

    def ask(self, question: str) -> dict:
        chunks, diag = self.retriever.search(question, self.mode, self.k)
        if is_out_of_scope(diag):
            return {"answer": DECLINE, "sources": [], "declined": True}
        resp = self.client.chat.completions.create(
            model=MODEL, temperature=0,
            messages=[{"role": "system", "content": SYSTEM},
                      {"role": "user", "content": build_prompt(question, chunks)}],
        )
        text = resp.choices[0].message.content.strip()
        if "NOT_IN_CONTEXT" in text:  # Guardrail 2: the model itself found no support
            return {"answer": DECLINE, "sources": [], "declined": True}
        sources = sorted({c["source"] for c in chunks})
        return {"answer": text, "sources": sources, "declined": False}
