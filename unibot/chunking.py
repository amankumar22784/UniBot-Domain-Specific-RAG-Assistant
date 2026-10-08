"""Load documents and split them into overlapping word-based chunks."""
from pathlib import Path


def load_documents(folder: str) -> dict[str, str]:
    """Return {filename: text} for .txt/.md/.pdf files in a folder."""
    docs = {}
    for path in sorted(Path(folder).glob("*")):
        if path.suffix.lower() in {".txt", ".md"}:
            docs[path.name] = path.read_text(encoding="utf-8")
        elif path.suffix.lower() == ".pdf":
            from pypdf import PdfReader  # lazy: only needed for PDFs
            docs[path.name] = "\n".join(p.extract_text() or "" for p in PdfReader(str(path)).pages)
    return docs


def chunk_text(text: str, size: int = 120, overlap: int = 30) -> list[str]:
    """Split text into chunks of ~`size` words with `overlap` words shared between neighbours."""
    if overlap >= size:
        raise ValueError("overlap must be smaller than size")
    words = text.split()
    step = size - overlap
    chunks = []
    for start in range(0, max(len(words), 1), step):
        piece = words[start:start + size]
        if piece:
            chunks.append(" ".join(piece))
        if start + size >= len(words):
            break
    return chunks


def build_chunks(docs: dict[str, str], size: int = 120, overlap: int = 30) -> list[dict]:
    out = []
    for name, text in docs.items():
        for i, c in enumerate(chunk_text(text, size, overlap)):
            out.append({"id": f"{name}::{i}", "source": name, "chunk_no": i, "text": c})
    return out
