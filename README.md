# UniBot: Domain-Specific RAG Assistant

UniBot is a retrieval-augmented question-answering assistant for university documents. It finds the most relevant passages, answers **only** from them, cites its sources, and politely declines questions that fall outside the documents.

![Python](https://img.shields.io/badge/Python-3.12-blue) ![ChromaDB](https://img.shields.io/badge/Vector%20DB-ChromaDB-orange) ![Groq](https://img.shields.io/badge/LLM-Groq%20API-green)

## Why this project

General-purpose chatbots can invent policies, such as attendance limits or fee deadlines. UniBot reduces that risk by grounding every answer in the university's own documents and refusing when it cannot find support.

## Features

- **Hybrid retrieval:** combines keyword search (BM25) with semantic search (ChromaDB embeddings) using Reciprocal Rank Fusion.
- **Reranking:** reorders the fused candidates with a weighted blend of normalised BM25 and vector scores.
- **Cited answers:** every claim points to a numbered source passage, and the file names are listed after each answer.
- **Two-layer guardrail:** declines out-of-scope questions before calling the LLM (low semantic similarity) and when the model reports the context lacks the answer.
- **Measured, not assumed:** an evaluation script compares BM25-only, vector-only and hybrid retrieval on a labelled question set.
- **Multiple formats:** loads `.md`, `.txt` and `.pdf` documents.

## Architecture

```
Documents (.md / .txt / .pdf)
        │
        ▼
 Chunking (~120 words, 30-word overlap)
        │
        ├──────────────► BM25 index
        └──────────────► ChromaDB (embeddings, cosine)
                                │
 User question ─► BM25 + vector search ─► Reciprocal Rank Fusion ─► Rerank
                                                                      │
                          Guardrail 1: similarity below threshold? ───┤ decline
                                                                      ▼
                                      Groq LLM (answers only from numbered passages)
                                                                      │
                          Guardrail 2: model says NOT_IN_CONTEXT? ────┤ decline
                                                                      ▼
                                              Answer with [1] citations + sources
```

## Tech stack

| Area | Tools |
|---|---|
| Language | Python |
| Vector store | ChromaDB (default embedding model, cosine similarity) |
| Keyword search | rank-bm25 |
| LLM | Groq API |
| Documents | pypdf for PDF text extraction |

## Getting started

### 1. Install

```bash
git clone https://github.com/<your-username>/unibot.git
cd unibot
python -m venv venv
```

Activate the environment:

```bash
# Windows (PowerShell)
Set-ExecutionPolicy -Scope Process Bypass
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

Then install the dependencies:

```bash
pip install -r requirements.txt
```

Python 3.11 or 3.12 is recommended. ChromaDB may not install on the newest Python versions.

### 2. Configure

Get a free API key from [console.groq.com](https://console.groq.com) and set it as an environment variable:

```bash
# Windows (PowerShell)
$env:GROQ_API_KEY="your_key_here"

# macOS / Linux
export GROQ_API_KEY="your_key_here"
```

Groq changes its available models over time, and which ones you can use depends on your account. List what your key can call:

```bash
python -c "from groq import Groq; print('\n'.join(m.id for m in Groq().models.list().data))"
```

Then choose a chat model with `GROQ_MODEL` (the default is `llama-3.3-70b-versatile`):

```bash
# Windows (PowerShell)
$env:GROQ_MODEL="your_model_id"

# macOS / Linux
export GROQ_MODEL="your_model_id"
```

Never commit your API key to the repository.

### 3. Run

```bash
python cli.py                  # start the chat (hybrid retrieval)
python cli.py --mode bm25      # BM25 only
python cli.py --mode vector    # vector search only
```

Press Enter on an empty line to quit.

### Example

```
You: What is the minimum attendance percentage required?
UniBot: Students must maintain at least 75 percent attendance in every course to
        be eligible for the end-semester examination [1].
Sources: attendance_policy.md

You: Who won the cricket world cup?
UniBot: I can only answer questions about the university documents I've been given,
        and I couldn't find anything relevant to that. Try rephrasing, or ask the
        relevant office.
```

## Using your own documents

The files in `data/docs/` are **sample policies** written for demonstration.

1. Replace them with your own documents (`.md`, `.txt` or `.pdf`).
2. Edit `eval/questions.json` so each question names the file that answers it.
3. Re-run the evaluation below and record your results.

Use `--docs path/to/folder` on `cli.py` to point at a different folder.

## Evaluation

Retrieval quality is measured on a labelled question set (`eval/questions.json`, 14 questions across 4 documents):

- **Hit@k:** the share of questions where a chunk from the correct document appears in the top *k* results.
- **MRR:** the mean reciprocal rank of the first chunk from the correct document.
- **Guardrail check:** how many out-of-scope questions (`eval/out_of_scope.json`) are declined before any LLM call.

```bash
python -m eval.run_eval --k 3
```

This evaluation needs no API key.

### Results

Fill in the table with the numbers printed on your machine:

| Retrieval method | Hit@3 | MRR |
|---|---|---|
| BM25 only | _run eval_ | _run eval_ |
| Vector only | _run eval_ | _run eval_ |
| Hybrid + rerank | _run eval_ | _run eval_ |

Guardrail: _x_ of 4 out-of-scope questions declined.

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `GROQ_API_KEY` | required | Authenticates with the Groq API |
| `GROQ_MODEL` | `llama-3.3-70b-versatile` | Chat model used to write answers |
| `UNIBOT_MIN_SIMILARITY` | `0.30` | Guardrail threshold. Raise it to decline more, lower it to decline less |

Other settings sit in the source: chunk size and overlap in `unibot/chunking.py`, and the fusion constant and rerank weight in `unibot/retrieval.py`.

## Project structure

```
unibot/
├── cli.py                 # interactive chat
├── unibot/
│   ├── chunking.py        # load documents, split into overlapping chunks
│   ├── retrieval.py       # BM25, vector search, RRF fusion, reranking
│   └── bot.py             # prompt, citations, guardrail, Groq call
├── data/docs/             # sample documents (replace with your own)
├── eval/
│   ├── questions.json     # labelled evaluation questions
│   ├── out_of_scope.json  # questions the bot should decline
│   └── run_eval.py        # retrieval comparison + guardrail check
├── tests/test_core.py     # unit tests
└── requirements.txt
```

## Testing

```bash
python -m tests.test_core
```

The tests cover chunking, rank fusion, reranking, the guardrail threshold and prompt construction. They need no extra packages.

## Limitations

- Answers are only as good as the documents provided. Outdated or missing policies produce outdated or declined answers.
- The reranker blends existing scores. It is not a trained cross-encoder, and swapping one in is a natural upgrade.
- Chunking is word-based, so it can split tables or lists awkwardly.
- It answers single questions and does not remember earlier turns in a conversation.
- The guardrail threshold may need tuning for each document set.

## Roadmap

- [ ] Cross-encoder reranking
- [ ] Web interface (Streamlit or Flask)
- [ ] Conversation memory
- [ ] Larger evaluation set with answer-quality scoring

## Author

**Aman Kumar Sahu**: B.Tech Computer Science Engineering, Siksha 'O' Anusandhan University
[LinkedIn](https://linkedin.com/in/amankumar-316282322) · amankumar22784@gmail.com

## License

Released under the MIT License. Add a `LICENSE` file to the repository if you want to use this license.
