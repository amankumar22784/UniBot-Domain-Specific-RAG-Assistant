"""Interactive command-line chat with UniBot.  Usage: python cli.py [--mode hybrid|bm25|vector]"""
import argparse
import venv
from unibot.bot import UniBot
from unibot.chunking import build_chunks, load_documents
from unibot.retrieval import Retriever


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", default="data/docs")
    ap.add_argument("--mode", default="hybrid", choices=["hybrid", "bm25", "vector"])
    args = ap.parse_args()

    retriever = Retriever(build_chunks(load_documents(args.docs)))
    bot = UniBot(retriever, mode=args.mode)
    print("UniBot ready. Ask a question (blank line to quit).")
    while (q := input("\nYou: ").strip()):
        out = bot.ask(q)
        print(f"\nUniBot: {out['answer']}")
        if out["sources"]:
            print("Sources:", ", ".join(out["sources"]))


if __name__ == "__main__":
    main()
