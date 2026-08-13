"""
Document Q&A Pipeline — YOUR WORK GOES HERE.

The knowledge base (loading, chunking, vector store) is already built
for you in knowledge_base.py. Your job is to:

  1. Retrieve relevant chunks and generate an answer
  2. Wire it up into an interactive CLI

Useful docs:
  - Vector store search: https://python.langchain.com/docs/how_to/vectorstores/
  - HuggingFace pipelines: https://python.langchain.com/docs/integrations/llms/huggingface_pipelines/
"""

import argparse
import os
import sys
from typing import Any, Callable, Dict, List

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from src.knowledge_base import build_knowledge_base


# ──────────────────────────────────────────────
# Provided: local LLM (no API key needed)
# ──────────────────────────────────────────────
def get_llm() -> Callable[[str], List[Dict[str, str]]]:
    """Return a callable local LLM using flan-t5-base.

    Downloads ~1GB on first run, then cached.
    Usage:
        llm = get_llm()
        result = llm("What color is the sky?")
        print(result[0]["generated_text"])  # "blue"
    """
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")

    def generate(prompt):
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        outputs = model.generate(**inputs, max_new_tokens=150)
        text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return [{"generated_text": text}]

    return generate


# ──────────────────────────────────────────────
# Provided: prompt template
# ──────────────────────────────────────────────
PROMPT_TEMPLATE = """You are a helpful assistant for a marketing agency. Use the following context to answer the client's question.
If the answer is not in the context, say "I don't have enough information to answer that."

Context:
{context}

Client question: {question}

Answer:"""


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TODO 1: Implement ask_question
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def ask_question(
    vector_store: Any,
    llm: Callable[[str], List[Dict[str, str]]],
    question: str,
) -> Dict[str, Any]:
    """Retrieve relevant chunks and generate an answer.

    Steps:
      1. Use vector_store.similarity_search(question, k=3) to get
         the top 3 most relevant document chunks.
      2. Combine the chunk text into a single context string.
         (Hint: each chunk has a .page_content attribute)
      3. Format the PROMPT_TEMPLATE with the context and question.
      4. Pass the formatted prompt to llm(...) and extract the
         generated text from the result.

    Args:
        vector_store: FAISS vector store from knowledge_base.py
        llm: Callable from get_llm()
        question: The user's question string

    Returns:
        dict with two keys:
            "answer"  -> str: the generated answer
            "sources" -> list[str]: the chunk texts that were retrieved
    """
    docs = vector_store.similarity_search(question, k=3)

    sources = [doc.page_content for doc in docs]

    context = "\n\n".join(sources)
    prompt = PROMPT_TEMPLATE.format(context=context, question=question)

    result = llm(prompt)
    answer = result[0]["generated_text"]

    return {"answer": answer, "sources": sources}


# ──────────────────────────────────────────────
# Bonus: friendly error handling + CLI plumbing
# ──────────────────────────────────────────────
def _build_knowledge_base_or_exit(data_dir: str) -> Any:
    """Build the knowledge base, exiting with a clear message instead of a
    raw traceback if the data directory is missing or empty."""
    if not os.path.isdir(data_dir):
        print(f"Error: data directory not found at '{data_dir}'.")
        print("Make sure a data/ folder with .txt files exists next to src/.")
        sys.exit(1)

    txt_files = [f for f in os.listdir(data_dir) if f.endswith(".txt")]
    if not txt_files:
        print(f"Error: no .txt files found in '{data_dir}'.")
        print("Add at least one .txt file to the data/ directory and try again.")
        sys.exit(1)

    try:
        return build_knowledge_base(data_dir)
    except Exception as exc:
        print(f"Error: failed to build the knowledge base: {exc}")
        sys.exit(1)


def _build_arg_parser() -> argparse.ArgumentParser:
    """Separated out so tests can exercise CLI parsing without touching models."""
    parser = argparse.ArgumentParser(
        description="Interactive Q&A chatbot over a marketing agency's docs."
    )
    parser.add_argument(
        "--query",
        "-q",
        type=str,
        default=None,
        help="Ask a single question and print the answer, instead of "
        "starting the interactive loop.",
    )
    return parser


def _print_result(result: Dict[str, Any]) -> None:
    print("\n📄 Sources:")
    for i, source in enumerate(result["sources"], start=1):
        snippet = source.replace("\n", " ").strip()
        if len(snippet) > 200:
            snippet = snippet[:200] + "..."
        print(f"  {i}. {snippet}")

    print(f"\n💬 Answer: {result['answer']}\n")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TODO 2: Complete the interactive loop
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def main() -> None:
    """Interactive Q&A loop (or single-question mode via --query).

    Steps:
      1. Build the knowledge base using build_knowledge_base()
         with the data/ directory path.
      2. Load the LLM using get_llm().
      3. Start a loop that:
         - Prompts the user for a question with input()
         - Exits if they type "quit"
         - Calls ask_question() with their input
         - Prints the retrieved sources and the answer
    """
    args = _build_arg_parser().parse_args()

    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")

    vector_store = _build_knowledge_base_or_exit(data_dir)

    llm = get_llm()

    # --query "..." mode: answer one question and exit, no interactive loop.
    if args.query is not None:
        question = args.query.strip()
        if not question:
            print("Error: --query cannot be empty.")
            sys.exit(1)
        result = ask_question(vector_store, llm, question)
        _print_result(result)
        return

    print("Ask a question about our services, pricing, or process.")
    print("Type 'quit' to exit.\n")

    while True:
        question = input("> ").strip()

        if not question:
            continue

        if question.lower() == "quit":
            print("Goodbye!")
            break

        result = ask_question(vector_store, llm, question)
        _print_result(result)


if __name__ == "__main__":
    main()