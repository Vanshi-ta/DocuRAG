"""
Retrieval evaluation for DocuRAG.

Measures exactly two things automatically, both defined precisely below.
It does NOT compute or print any "faithfulness" or "groundedness" score —
that would require either a human labeler or a separate NLI/grounding
model, neither of which this script has. See --generate-groundedness-csv
for a way to produce that measurement manually, and docs/EVALUATION.md for
the full methodology.

METRICS COMPUTED
-----------------
1. Top-k retrieval accuracy (k = 1, 3, 5), over every question whose
   `type` is NOT "unsupported":

       hit@k(question) = 1 if AT LEAST ONE of the question's
                            `relevant_sources` (source_filename,
                            page_number) pairs appears among the
                            top-k chunks retrieved for that question,
                            else 0.

       Top-k accuracy = mean(hit@k) over all answerable questions.

   A chunk "matches" a relevant_source if the chunk's source_filename AND
   page_number both equal the ground-truth pair. This is a page-level
   match, not an exact-chunk match, because a page can be split into
   multiple overlapping chunks and any of them containing the answer
   counts as a successful retrieval.

2. Correct-rejection rate, over every question whose `type` IS
   "unsupported" (relevant_sources == []):

       rejected(question) = 1 if the top-1 retrieved chunk's similarity
                               score is below SIMILARITY_THRESHOLD (i.e.
                               the system would produce no chunks and
                               therefore skip the LLM / answer "I could
                               not find this information"), else 0.

       Correct-rejection rate = mean(rejected) over unsupported questions.

Both metrics run against whatever documents are CURRENTLY indexed — this
script does not ingest anything itself. Run `python -m src.pipeline` (or
use the app) to index your documents first.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

sys.path.append(str(Path(__file__).resolve().parents[1]))

from config import CANDIDATE_POOL_SIZE, DEFAULT_TOP_K, MAX_CHUNKS_PER_SOURCE, SIMILARITY_THRESHOLD
from src.generation.llm_client import OllamaClient
from src.generation.rag_engine import answer_question
from src.ingestion.embedder import Embedder
from src.logging_config import configure_logging
from src.pipeline import load_vector_store
from src.retrieval.retriever import Retriever, retrieve_for_question

DEFAULT_DATASET_PATH = Path(__file__).resolve().parents[1] / "data" / "eval" / "eval_dataset.json"
DEFAULT_RESULTS_DIR = Path(__file__).resolve().parents[1] / "eval_results"
K_VALUES = (1, 3, 5)


@dataclass
class QuestionResult:
    id: str
    question: str
    type: str
    retrieved: List[Tuple[str, int, float]]  # (filename, page, score), best-first
    hit_at_k: dict  # {1: bool, 3: bool, 5: bool} -- only for answerable questions
    correctly_rejected: bool | None  # only for "unsupported" questions


def load_dataset(path: Path) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload["questions"]


def source_matches(retrieved_filename: str, retrieved_page: int, ground_truth: List[dict]) -> bool:
    return any(
        retrieved_filename == gt["source_filename"] and retrieved_page == gt["page_number"]
        for gt in ground_truth
    )


def evaluate_question(question_entry: dict, retriever: Retriever, max_k: int) -> QuestionResult:
    qid = question_entry["id"]
    question_text = question_entry["question"]
    qtype = question_entry["type"]
    ground_truth = question_entry.get("relevant_sources", [])

    # Uses the SAME retrieval strategy the app uses to answer questions
    # (two-stage diversity selection + entity fan-out for multi-entity
    # questions) — not the plain single-stage baseline — so these metrics
    # reflect what users actually experience. similarity_threshold=None
    # here so we get the full ranked candidate list to compute hit@k at
    # every k value; the rejection-rate check below applies the threshold
    # itself, independent of the live app setting.
    chunks = retrieve_for_question(
        question_text, retriever, top_k=max_k,
        candidate_pool_size=CANDIDATE_POOL_SIZE, max_chunks_per_source=MAX_CHUNKS_PER_SOURCE,
        similarity_threshold=None,
    )
    retrieved = [(c.source_filename, c.page_number, c.similarity_score) for c in chunks]

    hit_at_k = {}
    correctly_rejected = None

    if qtype == "unsupported":
        top_score = retrieved[0][2] if retrieved else float("-inf")
        threshold = SIMILARITY_THRESHOLD if SIMILARITY_THRESHOLD is not None else float("inf")
        correctly_rejected = top_score < threshold
    else:
        for k in K_VALUES:
            top_k_slice = retrieved[:k]
            hit = any(source_matches(f, p, ground_truth) for f, p, _ in top_k_slice)
            hit_at_k[k] = hit

    return QuestionResult(
        id=qid, question=question_text, type=qtype,
        retrieved=retrieved, hit_at_k=hit_at_k, correctly_rejected=correctly_rejected,
    )


def print_report(results: List[QuestionResult]) -> None:
    answerable = [r for r in results if r.type != "unsupported"]
    unsupported = [r for r in results if r.type == "unsupported"]

    print("\n" + "=" * 70)
    print("DOCURAG RETRIEVAL EVALUATION")
    print("=" * 70)
    print(f"Total questions: {len(results)}  "
          f"(answerable: {len(answerable)}, unsupported: {len(unsupported)})\n")

    if answerable:
        print(f"{'k':>4} | {'Top-k accuracy':>16} | {'hits':>6} / {'total':<6}")
        print("-" * 40)
        for k in K_VALUES:
            hits = sum(1 for r in answerable if r.hit_at_k.get(k))
            acc = hits / len(answerable)
            print(f"{k:>4} | {acc:>15.1%} | {hits:>6} / {len(answerable):<6}")

        # Breakdown by question type, since multi_doc/multi_chunk questions
        # are expected to be harder than direct ones — collapsing them into
        # one number would hide that.
        print("\nBreakdown by question type (Top-{} accuracy):".format(max(K_VALUES)))
        for qtype in ("direct", "multi_chunk", "multi_doc"):
            subset = [r for r in answerable if r.type == qtype]
            if not subset:
                continue
            hits = sum(1 for r in subset if r.hit_at_k.get(max(K_VALUES)))
            print(f"  {qtype:<14} {hits}/{len(subset)} ({hits / len(subset):.1%})")
    else:
        print("No answerable (non-'unsupported') questions in the dataset.")

    if unsupported:
        rejected = sum(1 for r in unsupported if r.correctly_rejected)
        print(f"\nCorrect-rejection rate (unsupported questions): "
              f"{rejected}/{len(unsupported)} ({rejected / len(unsupported):.1%})")
    else:
        print("\nNo 'unsupported' questions in the dataset — rejection rate not measured.")

    print("=" * 70)
    print(
        "NOTE: These are RETRIEVAL metrics only (whether the right page was "
        "found). No answer-faithfulness/groundedness score is computed by "
        "this script — see docs/EVALUATION.md and --generate-groundedness-csv."
    )


def save_json_results(results: List[QuestionResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [
        {
            "id": r.id,
            "question": r.question,
            "type": r.type,
            "retrieved": [{"source_filename": f, "page_number": p, "similarity_score": s} for f, p, s in r.retrieved],
            "hit_at_k": r.hit_at_k,
            "correctly_rejected": r.correctly_rejected,
        }
        for r in results
    ]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"\nPer-question results saved to: {path}")


def generate_groundedness_csv(dataset: List[dict], retriever: Retriever, llm_client: OllamaClient, path: Path) -> None:
    """
    Runs full end-to-end RAG (retrieval + generation) for every answerable
    question and writes a CSV with an empty 'grounded (Y/N)' column for a
    human to fill in by reading each answer against its cited sources.

    This script does NOT compute a groundedness score itself — labeling
    requires human judgment (or a separate NLI model this project does not
    include). Once labeled, groundedness rate = (# marked Y) / (# labeled).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for entry in dataset:
        if entry["type"] == "unsupported":
            continue
        result = answer_question(entry["question"], retriever, llm_client, top_k=DEFAULT_TOP_K)
        sources_str = "; ".join(
            f"{s.source_filename} p.{s.page_number} ({s.similarity_score:.2f})" for s in result.sources
        )
        rows.append({
            "id": entry["id"],
            "question": entry["question"],
            "answer": result.answer,
            "sources": sources_str,
            "grounded (Y/N)": "",  # fill in by hand
            "notes": "",
        })

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "question", "answer", "sources", "grounded (Y/N)", "notes"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nGroundedness-labeling template written to: {path}")
    print("Open it, read each answer against its sources, and fill in Y/N by hand.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate DocuRAG retrieval quality.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument(
        "--generate-groundedness-csv", action="store_true",
        help="Also run full RAG generation and write a CSV template for manual groundedness labeling.",
    )
    args = parser.parse_args()

    configure_logging()

    print("Loading embedding model and vector store...")
    embedder = Embedder()
    try:
        faiss_store, metadata_store, _registry = load_vector_store(embedder.embedding_dimension)
    except FileNotFoundError:
        print(
            "No persisted vector store found. Index your documents first "
            "(run the app and click 'Process Documents', or `python -m src.pipeline`)."
        )
        return

    if faiss_store.ntotal == 0:
        print("The vector store is empty (0 chunks indexed). Nothing to evaluate.")
        return

    retriever = Retriever(embedder, faiss_store, metadata_store)
    dataset = load_dataset(args.dataset)
    max_k = max(K_VALUES)

    results = [evaluate_question(q, retriever, max_k) for q in dataset]
    print_report(results)
    save_json_results(results, args.output_dir / "eval_results.json")

    if args.generate_groundedness_csv:
        llm_client = OllamaClient()
        generate_groundedness_csv(
            dataset, retriever, llm_client, args.output_dir / "groundedness_template.csv"
        )


if __name__ == "__main__":
    main()
