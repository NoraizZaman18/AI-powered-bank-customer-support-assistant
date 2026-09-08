# evaluation/evaluator.py

import os
import json
import time
from typing import List, Dict
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

GOLDEN_SET_PATH = "evaluation/golden_set.json"
RESULTS_PATH = "evaluation/eval_results.json"
MODEL_NAME = "openai/gpt-oss-20b"


# ── TOPIC 60 — WHY MEASURE RAG QUALITY ───────────────────────────────────────

def load_golden_set() -> List[dict]:
    """
    Topic 60 — Load the golden test set.
    Golden set = questions with known correct answers.
    """

    with open(GOLDEN_SET_PATH, "r") as f:
        data = json.load(f)

    questions = data["questions"]
    print(f"Loaded {len(questions)} test questions from golden set")
    return questions


# ── TOPIC 61 — RAGAS SETUP ────────────────────────────────────────────────────

def setup_ragas():
    """
    Topic 61 — Set up RAGAS evaluation framework.
    pip install ragas
    """

    try:
        from ragas import evaluate
        from ragas.metrics import (
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
            answer_correctness
        )
        from ragas.llms import LangchainLLMWrapper
        from ragas.embeddings import LangchainEmbeddingsWrapper

        print("RAGAS framework loaded successfully")
        return True

    except ImportError:
        print("RAGAS not installed. Run: pip install ragas")
        return False


# ── TOPICS 62 TO 66 — RAGAS METRICS ──────────────────────────────────────────

def run_ragas_evaluation(
    questions: List[str],
    answers: List[str],
    contexts: List[List[str]],
    ground_truths: List[str],
    sample_size: int = None
) -> dict:
    """
    Topics 62 to 66 — Run RAGAS evaluation.

    Computes:
    - Faithfulness (topic 62)
    - Answer Relevance (topic 63)
    - Context Precision (topic 64)
    - Context Recall (topic 65)
    - Answer Correctness (topic 66)

    questions: list of user questions
    answers: list of generated answers
    contexts: list of lists — retrieved chunks for each question
    ground_truths: list of known correct answers
    sample_size: run on subset if specified
    """

    from ragas import evaluate
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
        answer_correctness
    )
    from datasets import Dataset
    from langchain_groq import ChatGroq

    if sample_size:
        questions = questions[:sample_size]
        answers = answers[:sample_size]
        contexts = contexts[:sample_size]
        ground_truths = ground_truths[:sample_size]

    print(f"\nRunning RAGAS on {len(questions)} questions...")

    data = {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths
    }

    dataset = Dataset.from_dict(data)

    llm = ChatGroq(
        model=MODEL_NAME,
        temperature=0,
        api_key=os.getenv("GROQ_API_KEY")
    )

    result = evaluate(
        dataset=dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
            answer_correctness
        ],
        llm=llm
    )

    scores = {
        "faithfulness": round(float(result["faithfulness"]), 3),
        "answer_relevancy": round(float(result["answer_relevancy"]), 3),
        "context_precision": round(float(result["context_precision"]), 3),
        "context_recall": round(float(result["context_recall"]), 3),
        "answer_correctness": round(float(result["answer_correctness"]), 3),
    }

    return scores


# ── TOPIC 67 — LLM AS JUDGE ───────────────────────────────────────────────────

def llm_judge_single(
    question: str,
    answer: str,
    context: str,
    ground_truth: str
) -> dict:
    """
    Topic 67 — Use LLM to judge a single question-answer pair.
    Faster than RAGAS for quick checks during development.
    """

    from langchain_groq import ChatGroq
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    import json as json_module

    llm = ChatGroq(
        model=MODEL_NAME,
        temperature=0,
        api_key=os.getenv("GROQ_API_KEY")
    )

    judge_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert RAG system evaluator.

Evaluate the answer based on the question, context, and ground truth.

Score each dimension from 0.0 to 1.0:
- faithfulness: Is every claim in the answer supported by the context?
- relevance: Does the answer address the question asked?
- correctness: Does the answer match the ground truth?

Return ONLY a valid JSON object with these exact keys:
{{"faithfulness": 0.0, "relevance": 0.0, "correctness": 0.0, "explanation": ""}}"""),
        ("human", """Question: {question}

Context:
{context}

Ground Truth: {ground_truth}

Answer to evaluate: {answer}""")
    ])

    chain = judge_prompt | llm | StrOutputParser()

    raw = chain.invoke({
        "question": question,
        "context": context,
        "ground_truth": ground_truth,
        "answer": answer
    })

    try:
        raw_clean = raw.strip()
        if raw_clean.startswith("```"):
            raw_clean = raw_clean.split("```")[1]
            if raw_clean.startswith("json"):
                raw_clean = raw_clean[4:]
        scores = json_module.loads(raw_clean.strip())
    except Exception:
        scores = {
            "faithfulness": 0.0,
            "relevance": 0.0,
            "correctness": 0.0,
            "explanation": "Parse error"
        }

    return scores


def run_llm_judge_evaluation(
    test_items: List[dict],
    pipeline_fn,
    sample_size: int = 10
) -> dict:
    """
    Topic 67 — Run LLM judge evaluation on a sample of questions.
    Faster than RAGAS — good for quick development checks.
    """

    sample = test_items[:sample_size]
    all_scores = []

    print(f"\nLLM Judge evaluation on {len(sample)} questions...")
    print("─" * 50)

    for i, item in enumerate(sample, start=1):
        question = item["question"]
        ground_truth = item["ground_truth"]

        result = pipeline_fn(question)
        answer = result.get("answer", "")
        context = result.get("context_used", "")

        scores = llm_judge_single(question, answer, context, ground_truth)
        all_scores.append(scores)

        print(f"  Q{i:02d}: F={scores.get('faithfulness', 0):.2f} | "
              f"R={scores.get('relevance', 0):.2f} | "
              f"C={scores.get('correctness', 0):.2f} | "
              f"{question[:45]}...")

    avg_scores = {
        "faithfulness": round(
            sum(s.get("faithfulness", 0) for s in all_scores) / len(all_scores), 3
        ),
        "relevance": round(
            sum(s.get("relevance", 0) for s in all_scores) / len(all_scores), 3
        ),
        "correctness": round(
            sum(s.get("correctness", 0) for s in all_scores) / len(all_scores), 3
        ),
        "questions_evaluated": len(all_scores)
    }

    return avg_scores


# ── TOPIC 68 — BUILDING THE GOLDEN TEST SET ───────────────────────────────────

def validate_golden_set(golden_set: List[dict]) -> dict:
    """
    Topic 68 — Validate that the golden test set is complete.
    Checks that all required fields are present.
    """

    required_fields = ["id", "question", "ground_truth", "expected_source"]
    issues = []

    for item in golden_set:
        for field in required_fields:
            if field not in item or not item[field]:
                issues.append(
                    f"Question {item.get('id', '?')}: missing '{field}'"
                )

    report = {
        "total_questions": len(golden_set),
        "valid": len(issues) == 0,
        "issues": issues
    }

    if report["valid"]:
        print(f"Golden set validation: PASS — {len(golden_set)} questions")
    else:
        print(f"Golden set validation: ISSUES FOUND")
        for issue in issues:
            print(f"  {issue}")

    return report


# ── TOPIC 69 — RUNNING RAGAS EVALUATION ──────────────────────────────────────

def run_full_evaluation(
    pipeline_fn,
    use_ragas: bool = False,
    sample_size: int = 10
) -> dict:
    """
    Topic 69 — Run the complete evaluation pipeline.

    pipeline_fn: function that takes a question and returns result dict
                 with keys: answer, context_used, sources

    use_ragas: True for full RAGAS evaluation
               False for faster LLM judge evaluation

    sample_size: number of questions to evaluate
    """

    print(f"\n{'═'*60}")
    print(f"  EVALUATION PIPELINE")
    print(f"{'═'*60}")
    print(f"  Method: {'RAGAS' if use_ragas else 'LLM Judge'}")
    print(f"  Sample size: {sample_size} questions")

    golden_set = load_golden_set()
    validate_golden_set(golden_set)

    start_time = time.time()

    if use_ragas:
        questions = []
        answers = []
        contexts = []
        ground_truths = []

        sample = golden_set[:sample_size]

        print(f"\nRunning pipeline on {len(sample)} questions...")

        for i, item in enumerate(sample, start=1):
            question = item["question"]
            ground_truth = item["ground_truth"]

            print(f"  [{i:02d}/{len(sample)}] {question[:55]}...")

            result = pipeline_fn(question)

            questions.append(question)
            answers.append(result.get("answer", ""))
            ground_truths.append(ground_truth)

            context_chunks = result.get("context_used", "")
            if isinstance(context_chunks, str):
                contexts.append([context_chunks])
            else:
                contexts.append(context_chunks)

            time.sleep(0.5)

        scores = run_ragas_evaluation(
            questions, answers, contexts, ground_truths
        )

    else:
        scores = run_llm_judge_evaluation(
            golden_set, pipeline_fn, sample_size
        )

    elapsed = round(time.time() - start_time, 1)

    results = {
        "timestamp": datetime.now().isoformat(),
        "method": "ragas" if use_ragas else "llm_judge",
        "sample_size": sample_size,
        "elapsed_seconds": elapsed,
        "scores": scores
    }

    save_eval_results(results)
    print_eval_results(results)

    return results


# ── TOPIC 70 — INTERPRETING RAGAS SCORES ─────────────────────────────────────

def interpret_scores(scores: dict) -> dict:
    """
    Topic 70 — Interpret RAGAS scores and give actionable recommendations.

    Score thresholds:
    - Above 0.8: production ready
    - 0.6 to 0.8: needs improvement
    - Below 0.6: significant problems
    """

    thresholds = {
        "faithfulness": {
            "good": 0.8,
            "fix": "Add stricter grounding instructions to system prompt. "
                   "Use STRICT_GROUNDING_PROMPT."
        },
        "answer_relevancy": {
            "good": 0.8,
            "fix": "Improve prompt to be more focused. "
                   "Check if questions match document content."
        },
        "context_precision": {
            "good": 0.7,
            "fix": "Retrieval is returning irrelevant chunks. "
                   "Add reranking or improve metadata filtering."
        },
        "context_recall": {
            "good": 0.7,
            "fix": "Missing relevant chunks. "
                   "Try multi-query retrieval or increase k value."
        },
        "correctness": {
            "good": 0.7,
            "fix": "Answers are factually wrong. "
                   "Check document quality and chunking strategy."
        },
        "relevance": {
            "good": 0.8,
            "fix": "Answers are off-topic. "
                   "Improve system prompt and routing logic."
        }
    }

    interpretations = {}

    for metric, score in scores.items():
        if metric in ("questions_evaluated", "elapsed_seconds"):
            continue

        threshold_info = thresholds.get(metric, {"good": 0.7, "fix": "Review pipeline"})
        good_threshold = threshold_info["good"]

        if score >= good_threshold:
            status = "GOOD"
            recommendation = "No action needed"
        elif score >= good_threshold - 0.15:
            status = "NEEDS IMPROVEMENT"
            recommendation = threshold_info["fix"]
        else:
            status = "CRITICAL"
            recommendation = f"URGENT: {threshold_info['fix']}"

        interpretations[metric] = {
            "score": score,
            "status": status,
            "threshold": good_threshold,
            "recommendation": recommendation
        }

    return interpretations


def print_eval_results(results: dict):
    """Print clean evaluation results to terminal."""

    scores = results["scores"]
    interpretations = interpret_scores(scores)

    print(f"\n{'═'*60}")
    print(f"  EVALUATION RESULTS")
    print(f"{'═'*60}")
    print(f"  Method:    {results['method'].upper()}")
    print(f"  Questions: {results['sample_size']}")
    print(f"  Time:      {results['elapsed_seconds']}s")
    print(f"{'─'*60}")

    for metric, info in interpretations.items():
        score = info["score"]
        status = info["status"]

        if status == "GOOD":
            icon = "✓"
        elif status == "NEEDS IMPROVEMENT":
            icon = "~"
        else:
            icon = "✗"

        bar_len = int(score * 20)
        bar = "█" * bar_len + "░" * (20 - bar_len)

        print(f"\n  {icon} {metric.upper()}")
        print(f"    Score:  {score:.3f}  [{bar}]")
        print(f"    Status: {status}")

        if status != "GOOD":
            print(f"    Fix:    {info['recommendation']}")

    print(f"\n{'═'*60}")


# ── TOPIC 71 — REGRESSION TESTING ────────────────────────────────────────────

def save_eval_results(results: dict):
    """Save evaluation results to disk for regression tracking."""

    existing = []

    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH, "r") as f:
            existing = json.load(f)

    existing.append(results)

    with open(RESULTS_PATH, "w") as f:
        json.dump(existing, f, indent=2)

    print(f"\n  Results saved to {RESULTS_PATH}")


def check_regression(
    current_scores: dict,
    tolerance: float = 0.05
) -> dict:
    """
    Topic 71 — Compare current scores to previous run.
    Detects regressions — metrics that dropped significantly.

    tolerance: how much a score can drop before flagging as regression
               0.05 = flag if any metric drops more than 5 points
    """

    if not os.path.exists(RESULTS_PATH):
        print("No previous results found — this is the baseline run")
        return {"regression_detected": False, "details": []}

    with open(RESULTS_PATH, "r") as f:
        history = json.load(f)

    if len(history) < 2:
        print("Only one result in history — no regression check possible")
        return {"regression_detected": False, "details": []}

    previous = history[-2]["scores"]
    regressions = []

    for metric, current_score in current_scores.items():
        if metric in ("questions_evaluated", "elapsed_seconds"):
            continue

        prev_score = previous.get(metric, 0)
        drop = prev_score - current_score

        if drop > tolerance:
            regressions.append({
                "metric": metric,
                "previous": prev_score,
                "current": current_score,
                "drop": round(drop, 3)
            })

    if regressions:
        print(f"\n  REGRESSION DETECTED — {len(regressions)} metric(s) dropped:")
        for r in regressions:
            print(
                f"    {r['metric']}: {r['previous']:.3f} → "
                f"{r['current']:.3f} (dropped {r['drop']:.3f})"
            )
    else:
        print("\n  No regression detected — scores stable or improved")

    return {
        "regression_detected": len(regressions) > 0,
        "regressions": regressions
    }


def show_score_history():
    """
    Topic 71 — Show score history across all evaluation runs.
    Helps you see if your changes are improving the system over time.
    """

    if not os.path.exists(RESULTS_PATH):
        print("No evaluation history found")
        return

    with open(RESULTS_PATH, "r") as f:
        history = json.load(f)

    print(f"\n{'═'*60}")
    print(f"  EVALUATION HISTORY ({len(history)} runs)")
    print(f"{'═'*60}")

    metrics = ["faithfulness", "answer_relevancy", "context_precision",
               "context_recall", "correctness", "relevance"]

    for i, run in enumerate(history, start=1):
        timestamp = run.get("timestamp", "unknown")[:16]
        method = run.get("method", "unknown")
        scores = run.get("scores", {})

        print(f"\n  Run {i} | {timestamp} | {method}")
        print(f"  {'─'*40}")

        for metric in metrics:
            score = scores.get(metric)
            if score is not None:
                bar_len = int(score * 15)
                bar = "█" * bar_len + "░" * (15 - bar_len)
                print(f"    {metric:25s}: {score:.3f} [{bar}]")


# ── TOPIC 72 — A/B TESTING ────────────────────────────────────────────────────

def ab_test(
    pipeline_a_fn,
    pipeline_b_fn,
    pipeline_a_name: str,
    pipeline_b_name: str,
    sample_size: int = 10
) -> dict:
    """
    Topic 72 — A/B test two different RAG configurations.
    Runs both pipelines on the same questions and compares scores.

    Use this to compare:
    - Different chunking strategies
    - Different retrieval methods
    - Different prompt templates
    - Different reranking settings

    pipeline_a_fn: first pipeline function
    pipeline_b_fn: second pipeline function
    """

    golden_set = load_golden_set()
    sample = golden_set[:sample_size]

    print(f"\n{'═'*60}")
    print(f"  A/B TEST")
    print(f"{'═'*60}")
    print(f"  A: {pipeline_a_name}")
    print(f"  B: {pipeline_b_name}")
    print(f"  Questions: {sample_size}")
    print(f"{'─'*60}")

    def evaluate_pipeline(pipeline_fn, name):
        scores_list = []
        print(f"\n  Evaluating {name}...")

        for i, item in enumerate(sample, start=1):
            question = item["question"]
            ground_truth = item["ground_truth"]

            result = pipeline_fn(question)
            answer = result.get("answer", "")
            context = result.get("context_used", "")

            scores = llm_judge_single(
                question, answer, context, ground_truth
            )
            scores_list.append(scores)

            print(
                f"    [{i:02d}] F={scores.get('faithfulness', 0):.2f} | "
                f"R={scores.get('relevance', 0):.2f} | "
                f"C={scores.get('correctness', 0):.2f}"
            )

        avg = {
            "faithfulness": round(
                sum(s.get("faithfulness", 0) for s in scores_list)
                / len(scores_list), 3
            ),
            "relevance": round(
                sum(s.get("relevance", 0) for s in scores_list)
                / len(scores_list), 3
            ),
            "correctness": round(
                sum(s.get("correctness", 0) for s in scores_list)
                / len(scores_list), 3
            )
        }

        return avg

    scores_a = evaluate_pipeline(pipeline_a_fn, pipeline_a_name)
    scores_b = evaluate_pipeline(pipeline_b_fn, pipeline_b_name)

    print(f"\n{'═'*60}")
    print(f"  A/B TEST RESULTS")
    print(f"{'═'*60}")
    print(f"\n  {'Metric':25s} {'A':>8} {'B':>8} {'Winner':>12}")
    print(f"  {'─'*55}")

    winner_counts = {"A": 0, "B": 0, "Tie": 0}

    for metric in ["faithfulness", "relevance", "correctness"]:
        score_a = scores_a.get(metric, 0)
        score_b = scores_b.get(metric, 0)
        diff = abs(score_a - score_b)

        if diff < 0.02:
            winner = "Tie"
            winner_counts["Tie"] += 1
        elif score_a > score_b:
            winner = f"A ({pipeline_a_name[:8]})"
            winner_counts["A"] += 1
        else:
            winner = f"B ({pipeline_b_name[:8]})"
            winner_counts["B"] += 1

        print(
            f"  {metric:25s} {score_a:>8.3f} {score_b:>8.3f} "
            f"{winner:>12}"
        )

    print(f"\n  Overall winner: ", end="")
    if winner_counts["A"] > winner_counts["B"]:
        print(f"A — {pipeline_a_name}")
    elif winner_counts["B"] > winner_counts["A"]:
        print(f"B — {pipeline_b_name}")
    else:
        print("Tie — no significant difference")

    return {
        "pipeline_a": {"name": pipeline_a_name, "scores": scores_a},
        "pipeline_b": {"name": pipeline_b_name, "scores": scores_b},
        "winner_counts": winner_counts
    }


# ── TEST BLOCK ────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from retrieval.store import get_vectorstore
    from retrieval.retriever import smart_retrieve
    from generation.generator import generate_rag_response
    from ingestion.loader import load_all_documents
    from ingestion.cleaner import clean_documents
    from ingestion.chunker import (
        detect_document_types,
        recommend_overlap,
        recursive_chunking,
        enrich_metadata
    )
    from retrieval.store import incremental_index

    # setup
    documents = load_all_documents("data/documents/")
    documents = clean_documents(documents)

    def incremental_chunking(docs):
        document_types = detect_document_types(docs)
        final_chunks = []
        for source, doc_type in document_types.items():
            settings = recommend_overlap(doc_type)
            source_docs = [d for d in docs if d.metadata.get("source") == source]
            chunks = recursive_chunking(source_docs, settings["chunk_size"], settings["overlap"])
            chunks = enrich_metadata(chunks)
            final_chunks.extend(chunks)
        return final_chunks

    vectorstore = incremental_index("data/documents/", incremental_chunking, False)
    all_chunks = incremental_chunking(documents)

    def run_pipeline(question: str) -> dict:
        """Pipeline function for evaluation."""
        context, sources, retrieved_docs = smart_retrieve(
            question=question,
            vectorstore=vectorstore,
            all_chunks=all_chunks,
            k=6,
            strategy="ensemble",
            apply_reorder=True,
            use_smart_filter=True,
            use_reranking=True
        )
        result = generate_rag_response(
            question=question,
            docs=retrieved_docs,
            citation_style="end"
        )
        result["context_used"] = context
        return result

    print("\n" + "="*60)
    print("SECTION 8 — EVALUATION")
    print("="*60)

    # topic 68 — validate golden set
    print("\nTOPIC 68 — VALIDATING GOLDEN SET")
    golden_set = load_golden_set()
    validate_golden_set(golden_set)

    # topic 67 — LLM judge on 5 questions
    print("\nTOPIC 67 — LLM JUDGE EVALUATION (5 questions)")
    llm_scores = run_llm_judge_evaluation(
        golden_set,
        run_pipeline,
        sample_size=5
    )
# if you want to use ragas system this remove the comments 
#     print("\nTOPIC 69 — RAGAS EVALUATION")
#     ragas_results = run_full_evaluation(
#     pipeline_fn=run_pipeline,
#     use_ragas=True,
#     sample_size=5
#  )




    print(f"\nLLM Judge average scores:")
    for metric, score in llm_scores.items():
        if isinstance(score, float):
            print(f"  {metric}: {score:.3f}")

    # topic 70 — interpret scores
    print("\nTOPIC 70 — SCORE INTERPRETATION")
    interpretations = interpret_scores(llm_scores)

    # topic 71 — regression check
    print("\nTOPIC 71 — REGRESSION CHECK")
    current_results = {
        "timestamp": datetime.now().isoformat(),
        "method": "llm_judge",
        "sample_size": 5,
        "elapsed_seconds": 0,
        "scores": llm_scores
    }
    save_eval_results(current_results)
    check_regression(llm_scores)
    show_score_history()

    # topic 72 — A/B test example
    print("\nTOPIC 72 — A/B TEST EXAMPLE")

    def pipeline_with_reranking(question):
        context, sources, retrieved_docs = smart_retrieve(
            question=question,
            vectorstore=vectorstore,
            all_chunks=all_chunks,
            k=6,
            strategy="ensemble",
            apply_reorder=True,
            use_smart_filter=True,
            use_reranking=True
        )
        result = generate_rag_response(question=question, docs=retrieved_docs)
        result["context_used"] = context
        return result

    def pipeline_without_reranking(question):
        context, sources, retrieved_docs = smart_retrieve(
            question=question,
            vectorstore=vectorstore,
            all_chunks=all_chunks,
            k=6,
            strategy="ensemble",
            apply_reorder=True,
            use_smart_filter=True,
            use_reranking=False
        )
        result = generate_rag_response(question=question, docs=retrieved_docs)
        result["context_used"] = context
        return result

    ab_test(
        pipeline_a_fn=pipeline_with_reranking,
        pipeline_b_fn=pipeline_without_reranking,
        pipeline_a_name="With Reranking",
        pipeline_b_name="Without Reranking",
        sample_size=5
    )