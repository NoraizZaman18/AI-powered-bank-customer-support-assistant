# # add to generation/generator.py
# 1. Count approximate tokens
#         ↓
# 2. Calculate available context budget
#         ↓
# 3. Check retrieved chunks
#         ↓
# 4. If too large:
#       remove lower-priority chunks
#         ↓
# 5. Verify final context fits
#         ↓
# 6. Send to LLM


# generation/generator.py

import os
import re
from typing import List
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

MODEL_NAME = "openai/gpt-oss-20b"


def get_llm(temperature: float = 0):
    """Returns the LLM for answer generation."""
    from langchain_groq import ChatGroq
    return ChatGroq(
        model=MODEL_NAME,
        temperature=temperature,
        api_key=os.getenv("GROQ_API_KEY")
    )


# ── HELPERS — must be defined first ──────────────────────────────────────────

def format_context_basic(docs: List[Document]) -> str:
    """Basic context — joins all chunks into one block."""
    return "\n\n".join(doc.page_content for doc in docs)


def format_context_with_sources(docs: List[Document]) -> str:
    """
    Topic 49 — Context stuffing with source labels.
    Each chunk is labeled with document name and page number.
    """

    if not docs:
        return "No relevant information found in the documents."

    context_parts = []

    for i, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "unknown document")
        page = doc.metadata.get("page", "unknown page")
        section = doc.metadata.get("section", "")

        header = f"[Document {i}: {source} | Page {page}"
        if section and section not in ("general", ""):
            header += f" | Section: {section}"
        header += "]"

        context_parts.append(f"{header}\n{doc.page_content}")

    return "\n\n---\n\n".join(context_parts)


def get_unique_sources(docs: List[Document]) -> List[str]:
    """Get list of unique source documents from retrieved chunks."""
    sources = []
    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        if source not in sources:
            sources.append(source)
    return sources


# ── TOPIC 50 — CONTEXT WINDOW LIMITS ─────────────────────────────────────────

CONTEXT_LIMITS = {
    "openai/gpt-oss-20b": 8192,
    "llama-3.1-8b-instant": 8192,
    "mixtral-8x7b-32768": 32768,
    "gpt-4o": 128000,
    "claude-3-5-sonnet-20241022": 200000,
}

RESERVED_TOKENS = 1500


def count_tokens_approximate(text: str) -> int:
    """1 token ≈ 4 characters for English text."""
    return len(text) // 4


def get_context_limit(model_name: str = MODEL_NAME) -> int:
    """Get maximum tokens available for context."""
    total_limit = CONTEXT_LIMITS.get(model_name, 8192)
    return total_limit - RESERVED_TOKENS


def trim_context_to_limit(
    docs: List[Document],
    question: str,
    model_name: str = MODEL_NAME
) -> List[Document]:
    """
    Topic 50 — Trim retrieved chunks to fit within context window.
    Keeps chunks in order, drops from the end if over limit.
    """

    max_tokens = get_context_limit(model_name)
    question_tokens = count_tokens_approximate(question)
    available = max_tokens - question_tokens

    if not docs:
        return docs

    kept_docs = []
    used_tokens = 0

    for doc in docs:
        tokens = count_tokens_approximate(doc.page_content)
        if used_tokens + tokens <= available:
            kept_docs.append(doc)
            used_tokens += tokens

    if len(kept_docs) < len(docs):
        print(f"  Trimmed: kept {len(kept_docs)} of {len(docs)} chunks")

    return kept_docs


def check_context_health(
    docs: List[Document],
    question: str,
    model_name: str = MODEL_NAME
) -> dict:
    """Check if context fits within window before sending to LLM."""

    max_tokens = get_context_limit(model_name)
    context_text = format_context_with_sources(docs)
    context_tokens = count_tokens_approximate(context_text)
    question_tokens = count_tokens_approximate(question)
    total_tokens = context_tokens + question_tokens

    health = {
        "total_tokens": total_tokens,
        "context_tokens": context_tokens,
        "question_tokens": question_tokens,
        "limit": max_tokens,
        "within_limit": total_tokens <= max_tokens,
        "usage_percent": round(total_tokens / max_tokens * 100, 1),
        "num_chunks": len(docs),
        "unique_sources": len(get_unique_sources(docs))
    }

    return health


# ── TOPIC 51 — GROUNDING CHECK ────────────────────────────────────────────────

def is_context_sufficient(
    docs: List[Document],
    question: str,
    min_chars: int = 100
) -> bool:
    """Check if retrieved context is sufficient to answer."""

    if not docs:
        return False

    total_content = sum(len(doc.page_content) for doc in docs)
    return total_content >= min_chars


# ── TOPIC 52 — INSUFFICIENT CONTEXT HANDLING ─────────────────────────────────

INSUFFICIENT_CONTEXT_RESPONSE = (
    "I don't have specific information about this in our current policy documents.\n\n"
    "For accurate and up-to-date information, please:\n"
    "  • Call our helpline: 0800-ABCBANK (available 24/7)\n"
    "  • Email us: support@abcbank.com\n"
    "  • Visit any ABC Bank branch"
)


def detect_insufficient_context_in_answer(answer: str) -> bool:
    """Detect if the model indicated it could not answer."""

    phrases = [
        "i don't have information",
        "i don't have specific information",
        "not mentioned in",
        "not provided in the context",
        "cannot find",
        "no information about",
        "context does not contain",
        "not covered in",
    ]

    answer_lower = answer.lower()
    return any(phrase in answer_lower for phrase in phrases)


# ── TOPIC 54 — CITATION BUILDING ─────────────────────────────────────────────

def build_citation(doc: Document, citation_number: int) -> dict:
    """Build a single citation object from a document."""
    return {
        "number": citation_number,
        "source": doc.metadata.get("source", "unknown"),
        "page": doc.metadata.get("page", "unknown"),
        "section": doc.metadata.get("section", ""),
        "content_preview": doc.page_content[:150],
    }


def build_citations(docs: List[Document]) -> List[dict]:
    """Build citation list from all retrieved documents."""
    return [build_citation(doc, i) for i, doc in enumerate(docs, start=1)]


# ── TOPIC 55 — METADATA VALIDATION ───────────────────────────────────────────

def validate_citation_metadata(docs: List[Document]) -> dict:
    """Validate that chunks have complete citation metadata."""

    required = ["source", "page"]
    missing = {}

    for field in required:
        count = sum(1 for doc in docs if not doc.metadata.get(field))
        if count > 0:
            missing[field] = count

    return {
        "total_chunks": len(docs),
        "all_valid": len(missing) == 0,
        "missing_fields": missing
    }


# ── TOPIC 58 — DEDUPLICATION ─────────────────────────────────────────────────

def deduplicate_citations(citations: List[dict]) -> List[dict]:
    """Remove duplicate citations — same source and page."""

    seen = set()
    unique = []

    for citation in citations:
        key = f"{citation.get('source')}::{citation.get('page')}"
        if key not in seen:
            seen.add(key)
            unique.append(citation)

    for i, c in enumerate(unique, start=1):
        c["number"] = i

    return unique


def deduplicate_sources(sources: List[dict]) -> List[dict]:
    """Deduplicate sources list."""

    seen = set()
    unique = []

    for source in sources:
        key = f"{source.get('source')}::{source.get('page')}"
        if key not in seen:
            seen.add(key)
            unique.append(source)

    return unique


# ── TOPIC 59 — DISPLAY FORMATTING ────────────────────────────────────────────

def format_citations_text(citations: List[dict]) -> str:
    """Format citations as clean plain text."""

    if not citations:
        return ""

    lines = ["\n" + "─" * 40]
    lines.append("📄  Sources")
    lines.append("─" * 40)

    for c in citations:
        source = c.get("source", "unknown")
        page = c.get("page", "?")
        section = c.get("section", "")

        line = f"  [{c.get('number', '?')}] {source}  |  Page {page}"
        if section and section not in ("general", ""):
            line += f"  |  {section}"

        lines.append(line)

    return "\n".join(lines)


def format_full_response(
    answer: str,
    citations: List[dict],
    output_format: str = "text"
) -> str:
    """Combine answer and citations into final clean response."""

    citation_text = format_citations_text(citations)

    separator = "\n" + "═" * 50

    if citation_text:
        return f"{answer}\n{citation_text}"

    return answer


# ── PROMPT IMPORTS ────────────────────────────────────────────────────────────

from generation.prompts import (
    RAG_PROMPT,
    STRICT_GROUNDING_PROMPT,
    MULTI_DOC_SYNTHESIS_PROMPT,
    select_prompt,
)


# ── MAIN GENERATION FUNCTION — TOPICS 48 TO 59 ───────────────────────────────

def generate_rag_response(
    question: str,
    docs: List[Document],
    use_strict: bool = False,
    citation_style: str = "end",
    model_name: str = MODEL_NAME
) -> dict:
    """
    MAIN GENERATION FUNCTION — topics 48 to 59.

    citation_style: end / inline / none
    """

    # ── check sufficiency ─────────────────────────────────────────
    if not is_context_sufficient(docs, question):
        return {
            "answer": INSUFFICIENT_CONTEXT_RESPONSE,
            "formatted_response": INSUFFICIENT_CONTEXT_RESPONSE,
            "citations": [],
            "sources": [],
            "context_used": "",
            "sufficient_context": False,
            "fallback_used": True,
            "synthesis_used": False,
            "num_chunks": 0,
            "unique_sources": [],
            "health": {}
        }

    # ── check context health ──────────────────────────────────────
    health = check_context_health(docs, question, model_name)

    # ── trim if needed ────────────────────────────────────────────
    if not health["within_limit"]:
        docs = trim_context_to_limit(docs, question, model_name)

    # ── detect synthesis need ─────────────────────────────────────
    unique_sources = get_unique_sources(docs)
    synthesis_used = len(unique_sources) > 1

    # ── format context ────────────────────────────────────────────
    context = format_context_with_sources(docs)

    # ── select prompt ─────────────────────────────────────────────
    prompt = select_prompt(
        num_sources=len(unique_sources),
        use_strict=use_strict
    )

    # ── generate answer ───────────────────────────────────────────
    llm = get_llm()
    chain = prompt | llm | StrOutputParser()

    answer = chain.invoke({
        "context": context,
        "question": question
    })

    # ── fallback check ────────────────────────────────────────────
    fallback_used = detect_insufficient_context_in_answer(answer)
    if fallback_used:
        answer = INSUFFICIENT_CONTEXT_RESPONSE

    # ── build citations ───────────────────────────────────────────
    raw_citations = build_citations(docs)
    citations = deduplicate_citations(raw_citations)

    # ── format final response ─────────────────────────────────────
    formatted_response = format_full_response(answer, citations)

    # ── build sources ─────────────────────────────────────────────
    sources = deduplicate_sources([
        {
            "source": doc.metadata.get("source", "unknown"),
            "page": doc.metadata.get("page", "?"),
            "preview": doc.page_content[:100]
        }
        for doc in docs
    ])

    return {
        "answer": answer,
        "formatted_response": formatted_response,
        "citations": citations,
        "sources": sources,
        "context_used": context,
        "sufficient_context": True,
        "fallback_used": fallback_used,
        "synthesis_used": synthesis_used,
        "num_chunks": len(docs),
        "unique_sources": unique_sources,
        "health": health
    }


# ── TEST BLOCK ────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    from retrieval.store import get_vectorstore
    from retrieval.retriever import smart_retrieve
    from ingestion.loader import load_all_documents
    from ingestion.cleaner import clean_documents
    from ingestion.chunker import recursive_chunking

    vectorstore = get_vectorstore()
    docs = load_all_documents("data/documents/")
    docs = clean_documents(docs)
    all_chunks = recursive_chunking(docs)

    test_questions = [
        "What is the minimum balance for a savings account?",
        "What are all the costs associated with taking a personal loan?",
        "What is the current SBP policy rate?",
    ]

    for question in test_questions:
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

        print(f"\n{'═'*50}")
        print(f"Q: {question}")
        print(f"{'═'*50}")
        print(result["formatted_response"])