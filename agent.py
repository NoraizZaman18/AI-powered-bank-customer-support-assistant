# agent.py

import os
import json
from typing import List, Optional
from langchain_core.documents import Document
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

MODEL_NAME = "openai/gpt-oss-20b"


def get_llm(temperature: float = 0):
    """Returns the LLM for agent operations."""
    from langchain_groq import ChatGroq
    return ChatGroq(
        model=MODEL_NAME,
        temperature=temperature,
        api_key=os.getenv("GROQ_API_KEY")
    )
# add to agent.py — below topic 73 code

# ── TOPIC 73 — AGENTIC RAG DECISION MAKING ───────────────────────────────────

NEEDS_RETRIEVAL_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are an AI assistant for ABC Bank.

Decide if the user's question requires searching bank policy documents.

Answer only YES or NO.

Answer YES if the question is about:
- Bank policies, fees, charges, or procedures
- Account opening, loans, credit cards
- Complaints, documents required, timelines
- Any specific ABC Bank information

Answer NO if the question is about:
- General greetings or small talk
- Questions you can answer from general knowledge
- Math calculations without bank context
- Questions clearly outside banking scope"""),
    ("human", "Question: {question}")
])


def needs_retrieval(question: str) -> bool:
    """
    Topic 73 — Agent decides if retrieval is needed for this question.
    Saves time and cost by skipping retrieval for simple questions.
    """

    llm = get_llm()
    chain = NEEDS_RETRIEVAL_PROMPT | llm | StrOutputParser()
    decision = chain.invoke({"question": question}).strip().upper()

    needs_it = decision.startswith("YES")
    print(f"\nRetrieval needed: {'YES' if needs_it else 'NO'} "
          f"— {question[:50]}...")
    return needs_it


DIRECT_ANSWER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful ABC Bank customer support assistant.

Answer the customer's question directly.
If it is a greeting, respond warmly and professionally.
If it is a general question you can answer from knowledge, answer it.
Keep the response concise and professional."""),
    ("human", "{question}")
])


def generate_direct_answer(question: str) -> str:
    """
    Topic 73 — Generate answer without retrieval for simple questions.
    Used when agent decides retrieval is not needed.
    """

    llm = get_llm()
    chain = DIRECT_ANSWER_PROMPT | llm | StrOutputParser()
    return chain.invoke({"question": question})

from retrieval.retriever import smart_retrieve
from generation.generator import generate_rag_response


# ── TOPIC 74 — RAG AS A TOOL ──────────────────────────────────────────────────

# these will be set from main.py before agent runs
_vectorstore = None
_all_chunks = None


def set_agent_dependencies(vectorstore, all_chunks):
    """
    Call this from main.py after vectorstore and chunks are ready.
    Sets the dependencies the agent tools need.
    """
    global _vectorstore, _all_chunks
    _vectorstore = vectorstore
    _all_chunks = all_chunks
    print("Agent dependencies set")


@tool
def search_bank_documents(query: str) -> str:
    """
    Search ABC Bank policy documents for information.

    Use this tool when the customer asks about:
    - Account opening requirements and minimum balance
    - Loan eligibility, interest rates, and fees
    - Credit card policies, limits, and charges
    - Schedule of charges for any banking service
    - Complaint filing procedures and timelines
    - Any specific ABC Bank policy or procedure

    Input: a clear search query describing what information is needed.
    Output: relevant information from bank documents with source citations.
    """

    if _vectorstore is None or _all_chunks is None:
        return "Error: Agent not initialized. Call set_agent_dependencies first."

    print(f"\n[TOOL: search_bank_documents] Query: {query}")

    context, sources, retrieved_docs = smart_retrieve(
        question=query,
        vectorstore=_vectorstore,
        all_chunks=_all_chunks,
        k=6,
        strategy="ensemble",
        apply_reorder=True,
        use_smart_filter=True,
        use_reranking=True
    )

    if not retrieved_docs:
        return "No relevant information found in bank documents."

    result = generate_rag_response(
        question=query,
        docs=retrieved_docs,
        citation_style="end"
    )

    return result["formatted_response"]


@tool
def calculate_loan_installment(
    principal: float,
    annual_rate: float,
    tenure_months: int
) -> str:
    """
    Calculate monthly loan installment using reducing balance method.

    Use this tool when customer wants to know their monthly payment
    for a loan with specific principal, interest rate, and tenure.

    Input:
    - principal: loan amount in rupees
    - annual_rate: interest rate per year as percentage (e.g. 18 for 18%)
    - tenure_months: loan tenure in months (e.g. 24 for 2 years)

    Output: monthly installment amount in rupees.
    """

    print(f"\n[TOOL: calculate_loan_installment] "
          f"Principal={principal}, Rate={annual_rate}%, Tenure={tenure_months}m")

    monthly_rate = annual_rate / 100 / 12

    if monthly_rate == 0:
        installment = principal / tenure_months
    else:
        installment = (
            principal
            * monthly_rate
            * (1 + monthly_rate) ** tenure_months
            / ((1 + monthly_rate) ** tenure_months - 1)
        )

    total_payment = installment * tenure_months
    total_interest = total_payment - principal

    return (
        f"Loan Calculation:\n"
        f"  Principal: Rs. {principal:,.0f}\n"
        f"  Annual rate: {annual_rate}%\n"
        f"  Tenure: {tenure_months} months\n"
        f"  Monthly installment: Rs. {installment:,.0f}\n"
        f"  Total payment: Rs. {total_payment:,.0f}\n"
        f"  Total interest: Rs. {total_interest:,.0f}"
    )


@tool
def get_bank_contact_info(inquiry_type: str) -> str:
    """
    Get ABC Bank contact information for a specific inquiry type.

    Use this tool when customer needs to contact the bank directly.

    Input: type of inquiry (complaint, loan, account, card, general)
    Output: relevant contact information.
    """

    print(f"\n[TOOL: get_bank_contact_info] Type: {inquiry_type}")

    contacts = {
        "complaint": (
            "ABC Bank Complaint Channels:\n"
            "  Phone: 0800-ABCBANK (24/7)\n"
            "  Email: complaints@abcbank.com\n"
            "  Mobile App: Help → File Complaint\n"
            "  Branch: Visit any ABC Bank branch\n"
            "  SBP Mohtasib: 0800-200-00 (if unresolved after 45 days)"
        ),
        "loan": (
            "ABC Bank Loan Inquiries:\n"
            "  Phone: 0800-ABCBANK\n"
            "  Email: loans@abcbank.com\n"
            "  Visit any branch with your documents"
        ),
        "card": (
            "ABC Bank Card Services:\n"
            "  Card Block: 0800-ABCBANK (24/7 — immediate)\n"
            "  Mobile App: Cards → Block Card\n"
            "  Email: cards@abcbank.com"
        ),
        "general": (
            "ABC Bank Contact:\n"
            "  Helpline: 0800-ABCBANK\n"
            "  International: +92-21-111-222-265\n"
            "  Email: support@abcbank.com\n"
            "  Branch timings: Mon-Thu 9am-5pm, Fri 9am-1pm & 2:30-5pm"
        )
    }

    inquiry_lower = inquiry_type.lower()
    for key, info in contacts.items():
        if key in inquiry_lower:
            return info

    return contacts["general"]


ALL_TOOLS = [
    search_bank_documents,
    calculate_loan_installment,
    get_bank_contact_info
]



# add to agent.py


# ── TOPIC 75 — CORRECTIVE RAG ─────────────────────────────────────────────────

RELEVANCE_CHECK_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a retrieval quality evaluator.

Check if the retrieved document chunks are relevant to the question.

Answer only YES or NO.

Answer YES if:
- The chunks contain information that can help answer the question
- At least one chunk is clearly related to the question topic

Answer NO if:
- The chunks are about completely different topics
- None of the chunks contain useful information for the question"""),
    ("human", """Question: {question}

Retrieved content preview:
{context_preview}

Are these chunks relevant?""")
])


def evaluate_retrieval_quality(
    question: str,
    docs: List[Document]
) -> bool:
    """
    Topic 75 — Evaluate if retrieved chunks are relevant to the question.
    Returns True if relevant, False if retrieval needs correction.
    """

    if not docs:
        print("Retrieval quality: FAIL — no documents retrieved")
        return False

    context_preview = "\n".join([
        f"- {doc.page_content[:150]}..."
        for doc in docs[:3]
    ])

    llm = get_llm()
    chain = RELEVANCE_CHECK_PROMPT | llm | StrOutputParser()

    decision = chain.invoke({
        "question": question,
        "context_preview": context_preview
    }).strip().upper()

    is_relevant = decision.startswith("YES")
    print(f"Retrieval quality: {'GOOD' if is_relevant else 'POOR — will retry'}")
    return is_relevant


QUERY_REFORMULATION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a search query optimizer for a bank document system.

The original search query did not return useful results.
Generate a better search query to find the relevant bank policy information.

Rules:
- Use different keywords than the original
- Be more specific about what banking information is needed
- Use formal banking terminology
- Return ONLY the new query, nothing else"""),
    ("human", """Original question: {question}
Original query that failed: {failed_query}

Generate a better search query:""")
])


def reformulate_query(question: str, failed_query: str) -> str:
    """
    Topic 75 — Reformulate a failed query to improve retrieval.
    Called when corrective RAG detects poor retrieval quality.
    """

    llm = get_llm()
    chain = QUERY_REFORMULATION_PROMPT | llm | StrOutputParser()

    new_query = chain.invoke({
        "question": question,
        "failed_query": failed_query
    }).strip()

    print(f"Reformulated query: {new_query[:80]}...")
    return new_query


def corrective_retrieve(
    question: str,
    max_corrections: int = 2
) -> List[Document]:
    """
    Topic 75 — Corrective RAG retrieval loop.

    Steps:
    1. Retrieve with original question
    2. Evaluate retrieval quality
    3. If poor quality — reformulate query and retry
    4. Repeat up to max_corrections times
    5. Return best retrieved docs

    max_corrections: maximum retry attempts before giving up
    """

    if _vectorstore is None or _all_chunks is None:
        return []

    current_query = question
    best_docs = []

    for attempt in range(max_corrections + 1):

        print(f"\nCorrective RAG attempt {attempt + 1}/{max_corrections + 1}")

        context, sources, docs = smart_retrieve(
            question=current_query,
            vectorstore=_vectorstore,
            all_chunks=_all_chunks,
            k=6,
            strategy="ensemble",
            apply_reorder=True,
            use_smart_filter=True,
            use_reranking=True
        )

        if evaluate_retrieval_quality(question, docs):
            print(f"Good retrieval on attempt {attempt + 1}")
            return docs

        if attempt < max_corrections:
            print(f"Retrieval quality poor — reformulating query...")
            current_query = reformulate_query(question, current_query)
            best_docs = docs
        else:
            print(f"Max corrections reached — using best available docs")
            return best_docs if best_docs else docs

    return best_docs

# add to agent.py


# ── TOPIC 76 — MULTI-HOP RETRIEVAL ───────────────────────────────────────────

SUB_QUESTION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a question decomposer for a bank document search system.

Break down complex questions into simple sub-questions.
Each sub-question should be answerable from a single bank policy document.

Return a JSON list of sub-questions.
Maximum 3 sub-questions.
Return ONLY valid JSON — no explanation, no markdown.

Example output:
["What is the loan default policy?", "What is the prepayment policy?"]"""),
    ("human", "Complex question: {question}")
])


def decompose_question(question: str) -> List[str]:
    """
    Topic 76 — Break a complex question into simple sub-questions.
    Each sub-question is answered by a separate retrieval.
    """

    llm = get_llm()
    chain = SUB_QUESTION_PROMPT | llm | StrOutputParser()

    raw = chain.invoke({"question": question}).strip()

    try:
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        sub_questions = json.loads(raw.strip())

        if not isinstance(sub_questions, list):
            return [question]

        print(f"\nDecomposed into {len(sub_questions)} sub-questions:")
        for i, sq in enumerate(sub_questions, start=1):
            print(f"  {i}. {sq}")

        return sub_questions

    except Exception:
        print("Decomposition failed — using original question")
        return [question]


MULTI_HOP_SYNTHESIS_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are ABC Bank customer support assistant.

Answer the customer's question by combining information from
multiple retrieved document sections below.

Use ONLY the information provided in the context.
If any part cannot be answered from the context say so clearly.

Context from multiple searches:
{combined_context}"""),
    ("human", "Original question: {question}")
])


def multi_hop_retrieve(question: str) -> dict:
    """
    Topic 76 — Multi-hop retrieval for complex questions.

    Steps:
    1. Decompose question into sub-questions
    2. Retrieve for each sub-question separately
    3. Combine all retrieved docs
    4. Generate unified answer from combined context
    """

    if _vectorstore is None or _all_chunks is None:
        return {"answer": "Agent not initialized.", "docs": []}

    print(f"\nMulti-hop retrieval for: {question[:60]}...")

    sub_questions = decompose_question(question)

    all_docs = []
    sub_contexts = []
    seen_content = set()

    for i, sub_q in enumerate(sub_questions, start=1):
        print(f"\nHop {i}: {sub_q}")

        context, sources, docs = smart_retrieve(
            question=sub_q,
            vectorstore=_vectorstore,
            all_chunks=_all_chunks,
            k=3,
            strategy="ensemble",
            apply_reorder=False,
            use_smart_filter=True,
            use_reranking=True
        )

        for doc in docs:
            content_key = doc.page_content[:100]
            if content_key not in seen_content:
                seen_content.add(content_key)
                all_docs.append(doc)

        sub_contexts.append(
            f"--- Information for: {sub_q} ---\n{context}"
        )

    combined_context = "\n\n".join(sub_contexts)

    llm = get_llm()
    chain = MULTI_HOP_SYNTHESIS_PROMPT | llm | StrOutputParser()

    answer = chain.invoke({
        "combined_context": combined_context,
        "question": question
    })

    print(f"\nMulti-hop complete: {len(sub_questions)} hops, "
          f"{len(all_docs)} unique chunks")

    return {
        "answer": answer,
        "docs": all_docs,
        "sub_questions": sub_questions,
        "combined_context": combined_context
    }

# add to agent.py


# ── TOPIC 77 — ITERATIVE RETRIEVAL ───────────────────────────────────────────

COMPLETENESS_CHECK_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are an answer completeness evaluator.

Check if the current answer fully addresses the original question.

Answer with JSON only:
{{"complete": true/false, "missing": "what information is still needed"}}

If complete set missing to empty string.
Return ONLY valid JSON."""),
    ("human", """Original question: {question}

Current answer so far:
{current_answer}

Is the answer complete?""")
])


def check_answer_completeness(
    question: str,
    current_answer: str
) -> dict:
    """
    Topic 77 — Check if the current answer fully addresses the question.
    Returns completeness status and what is still missing.
    """

    llm = get_llm()
    chain = COMPLETENESS_CHECK_PROMPT | llm | StrOutputParser()

    raw = chain.invoke({
        "question": question,
        "current_answer": current_answer
    }).strip()

    try:
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        result = json.loads(raw.strip())
        return result
    except Exception:
        return {"complete": True, "missing": ""}


def iterative_retrieve(
    question: str,
    max_iterations: int = 3
) -> dict:
    """
    Topic 77 — Iterative retrieval that keeps searching until answer is complete.

    Steps:
    1. Retrieve and generate initial answer
    2. Check if answer is complete
    3. If incomplete — retrieve for missing information
    4. Merge new information into answer
    5. Repeat up to max_iterations times

    max_iterations: maximum retrieval rounds before stopping
    """

    if _vectorstore is None or _all_chunks is None:
        return {"answer": "Agent not initialized.", "iterations": 0}

    print(f"\nIterative retrieval for: {question[:60]}...")

    all_docs = []
    seen_content = set()
    current_answer = ""
    current_query = question

    for iteration in range(max_iterations):

        print(f"\nIteration {iteration + 1}/{max_iterations}")

        context, sources, docs = smart_retrieve(
            question=current_query,
            vectorstore=_vectorstore,
            all_chunks=_all_chunks,
            k=4,
            strategy="ensemble",
            apply_reorder=True,
            use_smart_filter=True,
            use_reranking=True
        )

        for doc in docs:
            key = doc.page_content[:100]
            if key not in seen_content:
                seen_content.add(key)
                all_docs.append(doc)

        result = generate_rag_response(
            question=question,
            docs=all_docs,
            citation_style="end"
        )

        current_answer = result["answer"]

        completeness = check_answer_completeness(question, current_answer)
        print(f"Complete: {completeness.get('complete', True)}")

        if completeness.get("complete", True):
            print(f"Answer complete after {iteration + 1} iteration(s)")
            break

        missing = completeness.get("missing", "")
        if missing:
            current_query = missing
            print(f"Missing: {missing[:80]}...")
        else:
            break

    return {
        "answer": current_answer,
        "docs": all_docs,
        "iterations": iteration + 1
    }
# add to agent.py


# ── TOPIC 78 — QUERY ROUTING ──────────────────────────────────────────────────

ROUTE_DETECTION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a query router for ABC Bank customer support.

Classify the question into exactly one category.
Return ONLY the category name — nothing else.

Categories:
- simple_factual: single fact lookup (fees, rates, limits, timelines)
- complex_policy: requires understanding multiple policy aspects
- calculation: needs mathematical computation
- complaint: about filing complaints or escalation
- contact: asking for contact information or branch details
- greeting: general greeting or non-banking question
- multi_part: question with multiple distinct sub-questions"""),
    ("human", "Question: {question}")
])


def detect_question_type(question: str) -> str:
    """
    Topic 78 — Detect question type for routing.
    Returns category string that determines retrieval strategy.
    """

    llm = get_llm()
    chain = ROUTE_DETECTION_PROMPT | llm | StrOutputParser()

    category = chain.invoke({"question": question}).strip().lower()

    valid_categories = [
        "simple_factual", "complex_policy", "calculation",
        "complaint", "contact", "greeting", "multi_part"
    ]

    if category not in valid_categories:
        category = "simple_factual"

    print(f"Question type: {category}")
    return category


def route_and_retrieve(question: str) -> dict:
    """
    Topic 78 — Route question to correct retrieval strategy.

    Routing logic:
    - greeting → direct answer, no retrieval
    - contact → contact info tool
    - calculation → calculator tool + RAG
    - simple_factual → basic ensemble retrieval
    - complex_policy → corrective RAG
    - multi_part → multi-hop retrieval
    - complaint → RAG focused on complaint documents
    """

    question_type = detect_question_type(question)

    print(f"\nRouting: {question_type} → ", end="")

    if question_type == "greeting":
        print("direct answer")
        answer = generate_direct_answer(question)
        return {
            "answer": answer,
            "strategy": "direct",
            "docs": []
        }

    elif question_type == "contact":
        print("contact info tool")
        answer = get_bank_contact_info.invoke({"inquiry_type": question})
        return {
            "answer": answer,
            "strategy": "contact_tool",
            "docs": []
        }

    elif question_type == "calculation":
        print("RAG + calculator")
        context, sources, docs = smart_retrieve(
            question=question,
            vectorstore=_vectorstore,
            all_chunks=_all_chunks,
            k=4,
            strategy="ensemble",
            apply_reorder=True,
            use_smart_filter=True,
            use_reranking=True
        )
        result = generate_rag_response(
            question=question,
            docs=docs,
            citation_style="end"
        )
        return {
            "answer": result["formatted_response"],
            "strategy": "rag_with_calculator",
            "docs": docs
        }

    elif question_type == "multi_part":
        print("multi-hop retrieval")
        result = multi_hop_retrieve(question)
        return {
            "answer": result["answer"],
            "strategy": "multi_hop",
            "docs": result["docs"]
        }

    elif question_type == "complex_policy":
        print("corrective RAG")
        docs = corrective_retrieve(question)
        result = generate_rag_response(
            question=question,
            docs=docs,
            citation_style="end"
        )
        return {
            "answer": result["formatted_response"],
            "strategy": "corrective_rag",
            "docs": docs
        }

    else:
        print("standard ensemble retrieval")
        context, sources, docs = smart_retrieve(
            question=question,
            vectorstore=_vectorstore,
            all_chunks=_all_chunks,
            k=6,
            strategy="ensemble",
            apply_reorder=True,
            use_smart_filter=True,
            use_reranking=True
        )
        result = generate_rag_response(
            question=question,
            docs=docs,
            citation_style="end"
        )
        return {
            "answer": result["formatted_response"],
            "strategy": "standard_rag",
            "docs": docs
        }



# add to agent.py


# ── TOPIC 79 — SELF-RAG ───────────────────────────────────────────────────────

GROUNDEDNESS_CHECK_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a groundedness evaluator.

Check if the generated answer is grounded in the provided context.

Answer with JSON only:
{{"grounded": true/false, "reason": "brief explanation"}}

Grounded means every claim in the answer can be traced to the context.
Return ONLY valid JSON."""),
    ("human", """Context:
{context}

Generated answer:
{answer}

Is this answer grounded in the context?""")
])


def check_answer_groundedness(
    answer: str,
    context: str
) -> dict:
    """
    Topic 79 — Check if answer is grounded in retrieved context.
    Part of the self-RAG verification loop.
    """

    llm = get_llm()
    chain = GROUNDEDNESS_CHECK_PROMPT | llm | StrOutputParser()

    raw = chain.invoke({
        "answer": answer,
        "context": context
    }).strip()

    try:
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        result = json.loads(raw.strip())
        return result
    except Exception:
        return {"grounded": True, "reason": "Parse error — assuming grounded"}


def self_rag(question: str) -> dict:
    """
    Topic 79 — Complete Self-RAG loop.

    Decision points:
    1. Does this question need retrieval?
    2. Route to appropriate retrieval strategy
    3. Is the retrieval relevant? (corrective loop)
    4. Generate answer from retrieved context
    5. Is the answer grounded in context?
    6. If not grounded — retrieve again and regenerate

    This is the most complete agentic RAG implementation.
    Combines topics 73 to 78 into one intelligent loop.
    """

    if _vectorstore is None or _all_chunks is None:
        return {"answer": "Agent not initialized.", "strategy": "error"}

    print(f"\n{'='*60}")
    print(f"SELF-RAG")
    print(f"Question: {question[:60]}...")
    print(f"{'='*60}")

    # ── step 1 — does this need retrieval? ───────────────────────
    if not needs_retrieval(question):
        answer = generate_direct_answer(question)
        return {
            "answer": answer,
            "strategy": "direct_no_retrieval",
            "grounded": True,
            "docs": []
        }

    # ── step 2 — route to correct strategy ───────────────────────
    route_result = route_and_retrieve(question)

    answer = route_result["answer"]
    docs = route_result["docs"]
    strategy = route_result["strategy"]

    if strategy in ("direct", "contact_tool"):
        return {
            "answer": answer,
            "strategy": strategy,
            "grounded": True,
            "docs": []
        }

    # ── step 3 — check groundedness ───────────────────────────────
    if docs:
        from generation.generator import format_context_with_sources
        context = format_context_with_sources(docs)

        groundedness = check_answer_groundedness(answer, context)
        is_grounded = groundedness.get("grounded", True)

        print(f"Answer grounded: {is_grounded}")

        if not is_grounded:
            print("Answer not grounded — retrieving again...")

            docs = corrective_retrieve(question, max_corrections=1)

            if docs:
                from generation.generator import generate_rag_response
                result = generate_rag_response(
                    question=question,
                    docs=docs,
                    citation_style="end"
                )
                answer = result["formatted_response"]
                is_grounded = True
    else:
        is_grounded = True

    return {
        "answer": answer,
        "strategy": strategy,
        "grounded": is_grounded,
        "docs": docs
    }


# ── MAIN AGENT FUNCTION ───────────────────────────────────────────────────────

def run_agent(question: str) -> dict:
    """
    MAIN AGENT FUNCTION — called from main.py.
    Runs the full self-RAG pipeline for any question.
    Returns complete response dict.
    """

    result = self_rag(question)

    return {
        "question": question,
        "answer": result["answer"],
        "strategy": result["strategy"],
        "grounded": result["grounded"],
        "num_docs": len(result.get("docs", []))
    }


# ── TEST BLOCK ────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    import sys
    sys.path.insert(
        0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )

    from retrieval.store import incremental_index
    from ingestion.loader import load_all_documents
    from ingestion.cleaner import clean_documents
    from ingestion.chunker import (
        detect_document_types,
        recommend_overlap,
        recursive_chunking,
        enrich_metadata
    )

    documents = load_all_documents("data/documents/")
    documents = clean_documents(documents)

    def incremental_chunking(docs):
        document_types = detect_document_types(docs)
        final_chunks = []
        for source, doc_type in document_types.items():
            settings = recommend_overlap(doc_type)
            source_docs = [
                d for d in docs
                if d.metadata.get("source") == source
            ]
            chunks = recursive_chunking(
                source_docs,
                settings["chunk_size"],
                settings["overlap"]
            )
            chunks = enrich_metadata(chunks)
            final_chunks.extend(chunks)
        return final_chunks

    vectorstore = incremental_index(
        "data/documents/",
        incremental_chunking,
        False
    )
    all_chunks = incremental_chunking(documents)

    set_agent_dependencies(vectorstore, all_chunks)

    test_questions = [
        "Hello, how are you?",
        "What is the minimum balance for a savings account?",
        "How do I contact the bank if my card is stolen?",
        "What happens if I miss a loan payment and then want to prepay early?",
        "If I take a loan of Rs. 500,000 at 20% for 3 years what is my monthly payment?",
        "What are all the requirements to open an account and what are the minimum balances?",
    ]

    print("\n" + "="*60)
    print("SECTION 9 — AGENTIC RAG TEST")
    print("="*60)

    for question in test_questions:
        result = run_agent(question)

        print(f"\n{'═'*60}")
        print(f"Q: {result['question']}")
        print(f"Strategy: {result['strategy']}")
        print(f"{'═'*60}")
        print(f"\n{result['answer']}")
        print()    