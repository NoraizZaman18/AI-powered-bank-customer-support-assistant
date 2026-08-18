from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv
import os

load_dotenv()

# step 1 — load all PDFs from documents folder
loader = PyPDFDirectoryLoader("data/documents/")
documents = loader.load()
print(f"Loaded {len(documents)} pages")

# step 2 — chunk documents
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
chunks = splitter.split_documents(documents)
print(f"Created {len(chunks)} chunks")

# step 3 — embed and store in ChromaDB
# using free local embeddings — no API cost
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"
)
print("Stored in ChromaDB")

# step 4 — create retriever
retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}
)

# step 5 — build RAG prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are ABC Bank customer support assistant.
     Answer the question using ONLY the context provided below.
     If the context does not contain the answer say:
     I don't have information about this in our documents.
     
     Context:
     {context}"""),
    ("human", "{question}")
])

# step 6 — setup LLM
llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY")
)

# step 7 — build chain
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)

# add this function to main.py
# it shows you exactly what happens at each step

def debug_rag_pipeline(question):
    print("\n" + "="*50)
    print(f"DEBUGGING: {question}")
    print("="*50)
    
    # STEP 1 — RETRIEVAL
    # what chunks did the system find?
    print("\n--- STEP 1: RETRIEVAL ---")
    retrieved_docs = retriever.invoke(question)
    print(f"Retrieved {len(retrieved_docs)} chunks")
    for i, doc in enumerate(retrieved_docs):
        print(f"\nChunk {i+1}:")
        print(f"  Source: {doc.metadata.get('source', 'unknown')}")
        print(f"  Page: {doc.metadata.get('page', 'unknown')}")
        print(f"  Content: {doc.page_content[:150]}...")
    
    # STEP 2 — AUGMENTATION
    # what does the full prompt look like after context is injected?
    print("\n--- STEP 2: AUGMENTATION ---")
    context = format_docs(retrieved_docs)
    print(f"Context length: {len(context)} characters")
    print(f"Context preview: {context[:300]}...")
    
    augmented_prompt = f"""
    Context: {context}
    
    Question: {question}
    """
    print(f"Full prompt length: {len(augmented_prompt)} characters")
    
    # STEP 3 — GENERATION
    # what answer does the LLM produce from this context?
    print("\n--- STEP 3: GENERATION ---")
    answer = rag_chain.invoke(question)
    print(f"Answer: {answer}")
    
    return {
        "question": question,
        "retrieved_chunks": len(retrieved_docs),
        "context_length": len(context),
        "answer": answer
    }

# test the debug function
result = debug_rag_pipeline(
    "What is the minimum balance for savings account?"
)
