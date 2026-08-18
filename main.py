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

# step 8 — test it
print("\n" + "="*50)
print("Testing naive RAG")
print("="*50)

test_questions = [
    "What documents do I need to open a savings account?",
    "What is the interest rate for personal loan?",
    "How do I file a complaint?"
]

for question in test_questions:
    print(f"\nQ: {question}")
    answer = rag_chain.invoke(question)
    print(f"A: {answer}")