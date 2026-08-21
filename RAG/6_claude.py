"""Minimal RAG pipeline over a PDF: load -> split -> embed -> retrieve -> answer."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

base_dir = Path(__file__).resolve().parent.parent
load_dotenv(base_dir / ".env")

PDF_PATH = base_dir / "RAG" / "attention_is_all_you_need.pdf"
PERSIST_DIR = base_dir / "RAG" / "chroma_db_pdf_rag"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 4

PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You answer questions using only the context below. "
            "If the context does not contain the answer, say you don't know.\n\n"
            "Context:\n{context}",
        ),
        ("human", "{question}"),
    ]
)


def load_and_split(pdf_path: Path) -> list[Document]:
    """Read the PDF page by page and split it into overlapping chunks."""
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    documents = PyPDFLoader(file_path=pdf_path.as_posix(), mode="page").load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_documents(documents)


def build_vector_store() -> Chroma:
    """Return the persisted Chroma store, indexing the PDF on first run."""
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    if PERSIST_DIR.exists():
        return Chroma(persist_directory=str(PERSIST_DIR), embedding_function=embeddings)

    chunks = load_and_split(PDF_PATH)
    print(f"Indexing {len(chunks)} chunks from {PDF_PATH.name} ...")
    return Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(PERSIST_DIR),
    )


def format_docs(docs: list[Document]) -> str:
    """Flatten retrieved chunks into a single context block with page numbers."""
    return "\n\n".join(
        f"[page {doc.metadata.get('page', '?')}]\n{doc.page_content}" for doc in docs
    )


def build_chain():
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("Set OPENAI_API_KEY to run this RAG example.")

    retriever = build_vector_store().as_retriever(search_kwargs={"k": TOP_K})
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    return (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | PROMPT
        | llm
        | StrOutputParser()
    )


if __name__ == "__main__":
    chain = build_chain()
    print("PDF RAG pipeline is ready. Type 'exit' to quit.")

    while True:
        question = input("\nAsk a question about the PDF: ").strip()
        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        print("\nAnswer:")
        print(chain.invoke(question))
