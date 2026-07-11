from __future__ import annotations

import os
from pathlib import Path
from typing import Annotated

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import tool
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma


os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

base_dir = Path(__file__).resolve().parent.parent
load_dotenv(base_dir / ".env")


@tool
def retrieve_from_pdf(question: Annotated[str, "Question to answer from the loaded PDF documents"]) -> str:
    """Retrieve relevant passages from the indexed PDF documents."""
    pdf_path = base_dir / "1_Openai" / "Warren_Buffett.pdf"
    persist_dir = base_dir / "RAG" / "chroma_db"

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    if not persist_dir.exists():
        documents = PyPDFLoader(str(pdf_path)).load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(documents)

        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=str(persist_dir),
        )
    else:
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vector_store = Chroma(persist_directory=str(persist_dir), embedding_function=embeddings)

    docs = vector_store.similarity_search(question, k=4)
    return "\n\n".join(doc.page_content for doc in docs)


def build_agent():
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("Set OPENAI_API_KEY to run this agentic RAG example.")

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    system_prompt = """
    You are a helpful assistant for PDF-based question answering.
    Use the retrieve_from_pdf tool whenever the question requires information from the document.
    If the answer is already known from the retrieved context, respond clearly and concisely.
    """

    return create_agent(
        model=llm,
        tools=[retrieve_from_pdf],
        system_prompt=system_prompt,
        name="pdf_rag_agent",
    )


if __name__ == "__main__":
    agent = build_agent()
    print("PDF RAG Assistant is ready. Type 'exit' to quit.")

    while True:
        question = input("\nAsk a question about the PDF: ").strip()
        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        result = agent.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": question,
                    }
                ]
            }
        )
        print("\nAnswer:")
        print(result["messages"][-1].content)
