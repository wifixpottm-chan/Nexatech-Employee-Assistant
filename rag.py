from functools import lru_cache
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


BASE_DIR = Path(__file__).resolve().parent
KB_DIR = BASE_DIR / "data" / "knowledge_base"
RELEVANCE_THRESHOLD = -0.1


def load_policy_documents() -> list[Document]:
    """Load the Markdown policy documents with source metadata."""
    return [
        Document(
            page_content=file.read_text(encoding="utf-8"),
            metadata={"source": file.name},
        )
        for file in sorted(KB_DIR.glob("*.md"))
    ]


def split_policy_documents(documents: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=150,
        separators=["\n\n", "\n", ".", " "],
    )
    return splitter.split_documents(documents)


@lru_cache(maxsize=1)
def get_retriever():
    """Build the vector index once per process, on first retrieval."""
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vector_store = Chroma.from_documents(
        documents=split_policy_documents(load_policy_documents()),
        embedding=embeddings,
        collection_name="nexatech_streamlit_policies",
    )
    return vector_store.as_retriever(search_kwargs={"k": 4})


def retrieve_policy_documents(question: str) -> list[Document]:
    vector_store = get_retriever().vectorstore
    scored_documents = vector_store.similarity_search_with_relevance_scores(
        question,
        k=4,
    )
    return [
        document
        for document, score in scored_documents
        if score >= RELEVANCE_THRESHOLD
    ]


def search_policy(question: str) -> str:
    retrieved_docs = retrieve_policy_documents(question)

    if not retrieved_docs:
        return "No relevant company policy was found."

    return "\n\n---\n\n".join(
        f"Source: {doc.metadata['source']}\n{doc.page_content}"
        for doc in retrieved_docs
    )