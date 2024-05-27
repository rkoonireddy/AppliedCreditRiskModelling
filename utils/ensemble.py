import os
from dotenv import load_dotenv
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_core.output_parsers import StrOutputParser
from langchain.docstore.document import Document

from utils.splitter import split_documents
from utils.vector_store import create_vector_db

def create_ensemble_retriever(docs, embeddings=None):
    """
    Create an ensemble retriever from a list of documents.

    Args:
        docs (list or Document): List of documents or single document.
        embeddings (optional): Embeddings for vector database (if applicable).

    Returns:
        EnsembleRetriever: Ensemble retriever instance.
    """
    # Convert single document to list if needed
    if isinstance(docs, Document):
        docs = [docs]
    
    # Split documents into text
    texts = split_documents(docs)
    
    # Create vector database retriever
    vector_db = create_vector_db(texts, embeddings)
    vector_db_retriever = vector_db.as_retriever()

    # Create BM25 retriever
    bm25_retriever = BM25Retriever.from_texts([doc.page_content for doc in docs])

    # Create ensemble retriever with equal weights
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_db_retriever],
        weights=[0.5, 0.5]
    )

    return ensemble_retriever
