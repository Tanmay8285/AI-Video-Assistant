import os
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

CHROMA_DIR="vector_db"
COLLLECTION_NAME="meeting_transcript"
EMBEDDING_MODEL="all-MiniLM-L6-v2"

def get_embedding_model():
    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        device = "cpu"

    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL, model_kwargs={"device": device})

def build_vector_store(transcript:str)-> Chroma:
    print("Building vector store...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_text(transcript)
    docs = [
        Document(page_content=chunk, metadata = {'chunk_index' : i})
        for i,chunk in enumerate(chunks)
    ]

    embedding_model = get_embedding_model()
    vector_store = Chroma.from_documents(docs, embedding_model, collection_name=COLLLECTION_NAME, persist_directory=CHROMA_DIR)
    return vector_store

def load_vector_store() -> Chroma:
    embedding_model = get_embedding_model()
    return Chroma(collection_name=COLLLECTION_NAME, embedding_function=embedding_model, persist_directory=CHROMA_DIR)

def get_retriever(vector_store: Chroma, k: int = 4):
    return vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k})