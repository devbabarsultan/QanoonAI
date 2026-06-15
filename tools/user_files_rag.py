from sentence_transformers import SentenceTransformer
import chromadb
from typing import List
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader
from pathlib import Path

# Global setup (reuse the same model and client as your legal RAG)
model = SentenceTransformer('all-MiniLM-L6-v2',local_files_only=True)
client = chromadb.PersistentClient(path="./chroma_db")

# Collection for all user files
USER_COLLECTION_NAME = "user_files"

def get_or_create_user_collection():
    """Get or create the user files collection with cosine similarity."""
    return client.get_or_create_collection(
        name=USER_COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )

def add_user_files_to_chroma(user_id: str, file_paths: List[str]):
    """
    Load, chunk, embed, and store user files in Chroma.
    Each chunk gets metadata: {"user_id": user_id, "source": file_path}
    """
    collection = get_or_create_user_collection()
    
    # Text splitter (same as before)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", " ", ""]
    )
    
    all_chunks = []
    all_metadatas = []
    all_ids = []
    
    for file_path in file_paths:
        # Load based on extension
        ext = Path(file_path).suffix.lower()
        if ext == ".pdf":
            loader = PyPDFLoader(file_path)
            docs = loader.load()
            raw_texts = [doc.page_content for doc in docs]
        elif ext == ".docx":
            loader = Docx2txtLoader(file_path)
            docs = loader.load()
            raw_texts = [doc.page_content for doc in docs]
        elif ext == ".txt":
            loader = TextLoader(file_path, encoding="utf-8")
            docs = loader.load()
            raw_texts = [doc.page_content for doc in docs]
        else:
            continue  # skip unsupported
        
        # Chunk each document
        for raw_text in raw_texts:
            chunks = splitter.split_text(raw_text)
            for i, chunk in enumerate(chunks):
                chunk_id = f"{user_id}_{Path(file_path).name}_{i}"
                all_ids.append(chunk_id)
                all_chunks.append(chunk)
                all_metadatas.append({
                    "user_id": user_id,
                    "source": file_path,
                    "chunk_index": i
                })
    
    if not all_chunks:
        return
    
    # Generate embeddings
    embeddings = model.encode(all_chunks).tolist()
    
    # Add to Chroma (existing chunks with same ID will be overwritten – adjust as needed)
    collection.upsert(
        embeddings=embeddings,
        documents=all_chunks,
        metadatas=all_metadatas,
        ids=all_ids
    )

def get_user_files_context(query: str, user_id: str, n_results: int = 3) -> str:
    """
    Retrieve relevant chunks from this user's files only.
    """
    collection = get_or_create_user_collection()
    
    # Embed the query
    query_embedding = model.encode([query]).tolist()
    
    # Query with metadata filter to restrict to this user_id
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=n_results,
        where={"user_id": user_id},   # only this user's documents
        include=["metadatas", "documents"]
    )
    
    if not results['documents'][0]:
        return "No relevant content found in your uploaded files."
    
    contexts = []
    for doc, meta in zip(results['documents'][0], results['metadatas'][0]):
        contexts.append(f"From file {meta.get('source', 'unknown')}:\n{doc}\n")
    
    return "\n".join(contexts)