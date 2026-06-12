from sentence_transformers import SentenceTransformer
import chromadb

def rag_context_collector(query: str) -> str:
    
    model = SentenceTransformer('all-MiniLM-L6-v2')

    client = chromadb.PersistentClient(path="./chroma_db")

    collection = client.get_or_create_collection(
    name="laws_collection",
    metadata={"hnsw:space": "cosine"})

    query_embedding = model.encode([query])
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=3,
        include=["metadatas", "documents"]
    )
    contexts = []
    for doc, meta in zip(results['documents'][0], results['metadatas'][0]):
        contexts.append(f"Document: {doc}\nMetadata: {meta}\n")

    return "\n".join(contexts)