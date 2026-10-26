import chromadb
from sentence_transformers import SentenceTransformer

MODEL_NAME="sentence-transformers/all-MiniLM-L6-v2"

model=SentenceTransformer(MODEL_NAME)
client=chromadb.PersistentClient(path="vectordb/Hybrid_db")
collection=client.get_collection("document_chunks")

def semantic_search(query,top_k=5):
    query_embedding=model.encode(
        query,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    results=collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=min(top_k,collection.count()),
        include=["documents","metadatas","distances"]
    )

    retrieved=[]

    for text,metadata,distance in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    ):
        retrieved.append({
            "title":metadata["title"],
            "text":text,
            "source_url":metadata["source_url"],
            "document_id":metadata["document_id"],
            "distance":distance
        })

    return retrieved

if __name__=="__main__":
    query=input("Enter your search query: ")

    for rank,result in enumerate(semantic_search(query),start=1):
        print(f"\nRank {rank}: {result['title']}")
        print(f"Distance: {result['distance']:.4f}")
        print(f"Excerpt: {result['text'][:300]}")

        if result["source_url"]:
            print(f"Source: {result['source_url']}")