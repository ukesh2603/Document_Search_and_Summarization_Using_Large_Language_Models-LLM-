import json
from pathlib import Path
import chromadb
from sentence_transformers import SentenceTransformer

CHUNKS_FILE=Path("processed_data/document_chunks.jsonl")
DB_DIR="vectordb/Hybrid_db"
COLLECTION_NAME="document_chunks"
MODEL_NAME="sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE=64

def load_chunks():
    if not CHUNKS_FILE.exists():
        raise FileNotFoundError("Chunk dataset not found. Run prepare_chunks.py first.")

    with CHUNKS_FILE.open("r",encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]

def build_index():
    records=load_chunks()

    if not records:
        raise ValueError("No chunks found.")

    seen_ids=set()

    for record in records:
        if record["chunk_id"] in seen_ids:
            raise ValueError(f"Duplicate chunk ID: {record['chunk_id']}")
        seen_ids.add(record["chunk_id"])

        if not record["text"].strip():
            raise ValueError(f"Empty chunk: {record['chunk_id']}")

    print(f"Loading embedding model: {MODEL_NAME}")
    model=SentenceTransformer(MODEL_NAME)

    client=chromadb.PersistentClient(path=DB_DIR)

    try:
        client.delete_collection(COLLECTION_NAME)
        print("Removed the previous collection.")
    except Exception:
        pass

    collection=client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space":"cosine"}
    )

    total=len(records)

    for start in range(0,total,BATCH_SIZE):
        batch=records[start:start+BATCH_SIZE]
        texts=[record["text"] for record in batch]

        embeddings=model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False
        )

        collection.add(
            ids=[record["chunk_id"] for record in batch],
            documents=texts,
            embeddings=embeddings.tolist(),
            metadatas=[
                {
                    "document_id":record["document_id"],
                    "title":record["title"],
                    "chunk_index":record["chunk_index"],
                    "source_url":record["source_url"],
                    "source":record["source"]
                }
                for record in batch
            ]
        )

        print(f"Indexed {min(start+len(batch),total)}/{total}")

    print("\nVector index built successfully.")
    print("Collection count:",collection.count())
    print("Database path:",DB_DIR)

if __name__=="__main__":
    build_index()