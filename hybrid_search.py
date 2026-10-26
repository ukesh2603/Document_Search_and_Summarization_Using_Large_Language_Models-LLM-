import json
import pickle
import re
from pathlib import Path
import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

CHUNKS_FILE=Path("processed_data/document_chunks.jsonl")
BM25_INDEX_FILE=Path("indexes/bm25_index.pkl")
CHROMA_PATH="vectordb/Hybrid_db"
COLLECTION_NAME="document_chunks"
EMBEDDING_MODEL="sentence-transformers/all-MiniLM-L6-v2"
CANDIDATE_K=10
FINAL_K=5
RRF_K=60

def tokenize(text):
    return re.findall(r"c\+\+|c#|[a-z0-9]+",text.lower())

def load_resources():
    if not BM25_INDEX_FILE.exists():
        raise FileNotFoundError(f"{BM25_INDEX_FILE} not found. Run: python bm25_search.py --build")

    if not CHUNKS_FILE.exists():
        raise FileNotFoundError(f"{CHUNKS_FILE} not found.")

    with BM25_INDEX_FILE.open("rb") as file:
        saved=pickle.load(file)

    bm25=saved["bm25"]
    records=saved["records"]
    client=chromadb.PersistentClient(path=CHROMA_PATH)
    collection=client.get_collection(name=COLLECTION_NAME)
    model=SentenceTransformer(EMBEDDING_MODEL)

    print(f"BM25 documents: {len(records)}")
    print(f"ChromaDB chunks: {collection.count()}")
    print("Hybrid search resources loaded.")

    return bm25,records,collection,model

def keyword_retrieve(query,bm25,records,top_k):
    query_tokens=tokenize(query)
    if not query_tokens:
        return []

    scores=bm25.get_scores(query_tokens)
    ranked_indices=sorted(range(len(scores)),key=lambda i:scores[i],reverse=True)
    results=[]

    for index in ranked_indices:
        score=float(scores[index])
        if score<=0:
            break

        record=records[index]
        results.append({
            "chunk_id":record["chunk_id"],
            "title":record.get("title",""),
            "text":record["text"],
            "source_url":record.get("source_url",""),
            "document_id":record.get("document_id",""),
            "bm25_score":score
        })

        if len(results)>=top_k:
            break

    return results

def semantic_retrieve(query,collection,model,top_k):
    query_embedding=model.encode([query],normalize_embeddings=True)[0].tolist()
    response=collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k,collection.count()),
        include=["documents","metadatas","distances"]
    )

    results=[]
    ids=response["ids"][0]
    documents=response["documents"][0]
    metadatas=response["metadatas"][0]
    distances=response["distances"][0]

    for chunk_id,text,metadata,distance in zip(ids,documents,metadatas,distances):
        metadata=metadata or {}
        results.append({
            "chunk_id":chunk_id,
            "title":metadata.get("title",""),
            "text":text or "",
            "source_url":metadata.get("source_url",""),
            "document_id":metadata.get("document_id",""),
            "semantic_distance":float(distance)
        })

    return results

def reciprocal_rank_fusion(keyword_results,semantic_results,rrf_k=60):
    fused={}

    for rank,result in enumerate(keyword_results,start=1):
        chunk_id=result["chunk_id"]
        fused[chunk_id]={
            **result,
            "rrf_score":1.0/(rrf_k+rank),
            "bm25_rank":rank,
            "semantic_rank":None
        }

    for rank,result in enumerate(semantic_results,start=1):
        chunk_id=result["chunk_id"]

        if chunk_id not in fused:
            fused[chunk_id]={
                **result,
                "rrf_score":0.0,
                "bm25_rank":None,
                "semantic_rank":rank
            }

        fused[chunk_id]["rrf_score"]+=1.0/(rrf_k+rank)
        fused[chunk_id]["semantic_rank"]=rank

        for field in ("title","text","source_url","document_id"):
            if not fused[chunk_id].get(field):
                fused[chunk_id][field]=result.get(field,"")

        fused[chunk_id]["semantic_distance"]=result.get("semantic_distance")

    return sorted(fused.values(),key=lambda item:item["rrf_score"],reverse=True)

def hybrid_search(query,bm25,records,collection,model,top_k=FINAL_K):
    if not query.strip():
        return []

    keyword_results=keyword_retrieve(query,bm25,records,CANDIDATE_K)
    semantic_results=semantic_retrieve(query,collection,model,CANDIDATE_K)
    fused_results=reciprocal_rank_fusion(keyword_results,semantic_results,RRF_K)

    return fused_results[:top_k]

def main():
    bm25,records,collection,model=load_resources()

    print("\nHybrid Search ready.")
    print("Enter a question, or type 'exit' to quit.")

    while True:
        query=input("\nYour query: ").strip()
        if query.lower() in {"exit","quit"}:
            break

        results=hybrid_search(query,bm25,records,collection,model)
        if not results:
            print("No matching passages found.")
            continue

        for rank,result in enumerate(results,start=1):
            print(f"\n{'='*65}")
            print(f"Rank: {rank}")
            print(f"Title: {result.get('title','Unknown')}")
            print(f"Chunk ID: {result['chunk_id']}")
            print(f"RRF score: {result['rrf_score']:.5f}")
            print(f"BM25 rank: {result.get('bm25_rank')}")
            print(f"Semantic rank: {result.get('semantic_rank')}")
            print(f"Excerpt: {result.get('text','')[:700]}")

            if result.get("source_url"):
                print(f"Source: {result['source_url']}")

if __name__=="__main__":
    main()