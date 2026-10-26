import json
import pickle
import re
import argparse
from pathlib import Path
from rank_bm25 import BM25Okapi

CHUNKS_FILE=Path("processed_data/document_chunks.jsonl")
INDEX_DIR=Path("indexes")
INDEX_FILE=INDEX_DIR/"bm25_index.pkl"
INDEX_DIR.mkdir(parents=True,exist_ok=True)

def tokenize(text):
    return re.findall(r"c\+\+|c#|[a-z0-9]+",text.lower())

def load_chunks():
    if not CHUNKS_FILE.exists():
        raise FileNotFoundError(f"{CHUNKS_FILE} not found. Generate chunks first.")

    records=[]
    with CHUNKS_FILE.open("r",encoding="utf-8") as file:
        for line in file:
            if line.strip():
                record=json.loads(line)
                if record.get("text","").strip():
                    records.append(record)

    if not records:
        raise ValueError("No valid chunks found.")

    ids=[record["chunk_id"] for record in records]
    if len(ids)!=len(set(ids)):
        raise ValueError("Duplicate chunk IDs found.")

    return records

def build_bm25_index():
    print("Loading document chunks...")
    records=load_chunks()
    print(f"Preparing {len(records)} chunks for BM25...")

    tokenized_corpus=[tokenize(record["text"]) for record in records]
    bm25=BM25Okapi(tokenized_corpus)

    with INDEX_FILE.open("wb") as file:
        pickle.dump({"bm25":bm25,"records":records},file)

    print("BM25 index built successfully.")
    print(f"Indexed chunks: {len(records)}")
    print(f"Saved index: {INDEX_FILE}")

def load_bm25_index():
    if not INDEX_FILE.exists():
        raise FileNotFoundError("BM25 index not found. Run with --build first.")

    with INDEX_FILE.open("rb") as file:
        data=pickle.load(file)

    return data["bm25"],data["records"]

def keyword_search(query,top_k=5):
    if not query.strip():
        return []

    bm25,records=load_bm25_index()
    query_tokens=tokenize(query)

    if not query_tokens:
        return []

    scores=bm25.get_scores(query_tokens)
    ranked_indices=sorted(range(len(scores)),key=lambda i:scores[i],reverse=True)

    results=[]
    for index in ranked_indices[:min(top_k,len(records))]:
        score=float(scores[index])
        if score<=0:
            continue

        record=records[index]
        results.append({
            "chunk_id":record["chunk_id"],
            "title":record.get("title",""),
            "text":record["text"],
            "source_url":record.get("source_url",""),
            "score":score
        })

    return results

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--build",action="store_true",help="Build or rebuild the BM25 index")
    args=parser.parse_args()

    if args.build:
        build_bm25_index()
        return

    bm25,records=load_bm25_index()
    print(f"BM25 ready: {len(records)} chunks")

    while True:
        query=input("\nEnter a query (or 'exit'): ").strip()
        if query.lower()=="exit":
            break

        results=keyword_search(query,top_k=5)
        if not results:
            print("No matching chunks found.")
            continue

        for rank,result in enumerate(results,start=1):
            print(f"\nRank {rank}: {result['title']}")
            print(f"BM25 score: {result['score']:.4f}")
            print(f"Chunk ID: {result['chunk_id']}")
            print(f"Excerpt: {result['text'][:300]}")
            if result["source_url"]:
                print(f"Source: {result['source_url']}")

if __name__=="__main__":
    main()