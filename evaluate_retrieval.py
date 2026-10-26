import json
from pathlib import Path
from hybrid_search import load_resources,keyword_retrieve,semantic_retrieve,reciprocal_rank_fusion

TEST_FILE=Path("evaluation/retrieval_test_set.jsonl")
TOP_K_VALUES=[1,3,5]
CANDIDATE_K=50

def load_test_cases():
    with TEST_FILE.open("r",encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]

def document_key(result):
    return result.get("document_id") or result.get("title") or result.get("chunk_id")

def unique_document_ranking(results):
    ranking=[]
    seen=set()

    for result in results:
        key=document_key(result)
        if key not in seen:
            seen.add(key)
            ranking.append(key)

    return ranking

def evaluate_method(test_cases,retrieve):
    reciprocal_ranks=[]
    hits={k:0 for k in TOP_K_VALUES}

    for case in test_cases:
        results=retrieve(case["query"])
        ranking=unique_document_ranking(results)
        expected_ids=set(case.get("relevant_document_ids") or [case["expected_document_id"]])

        relevant_ranks=[rank for rank,doc_id in enumerate(ranking,start=1) if doc_id in expected_ids]
        first_rank=min(relevant_ranks) if relevant_ranks else None
        reciprocal_ranks.append(1/first_rank if first_rank is not None else 0)

        for k in TOP_K_VALUES:
            if any(doc_id in expected_ids for doc_id in ranking[:k]):
                hits[k]+=1

    total=len(test_cases)
    if total==0:
        raise ValueError("The test set is empty.")

    return {
        f"Hit@{k}":round(hits[k]/total*100,2)
        for k in TOP_K_VALUES
    }|{
        "MRR":round(sum(reciprocal_ranks)/total,4)
    }

def main():
    test_cases=load_test_cases()
    if not test_cases:
        raise ValueError("No test cases found.")

    bm25,records,collection,model=load_resources()

    def bm25_method(query):
        return keyword_retrieve(query,bm25,records,CANDIDATE_K)

    def semantic_method(query):
        return semantic_retrieve(query,collection,model,CANDIDATE_K)

    def hybrid_method(query):
        keyword_results=bm25_method(query)
        semantic_results=semantic_method(query)
        return reciprocal_rank_fusion(keyword_results,semantic_results)

    methods={
        "BM25":bm25_method,
        "Semantic":semantic_method,
        "Hybrid":hybrid_method
    }

    all_results={}
    for name,method in methods.items():
        metrics=evaluate_method(test_cases,method)
        all_results[name]=metrics

        print(f"\n{name}")
        for metric,value in metrics.items():
            suffix="%" if metric.startswith("Hit@") else ""
            print(f"  {metric}: {value}{suffix}")

    output_file=Path("evaluation/retrieval_metrics.json")
    output_file.parent.mkdir(parents=True,exist_ok=True)
    output_file.write_text(json.dumps(all_results,indent=2),encoding="utf-8")

    print(f"\nMetrics saved to {output_file}")

if __name__=="__main__":
    main()