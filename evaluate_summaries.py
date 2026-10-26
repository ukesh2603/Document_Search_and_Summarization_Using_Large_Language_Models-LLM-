import json
from pathlib import Path
from rouge_score import rouge_scorer
from summarizer import summarize_query,load_resources

TEST_FILE=Path("evaluation/summary_test_set.jsonl")
OUTPUT_FILE=Path("evaluation/summary_metrics.json")

def load_test_cases():
    with TEST_FILE.open("r",encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]

def main():
    test_cases=load_test_cases()
    if not test_cases:
        raise ValueError("The summary test set is empty.")

    bm25,records,collection,model=load_resources()
    scorer=rouge_scorer.RougeScorer(["rouge1","rouge2","rougeL"],use_stemmer=True)

    metric_totals={"rouge1":0.0,"rouge2":0.0,"rougeL":0.0}
    evaluated=[]

    for case in test_cases:
        query=case["query"]
        reference=case["reference_summary"]
        result=summarize_query(query,bm25,records,collection,model)
        generated=result["answer"]
        scores=scorer.score(reference,generated)

        for metric in metric_totals:
            metric_totals[metric]+=scores[metric].fmeasure

        evaluated.append({
            "query":query,
            "reference_summary":reference,
            "generated_summary":generated,
            "rouge":{
                name:{
                    "precision":round(value.precision,4),
                    "recall":round(value.recall,4),
                    "f1":round(value.fmeasure,4)
                }
                for name,value in scores.items()
            },
            "sources":result["sources"]
        })

        print(f"\nQuery: {query}")
        print(f"Reference: {reference}")
        print(f"Generated: {generated}")

        for name,score in scores.items():
            print(f"{name} F1: {score.fmeasure:.4f}")

    count=len(evaluated)
    averages={metric:round(total/count,4) for metric,total in metric_totals.items()}

    output={
        "number_of_queries":count,
        "average_rouge_f1":averages,
        "individual_results":evaluated
    }

    OUTPUT_FILE.parent.mkdir(parents=True,exist_ok=True)
    OUTPUT_FILE.write_text(json.dumps(output,indent=2,ensure_ascii=False),encoding="utf-8")

    print("\nAverage ROUGE F1 scores:")
    for metric,score in averages.items():
        print(f"{metric}: {score:.4f}")

    print(f"\nResults saved to {OUTPUT_FILE}")

if __name__=="__main__":
    main()