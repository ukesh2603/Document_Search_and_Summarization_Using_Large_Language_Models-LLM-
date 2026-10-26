import json
import random
from pathlib import Path
from collections import defaultdict
from langchain_ollama import ChatOllama

CHUNKS_FILE=Path("processed_data/document_chunks.jsonl")
TEST_FILE=Path("evaluation/retrieval_test_set.jsonl")
MAX_TEST_DOCUMENTS=30
RANDOM_SEED=42

llm=ChatOllama(model="llama3",temperature=0)

def load_chunks():
    records=[]
    with CHUNKS_FILE.open("r",encoding="utf-8") as file:
        for line in file:
            if line.strip():
                record=json.loads(line)
                if record.get("text","").strip():
                    records.append(record)

    if not records:
        raise ValueError("No usable document chunks found.")

    return records

def generate_question(title,excerpts):
    prompt=f"""
Create ONE realistic information-seeking question that can be
answered using the supplied document excerpts.

Document title: {title}

Excerpts:
{excerpts}

Requirements:
- Ask about the document's actual content, not just its title.
- Make the question natural and specific.
- Do not answer the question.
- Return only the question, with no numbering or explanation.
"""

    response=llm.invoke(prompt)
    return response.content.strip().strip('"')

def main():
    TEST_FILE.parent.mkdir(parents=True,exist_ok=True)
    records=load_chunks()
    documents=defaultdict(list)

    for record in records:
        document_id=record.get("document_id") or record.get("title") or record["chunk_id"]
        documents[document_id].append(record)

    rng=random.Random(RANDOM_SEED)
    document_ids=sorted(documents.keys())
    rng.shuffle(document_ids)
    document_ids=document_ids[:MAX_TEST_DOCUMENTS]

    test_cases=[]
    seen_questions=set()

    for document_id in document_ids:
        chunks=documents[document_id]
        title=chunks[0].get("title") or str(document_id)
        excerpts="\n\n".join(chunk["text"][:1200] for chunk in chunks[:2])

        try:
            question=generate_question(title,excerpts)
        except Exception as error:
            print(f"Skipping {title}: {error}")
            continue

        if not question or question.lower() in seen_questions:
            continue

        seen_questions.add(question.lower())
        test_cases.append({
            "query":question,
            "expected_document_id":document_id,
            "expected_title":title,
            "relevant_document_ids":[document_id]
        })

        print(f"Created: {question}")
        print(f"Expected document: {title}\n")

    with TEST_FILE.open("w",encoding="utf-8") as file:
        for case in test_cases:
            file.write(json.dumps(case,ensure_ascii=False)+"\n")

    print(f"Saved {len(test_cases)} test cases to {TEST_FILE}")

if __name__=="__main__":
    main()