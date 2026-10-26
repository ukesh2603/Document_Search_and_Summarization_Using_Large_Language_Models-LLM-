import json
import pandas as pd
from pathlib import Path
from chunking import fixed_chunk

INPUT_DIR=Path("cleaned_knowledge_base")
METADATA_FILE=Path("metadata/metadata.csv")
OUTPUT_DIR=Path("processed_data")
OUTPUT_FILE=OUTPUT_DIR/"document_chunks.jsonl"
OUTPUT_DIR.mkdir(parents=True,exist_ok=True)

def load_source_urls():
    url_map={}

    if not METADATA_FILE.exists():
        print("Warning: metadata CSV not found.")
        return url_map

    df=pd.read_csv(METADATA_FILE)

    if not {"Article","URL"}.issubset(df.columns):
        print("Warning: expected Article and URL columns.")
        return url_map

    for _,row in df.iterrows():
        filename=str(row["Article"]).replace("/","_").replace(" ","_")+".txt"
        url_map[filename]=str(row["URL"])

    return url_map

def prepare_chunks():
    if not INPUT_DIR.exists():
        raise FileNotFoundError(f"Folder not found: {INPUT_DIR}")

    urls=load_source_urls()
    document_count=0
    chunk_count=0

    with OUTPUT_FILE.open("w",encoding="utf-8") as output:
        for path in sorted(INPUT_DIR.glob("*.txt")):
            text=path.read_text(encoding="utf-8").strip()
            if not text:
                continue

            chunks=fixed_chunk(text)
            title=path.stem.replace("_"," ")

            for index,chunk in enumerate(chunks):
                record={
                    "chunk_id":f"{path.stem}_{index}",
                    "document_id":path.stem,
                    "chunk_index":index,
                    "title":title,
                    "text":chunk,
                    "source_url":urls.get(path.name,""),
                    "source":"Wikipedia"
                }

                output.write(json.dumps(record,ensure_ascii=False)+"\n")
                chunk_count+=1

            document_count+=1

    print(f"Documents processed: {document_count}")
    print(f"Chunks generated: {chunk_count}")
    print(f"Output: {OUTPUT_FILE.resolve()}")

if __name__=="__main__":
    prepare_chunks()