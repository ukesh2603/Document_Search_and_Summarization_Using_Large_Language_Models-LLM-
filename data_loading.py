import wikipedia
import wikipediaapi
import os
import pandas as pd
from tqdm import tqdm

wiki=wikipediaapi.Wikipedia(language="en",user_agent="AIML-RAG-Dataset-Builder/1.0")

keywords=[
    "Artificial Intelligence",
    "Machine Learning",
    "Deep Learning",
    "Large Language Model",
    "Natural Language Processing",
    "Data Science",
    "Computer Vision",
    "Generative AI",
    "Transformer",
    "Neural Network",
    "Prompt Engineering",
    "Retrieval Augmented Generation",
    "Semantic Search",
    "Vector Database",
    "Feature Engineering"
]

def build_dataset(keywords,top_n=5):
    save_folder="knowledge_base"
    os.makedirs(save_folder,exist_ok=True)
    os.makedirs("metadata",exist_ok=True)

    articles=[]
    for keyword in keywords:
        results=wikipedia.search(keyword,results=top_n)
        articles.extend(results)

    articles=sorted(set(articles))
    metadata=[]

    for article in tqdm(articles):
        page=wiki.page(article)
        if page.exists():
            filename=article.replace("/","_").replace(" ","_")+".txt"
            filepath=os.path.join(save_folder,filename)

            with open(filepath,"w",encoding="utf-8") as f:
                f.write(page.text)

            metadata.append({
                "Article":article,
                "URL":page.fullurl,
                "Words":len(page.text.split())
            })

    metadata_df=pd.DataFrame(metadata)
    metadata_df.to_csv("metadata/metadata.csv",index=False)

    return metadata_df