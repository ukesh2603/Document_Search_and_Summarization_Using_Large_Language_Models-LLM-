import os
import re
import unicodedata
import hashlib
from bs4 import BeautifulSoup

input_folder="knowledge_base"
output_folder="cleaned_knowledge_base"

os.makedirs(output_folder,exist_ok=True)

def remove_html(text):
    return BeautifulSoup(text,"html.parser").get_text()

def normalize_unicode(text):
    return unicodedata.normalize("NFKC",text)

def clean_special_characters(text):
    text=re.sub(r"[^\w\s.,!?;:()'-]"," ",text)
    text=re.sub(r"\s+"," ",text)
    return text.strip()

seen=set()

def is_duplicate(text):
    file_hash=hashlib.md5(text.encode("utf-8")).hexdigest()
    if file_hash in seen:
        return True

    seen.add(file_hash)
    return False

for filename in os.listdir(input_folder):
    if not filename.endswith(".txt"):
        continue

    filepath=os.path.join(input_folder,filename)
    with open(filepath,"r",encoding="utf-8") as f:
        text=f.read()

    text=remove_html(text)
    text=normalize_unicode(text)
    text=clean_special_characters(text)

    if len(text.strip())==0:
        print(f"Skipped Empty File: {filename}")
        continue

    if is_duplicate(text):
        print(f"Duplicate Remove: {filename}")
        continue

    output_path=os.path.join(output_folder,filename)
    with open(output_path,"w",encoding="utf-8") as f:
        f.write(text)