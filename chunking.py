
from langchain_text_splitters import RecursiveCharacterTextSplitter


def fixed_chunk(text, chunk_size=1000, overlap=200):
    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("Invalid chunk size or overlap")

    chunks = []
    start = 0

    while start < len(text):
        chunks.append(text[start:start + chunk_size])
        start += chunk_size - overlap
    return chunks


# recursive_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200,separators=["\n\n", "\n", ". ", " ", ""])


# def recursive_chunk(text):
#     if not text or not text.strip():
#         return []

#     return recursive_splitter.split_text(text)
