import faiss
import numpy as np
import json
from transformers import AutoTokenizer, AutoModel
import torch
import sys

def embed_query(text):
    tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-large-en-v1.5")
    model = AutoModel.from_pretrained("BAAI/bge-large-en-v1.5")
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        model_output = model(**inputs)
    embedding = model_output.last_hidden_state[:, 0].squeeze().numpy()
    return embedding.astype("float32")

def format_time(seconds):
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m:02d}:{s:02d}"

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("使い方: python search_cli.py <YouTubeのURL> <検索クエリ>")
        sys.exit(1)

    youtube_url = sys.argv[1]
    query = sys.argv[2]

    query_vec = embed_query(query).reshape(1, -1)

    index = faiss.read_index("faiss.index")

    with open("metadata.json", "r", encoding="utf-8") as f:
        metadata = json.load(f)

    top_k = 5
    D, I = index.search(query_vec, top_k)

    for idx in I[0]:
        if idx < 0 or idx >= len(metadata):
            continue
        seg = metadata[idx]
        start_sec = seg["start"]
        end_sec = seg["end"]
        text = seg["text"]
        start_fmt = format_time(start_sec)
        end_fmt = format_time(end_sec)
        link = f"{youtube_url}&t={int(start_sec)}s"
        print(f"[{start_fmt} - {end_fmt}] {text}\n  → {link}\n")