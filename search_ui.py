import streamlit as st
import faiss
import numpy as np
import json

st.title("🎥 YouTube 動画検索ツール")

# YouTube URL 入力
youtube_url = st.text_input("YouTube動画のURLを入力してください", "")

# クエリ入力
query = st.text_input("検索クエリを入力してください", "")

from transformers import AutoTokenizer, AutoModel
import torch

def embed_query(text):
    tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-large-en-v1.5")
    model = AutoModel.from_pretrained("BAAI/bge-large-en-v1.5")
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        model_output = model(**inputs)
    embedding = model_output.last_hidden_state[:, 0].squeeze().numpy()
    return embedding.astype("float32")

if youtube_url and query:
    # クエリをベクトル化
    query_vec = embed_query(query).reshape(1, -1)

    # FAISSインデックスの読み込み
    try:
        index = faiss.read_index("faiss.index")
    except Exception as e:
        st.error(f"FAISSインデックスの読み込みに失敗しました: {e}")
        st.stop()

    # メタデータの読み込み
    try:
        with open("metadata.json", "r", encoding="utf-8") as f:
            metadata = json.load(f)
    except Exception as e:
        st.error(f"メタデータの読み込みに失敗しました: {e}")
        st.stop()

    # 類似ベクトル検索
    top_k = 5
    D, I = index.search(query_vec, top_k)

    def format_time(seconds):
        m = int(seconds // 60)
        s = int(seconds % 60)
        return f"{m:02d}:{s:02d}"

    st.subheader("検索結果")

    if len(I[0]) == 0 or I[0][0] == -1:
        st.write("該当する結果がありません。")
    else:
        for idx in I[0]:
            if idx < 0 or idx >= len(metadata):
                continue
            seg = metadata[idx]
            start_sec = seg["start"]
            end_sec = seg["end"]
            text = seg["text"]
            start_fmt = format_time(start_sec)
            end_fmt = format_time(end_sec)
            # YouTubeリンクにシーク時間を追加
            link = f"{youtube_url}&t={int(start_sec)}s"
            st.markdown(f"**[{start_fmt} - {end_fmt}]** [{text}]({link})")