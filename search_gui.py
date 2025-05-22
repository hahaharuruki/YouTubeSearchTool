import streamlit as st
import faiss
import numpy as np
import json
from transformers import AutoTokenizer, AutoModel
import torch
from rank_bm25 import BM25Okapi

def embed_query(text: str) -> np.ndarray:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-large-en-v1.5")
    model = AutoModel.from_pretrained("BAAI/bge-large-en-v1.5").to(device)
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True
    ).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        embedding = outputs.last_hidden_state[:, 0].cpu().numpy()
    return embedding.astype("float32")

def format_time(seconds: float) -> str:
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m:02d}:{s:02d}"

def hybrid_search(query: str, index: faiss.Index, metadata, top_k: int = 5):
    query_vec = embed_query(query)
    D, I = index.search(query_vec.reshape(1, -1), top_k * 2)
    corpus = [doc["text"] for doc in metadata]
    bm25 = BM25Okapi(corpus)
    bm25_scores = bm25.get_scores(query.split())
    combined_scores = {}
    for i, idx in enumerate(I[0]):
        if idx >= 0 and idx < len(metadata):
            combined_scores[idx] = 1.0 / (1 + D[0][i])
    for idx, score in enumerate(bm25_scores):
        if idx in combined_scores:
            combined_scores[idx] += score * 0.5
    sorted_results = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
    return [idx for idx, _ in sorted_results[:top_k]]

# ---------------------- Streamlit GUI部分 ----------------------

st.title("YouTube文字起こし検索ツール（ハイブリッド検索）")

youtube_url = st.text_input("YouTubeのURLを入力してください")
query = st.text_input("検索クエリを入力してください", help="例: 数学の参考書のおすすめ")

top_k = st.slider("表示件数", 1, 10, 5)

if st.button("検索") and youtube_url and query:
    try:
        st.info("インデックスとメタデータを読み込み中...")
        index = faiss.read_index("faiss.index")
        with open("metadata.json", "r", encoding="utf-8") as f:
            metadata = json.load(f)
        st.info("検索実行中...")
        results = hybrid_search(query, index, metadata, top_k=top_k)
        st.success("検索完了！")
        for idx in results:
            if idx < 0 or idx >= len(metadata):
                continue
            seg = metadata[idx]
            start_sec = seg["start"]
            end_sec = seg["end"]
            text = seg["text"]
            start_fmt = format_time(start_sec)
            end_fmt = format_time(end_sec)
            link = f"{youtube_url}&t={int(start_sec)}s"
            st.markdown(f"### {start_fmt} - {end_fmt}")
            st.write(text)
            if "context" in seg:
                st.caption(f"📝 コンテキスト: {seg['context']}")
            st.markdown(f"[🔗 このタイミングでYouTube再生]({link})")
            st.markdown("---")
    except Exception as e:
        st.error(f"エラーが発生しました: {str(e)}")