import faiss
import numpy as np
import json
from transformers import AutoTokenizer, AutoModel
import torch
import sys
from rank_bm25 import BM25Okapi
from typing import List, Dict, Tuple

def embed_query(text: str) -> np.ndarray:
    """クエリをベクトル化する"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # モデルとトークナイザーの初期化
    tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-large-en-v1.5")
    model = AutoModel.from_pretrained("BAAI/bge-large-en-v1.5").to(device)
    
    # テキストの前処理とエンコード
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True
    ).to(device)
    
    # ベクトル化
    with torch.no_grad():
        outputs = model(**inputs)
        embedding = outputs.last_hidden_state[:, 0].cpu().numpy()
    
    return embedding.astype("float32")

def format_time(seconds: float) -> str:
    """秒数を MM:SS 形式にフォーマット"""
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m:02d}:{s:02d}"

def hybrid_search(query: str, index: faiss.Index, metadata: List[Dict], top_k: int = 5) -> List[int]:
    """ハイブリッド検索を実行"""
    print("🔍 検索実行中...")
    
    # セマンティック検索
    query_vec = embed_query(query)
    D, I = index.search(query_vec.reshape(1, -1), top_k * 2)
    
    # キーワード検索
    corpus = [doc["text"] for doc in metadata]
    bm25 = BM25Okapi(corpus)
    bm25_scores = bm25.get_scores(query.split())
    
    # スコアの組み合わせ
    combined_scores = {}
    
    # FAISSの結果を処理
    for i, idx in enumerate(I[0]):
        if idx >= 0 and idx < len(metadata):
            combined_scores[idx] = 1.0 / (1 + D[0][i])
    
    # BM25のスコアを追加
    for idx, score in enumerate(bm25_scores):
        if idx in combined_scores:
            combined_scores[idx] += score * 0.5  # BM25のスコアに重み付け
    
    # 結果のソート
    sorted_results = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
    return [idx for idx, _ in sorted_results[:top_k]]

def main():
    # コマンドライン引数のチェック
    if len(sys.argv) < 3:
        print("使い方: python search_cli.py <YouTubeのURL> <検索クエリ>")
        sys.exit(1)

    youtube_url = sys.argv[1]
    query = sys.argv[2]

    try:
        # インデックスとメタデータの読み込み
        print("📂 インデックスとメタデータを読み込み中...")
        index = faiss.read_index("faiss.index")
        with open("metadata.json", "r", encoding="utf-8") as f:
            metadata = json.load(f)

        # 検索の実行
        results = hybrid_search(query, index, metadata)

        # 結果の表示
        print("\n🎯 検索結果:")
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
            
            print(f"\n[{start_fmt} - {end_fmt}] {text}")
            if "context" in seg:
                print(f"📝 コンテキスト: {seg['context']}")
            print(f"🔗 {link}\n")

    except Exception as e:
        print(f"エラーが発生しました: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 