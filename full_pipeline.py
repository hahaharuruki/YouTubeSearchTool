import subprocess
import whisper
import faiss
import numpy as np
import json
import os
import sys
from tqdm import tqdm

# ========== 1. 動画から音声をダウンロード ==========
def download_audio(youtube_url, output_path="output.mp3"):
    if os.path.exists(output_path):
        print(f"🗑️ 既存の音声ファイル {output_path} を削除します...")
        os.remove(output_path)

    print("🎧 音声をダウンロード中...")
    subprocess.run([
        "yt-dlp",
        "--no-cache-dir",
        "-x", "--audio-format", "mp3",
        "-o", output_path,
        youtube_url
    ])
    print(f"✅ 音声を保存しました: {output_path}")

# ========== 2. Whisperで文字起こし ==========
def transcribe_audio(audio_path):
    print("📝 Whisperで文字起こし中...")
    model = whisper.load_model("large-v3")  # "small" や "medium" に変えてもOK
    result = model.transcribe(audio_path)
    segments = result["segments"]
    print(f"✅ {len(segments)}件のセグメントを取得")

    # 10秒以上になるまでセグメントを連結
    chunks = []
    current_chunk = {
        "start": segments[0]["start"],
        "end": segments[0]["end"],
        "text": segments[0]["text"]
    }
    current_speaker = "Speaker 1"
    for seg in segments[1:]:
        current_chunk["end"] = seg["end"]
        current_chunk["text"] += " " + seg["text"]
        duration = current_chunk["end"] - current_chunk["start"]
        if duration >= 10:
            current_chunk["speaker"] = current_speaker
            chunks.append(current_chunk)
            current_speaker = "Speaker 2" if current_speaker == "Speaker 1" else "Speaker 1"
            current_chunk = {
                "start": seg["start"],
                "end": seg["end"],
                "text": seg["text"]
            }
    current_chunk["speaker"] = current_speaker
    chunks.append(current_chunk)

    print(f"✅ {len(chunks)}件のチャンクに再構成しました")
    return chunks

# ========== 3. 埋め込み（ベクトル化） ==========
def embed_text(text):
    import torch
    from transformers import AutoTokenizer, AutoModel
    
    # モデルの組み合わせ（重み付け）
    models = [
        ("BAAI/bge-large-en-v1.5", 0.6),  # 基本モデル
        ("intfloat/multilingual-e5-large", 0.4)  # 日本語対応補完
    ]
    
    final_embedding = None
    device = torch.device("cpu")  # MPSは不安定なのでCPU推奨
    
    for model_name, weight in models:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name).to(device)
        
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
        
        if final_embedding is None:
            final_embedding = embedding * weight
        else:
            # 次元が異なる場合の調整（必要に応じて）
            if embedding.shape != final_embedding.shape:
                from sklearn.preprocessing import normalize
                # リサイズして正規化
                embedding = normalize(embedding).reshape(final_embedding.shape)
            
            final_embedding += embedding * weight
        
        # メモリ解放
        del model, tokenizer
        import gc
        gc.collect()
    
    return final_embedding

# ========== 4. FAISS に格納 ==========
def save_to_faiss(embeddings, segments, index_path="faiss.index", meta_path="metadata.json"):
    # Load existing index if exists
    if os.path.exists(index_path):
        print(f"📂 既存のインデックス {index_path} を読み込み中...")
        index = faiss.read_index(index_path)
    else:
        dim = embeddings.shape[1]
        index = faiss.IndexFlatL2(dim)

    # Load existing metadata if exists
    if os.path.exists(meta_path):
        print(f"📂 既存のメタデータ {meta_path} を読み込み中...")
        with open(meta_path, "r", encoding="utf-8") as f:
            existing_metadata = json.load(f)
    else:
        existing_metadata = []

    # Prepare a set of existing entries for quick lookup to avoid duplicates
    existing_set = set((m["start"], m["end"], m["text"]) for m in existing_metadata)

    # Filter new segments and embeddings to exclude duplicates
    new_metadata = []
    new_embeddings = []
    for seg, emb in zip(segments, embeddings):
        key = (seg["start"], seg["end"], seg["text"])
        if key not in existing_set:
            new_metadata.append(seg)
            new_embeddings.append(emb)
            existing_set.add(key)

    if len(new_embeddings) > 0:
        new_embeddings_np = np.array(new_embeddings).astype("float32")
        index.add(new_embeddings_np)
        existing_metadata.extend(new_metadata)
        print(f"➕ {len(new_embeddings)}件の新しいベクトルとメタデータを追加しました")
    else:
        print("ℹ️ 新しいベクトルはありませんでした（重複なし）")

    print(f"💾 ベクトルを {index_path} に保存中...")
    faiss.write_index(index, index_path)

    print(f"💾 メタデータを {meta_path} に保存中...")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(existing_metadata, f, ensure_ascii=False, indent=2)

    print("✅ 保存完了")

# ========== 実行フロー ==========
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("使い方: python full_pipeline.py <YouTubeのURL>")
        sys.exit(1)

    url = sys.argv[1]
    download_audio(url)
    segments = transcribe_audio("output.mp3")
    embeddings = [embed_text(seg["text"]).squeeze() for seg in segments]
    embeddings = np.stack(embeddings).astype("float32")
    save_to_faiss(embeddings, segments)

# Silero VADへ変更（精度向上）
def get_vad_segments(audio_path):
    import torch
    model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                 model='silero_vad',
                                 force_reload=True)
    (get_speech_timestamps, _, _, _, _) = utils
    
    # オーディオの読み込み（16kHz、モノラル化）
    import librosa
    audio, sr = librosa.load(audio_path, sr=16000, mono=True)
    
    # 音声区間の検出
    speech_timestamps = get_speech_timestamps(
        torch.tensor(audio),
        model,
        threshold=0.5,  # 感度調整（0.3-0.7の間で調整可能）
        sampling_rate=16000
    )
    
    return speech_timestamps

# HNSWインデックスへの変更（精度と速度のバランス）
def create_improved_index(embeddings):
    import faiss
    import numpy as np
    
    dim = embeddings.shape[1]
    
    # HNSWインデックスの構築
    # M: グラフの次数（高いほど精度向上だがメモリ消費増加）
    # efConstruction: 構築時の探索幅（高いほど精度向上だが構築時間増加）
    index = faiss.IndexHNSWFlat(dim, M=32)
    index.hnsw.efConstruction = 80  # 構築時の探索幅
    index.hnsw.efSearch = 128  # 検索時の探索幅（高いほど精度向上）
    
    # 埋め込みを追加
    if len(embeddings) > 0:
        faiss.normalize_L2(embeddings)  # コサイン類似度のための正規化
        index.add(embeddings)
    
    return index

# クロスエンコーダーでリランキング（精度向上）
def rerank_results(query, results, metadata, top_k=5):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    import torch
    
    # 多言語クロスエンコーダー
    model_name = "amberoad/bert-multilingual-passage-reranking-msmarco"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    
    pairs = []
    ids = []
    
    for idx in results:
        if idx >= 0 and idx < len(metadata):
            text = metadata[idx]["text"]
            pairs.append([query, text])
            ids.append(idx)
    
    # バッチサイズを小さくして処理
    batch_size = 4
    scores = []
    
    for i in range(0, len(pairs), batch_size):
        batch = pairs[i:i+batch_size]
        inputs = tokenizer(
            batch, 
            padding=True, 
            truncation=True, 
            return_tensors="pt", 
            max_length=512
        )
        
        with torch.no_grad():
            outputs = model(**inputs)
            batch_scores = outputs.logits.squeeze(-1).tolist()
            scores.extend(batch_scores)
    
    # スコアと元のインデックスでソート
    scored_results = sorted(zip(ids, scores), key=lambda x: x[1], reverse=True)
    return [idx for idx, _ in scored_results[:top_k]]