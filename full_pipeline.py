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
def embed_texts(segments):
    from transformers import AutoTokenizer, AutoModel
    import torch

    tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-large-en-v1.5")
    model = AutoModel.from_pretrained("BAAI/bge-large-en-v1.5")

    texts = [seg["text"] for seg in segments]
    print("🔢 テキストをベクトル化中...")

    embeddings = []
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            model_output = model(**inputs)
        sentence_embedding = model_output.last_hidden_state[:, 0]
        embeddings.append(sentence_embedding.squeeze().numpy())

    return embeddings

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
    embeddings = embed_texts(segments)
    save_to_faiss(np.array(embeddings).astype("float32"), segments)