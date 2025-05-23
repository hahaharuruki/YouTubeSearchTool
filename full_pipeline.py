import subprocess
import whisper
import faiss
import numpy as np
import json
import os
import sys
from tqdm import tqdm
import noisereduce as nr
import librosa
import soundfile as sf
import torch
import whisperx

def extract_video_id(url):
    import re
    m = re.search(r"v=([a-zA-Z0-9_\-]+)", url)
    return m.group(1) if m else url

def get_video_title(youtube_url):
    import subprocess
    try:
        result = subprocess.run(
            ["yt-dlp", "--get-title", youtube_url],
            capture_output=True, text=True, check=True
        )
        return result.stdout.strip()
    except Exception:
        return ""

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

def denoise_audio(input_path="output.mp3", output_path="output_denoised.wav"):
    print("🧹 ノイズ除去中...")
    y, sr = librosa.load(input_path, sr=16000)
    reduced_noise = nr.reduce_noise(y=y, sr=sr)
    sf.write(output_path, reduced_noise, sr)
    print(f"✅ ノイズ除去済み音声を保存: {output_path}")

# ========== 2. WhisperXで文字起こし＋話者分離 ==========
def transcribe_audio_with_speaker(audio_path):
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        # トークンが見つからない場合、ユーザーに入力を促すか、エラーメッセージを表示して終了する
        # ここでは例としてNoneを渡すが、実際には適切なエラーハンドリングが必要
        print("警告: Hugging Faceのアクセストークンが環境変数 HF_TOKEN に設定されていません。")
        print("話者分離機能が正しく動作しない可能性があります。")
        # raise ValueError("Hugging Faceのアクセストークン(HF_TOKEN)が設定されていません。") # より厳格な場合
    print("📝 WhisperXで文字起こし＋話者分離中...")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. 文字起こし実行
    model = whisperx.load_model("large-v3", device, compute_type="float32")
    transcription_result = model.transcribe(audio_path) # 文字起こし結果 (辞書型)
    print(f"✅ 文字起こし完了: {len(transcription_result['segments'])}件のセグメントを取得")
    language_code = transcription_result["language"]
    print(f"🗣️ 検出された言語: {language_code}")

    # 2. アライメントモデルのロードと実行
    print("🔄 文字起こし結果のアライメント中...")
    try:
        align_model, metadata = whisperx.load_align_model(language_code=language_code, device=device)
        aligned_result = whisperx.align(transcription_result["segments"], align_model, metadata, audio_path, device)
        print(f"✅ アライメント完了: {len(aligned_result['segments'])}件のセグメントをアライメント")
        # メモリ解放
        del align_model
        import gc
        gc.collect()
    except Exception as e:
        print(f"⚠️ アライメント処理中にエラーが発生しました: {e}")
        print("アライメントなしで処理を続行します。")
        aligned_result = transcription_result # アライメント失敗時は元の文字起こし結果を使用

    # 3. 話者分離パイプラインの準備と実行
    diarize_model_instance = whisperx.diarize.DiarizationPipeline(device=device, use_auth_token=hf_token)
    print("🗣️ 話者分離を実行中...")
    # diarize_model_instance には音声ファイルパスのみを渡す
    diarization_annotation = diarize_model_instance(audio_path)
    # 4. アライメントされた文字起こし結果に話者情報を割り当て
    print("🔗 文字起こし結果に話者情報を割り当て中...")
    final_result_with_speakers = whisperx.assign_word_speakers(diarization_annotation, aligned_result)
    segments = final_result_with_speakers["segments"]
    print(f"✅ 話者ラベル付きセグメント数: {len(segments)}")
    return segments

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

    # Prepare a dict to map video_id to set of titles for existing metadata
    video_titles_map = {}
    for meta in existing_metadata:
        vid = meta.get("video_id")
        titles = meta.get("video_titles", [])
        if vid:
            if vid not in video_titles_map:
                video_titles_map[vid] = set()
            video_titles_map[vid].update(titles)

    # Filter new segments and embeddings to exclude duplicates
    new_metadata = []
    new_embeddings = []
    for seg, emb in zip(segments, embeddings):
        key = (seg["start"], seg["end"], seg["text"])
        if key not in existing_set:
            vid = seg.get("video_id")
            vtitle = seg.get("video_title", "")
            # Prepare video_titles list for this segment
            if vid:
                old_titles = video_titles_map.get(vid, set())
                updated_titles = set(old_titles)
                if vtitle:
                    updated_titles.add(vtitle)
                seg["video_titles"] = list(updated_titles)
                video_titles_map[vid] = updated_titles
            else:
                seg["video_titles"] = [vtitle] if vtitle else []
            new_metadata.append(seg)
            new_embeddings.append(emb)
            existing_set.add(key)
        else:
            # Even if duplicate, update video_titles in existing metadata if needed
            for meta in existing_metadata:
                if (meta["start"], meta["end"], meta["text"]) == key:
                    vid = seg.get("video_id")
                    vtitle = seg.get("video_title", "")
                    if vid:
                        old_titles = set(meta.get("video_titles", []))
                        if vtitle and vtitle not in old_titles:
                            old_titles.add(vtitle)
                            meta["video_titles"] = list(old_titles)
                    break

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
    video_id = extract_video_id(url)
    video_title = get_video_title(url)
    download_audio(url)
    denoise_audio("output.mp3", "output_denoised.wav")
    segments = transcribe_audio_with_speaker("output_denoised.wav")
    for seg in segments:
        seg["video_id"] = video_id
        seg["video_url"] = url
        seg["video_title"] = video_title
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