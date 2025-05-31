import torch
from typing import List
# Ensure compatibility: define get_default_device if missing
if not hasattr(torch, "get_default_device"):
    torch.get_default_device = lambda: torch.device("cpu")

import subprocess
import faiss
import whisperx # WhisperXのインポート
import numpy as np
import json
import os
import sys
from tqdm import tqdm
import noisereduce as nr
import soundfile as sf

# Global cache for models to avoid reloading
_vad_model_cache = None
_vad_utils_cache = None
# WhisperXモデル用のグローバルキャッシュ
_whisperx_asr_model_cache = None
_whisperx_align_model_cache = None
_whisperx_align_metadata_cache = None
_whisperx_diarize_pipeline_cache = None

# Silero VADへ変更（精度向上）
def get_vad_segments(audio_path_or_array, sr_target=16000):
    global _vad_model_cache, _vad_utils_cache
    import torch
    import librosa

    if _vad_model_cache is None or _vad_utils_cache is None:
        print("🔊 Silero VADモデルをロード中...")
        _vad_model_cache, _vad_utils_cache = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                                             model='silero_vad',
                                                             force_reload=False, # Use cached model after first download
                                                             trust_repo=True) # Recommended for hub.load

    (get_speech_timestamps, _, _, _, _) = _vad_utils_cache

    if isinstance(audio_path_or_array, str):
        # オーディオの読み込み（指定されたサンプルレート、モノラル化）
        audio_np, sr_orig = librosa.load(audio_path_or_array, sr=None, mono=True) # Load original sr
        if sr_orig != sr_target:
            print(f"🎤 音声をリサンプリング中: {sr_orig}Hz -> {sr_target}Hz")
            audio_np = librosa.resample(audio_np, orig_sr=sr_orig, target_sr=sr_target)
        current_sr = sr_target
    elif isinstance(audio_path_or_array, np.ndarray):
        audio_np = audio_path_or_array
        # Assuming if a numpy array is passed, its sample rate is sr_target
        # This should be ensured by the caller or sr should be passed.
        # For this pipeline, get_vad_segments is called with a path first.
        current_sr = sr_target # Assume sr_target if ndarray
    else:
        raise ValueError("audio_path_or_arrayはファイルパス(str)またはNumpy配列である必要があります。")

    audio_tensor = torch.from_numpy(audio_np)

    # 音声区間の検出
    speech_timestamps = get_speech_timestamps(
        audio_tensor,
        _vad_model_cache,
        threshold=0.5,  # 感度調整（0.3-0.7の間で調整可能）
        sampling_rate=current_sr # VADモデルが期待するサンプルレート
    )
    # speech_timestamps は current_sr でのサンプルインデックスのリスト
    return speech_timestamps, audio_np, current_sr

# ========== ノイズ除去 ==========
def reduce_noise(audio_path: str, output_path: str):
    """
    音声ファイルからノイズを除去し、指定されたパスに保存する。
    """
    import librosa
    print(f"🔊 ノイズ除去中: {audio_path} ...")
    
    # librosaで音声を読み込み (16kHzモノラルに統一)
    audio, sr = librosa.load(audio_path, sr=16000, mono=True)
    
    # ノイズ除去を実行 (定常的なノイズを想定)
    reduced_noise_audio = nr.reduce_noise(y=audio, sr=sr, stationary=True)
    
    # 処理後の音声を保存
    sf.write(output_path, reduced_noise_audio, sr)
    print(f"✅ ノイズ除去済み音声を保存しました: {output_path}")

# ========== 2. Whisperで文字起こし ==========
def transcribe_audio(audio_path: str, language_code: str = "ja") -> List[dict]:
    """
    WhisperXを使用して音声ファイルから文字起こしと話者分離を行う。
    """
    global _whisperx_asr_model_cache, _whisperx_align_model_cache, \
           _whisperx_align_metadata_cache, _whisperx_diarize_pipeline_cache
    
    import torch # for device selection

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # CPUの場合、float32の方が安定または高速な場合がありますが、int8はメモリ効率が良いです
    compute_type = "float16" if device == "cuda" else "int8" 
    batch_size = 16 # GPUメモリに応じて調整

    print(f"🚀 WhisperX (large-v3) を使用して処理開始 (device: {device}, compute_type: {compute_type})")

    # 1. ASRモデルのロード
    if _whisperx_asr_model_cache is None:
        print(f"🤫 WhisperX ASRモデル (large-v3, lang={language_code or 'auto'}) をロード中...")
        # language_codeがNoneの場合、WhisperXは自動検出を試みます
        _whisperx_asr_model_cache = whisperx.load_model("large-v3", device, compute_type=compute_type, language=language_code)

    # 2. 音声ファイルのロード
    print(f"🎧 WhisperX用に音声ファイルをロード中: {audio_path}")
    audio_input = whisperx.load_audio(audio_path) # デフォルトで16kHzにリサンプル

    # 3. 文字起こし実行
    print("📝 WhisperXで文字起こし中...")
    result = _whisperx_asr_model_cache.transcribe(audio_input, batch_size=batch_size)
    transcription_language = result["language"]
    print(f"🌍 WhisperXが使用/検出した言語: {transcription_language}")

    # 4. アラインメントモデルのロードと実行 (正確な単語タイムスタンプのため)
    # アラインメントモデルは言語に依存するため、キャッシュも言語を考慮
    if _whisperx_align_model_cache is None or \
       _whisperx_align_metadata_cache is None or \
       _whisperx_align_metadata_cache.get("language_code") != transcription_language:
        print(f"🔄 WhisperX Alignモデル ({transcription_language}) をロード中...")
        try:
            _whisperx_align_model_cache, _whisperx_align_metadata_cache = whisperx.load_align_model(
                language_code=transcription_language, device=device
            )
            # キャッシュ情報に言語コードを保存
            if _whisperx_align_metadata_cache: # metadataがNoneでないことを確認
                 _whisperx_align_metadata_cache["language_code"] = transcription_language
        except Exception as e:
            print(f"❌ Alignモデルのロードに失敗しました ({transcription_language}): {e}")
            print("話者分離なしで処理を続行します。")
            # アラインメント失敗時のフォールバック
            aligned_result = {"segments": result["segments"]} # 元のセグメントを使用
    
    if _whisperx_align_model_cache: # アラインメントモデルが正常にロードされた場合
        print("🔄 WhisperXでアラインメント処理中...")
        aligned_result = whisperx.align(result["segments"], _whisperx_align_model_cache, _whisperx_align_metadata_cache, audio_input, device, return_char_alignments=False)
    else: # アラインメントモデルのロードに失敗した場合
        print("⚠️ アラインメントモデルが利用できないため、アラインメントをスキップします。")
        aligned_result = {"segments": result["segments"]}

    # 5. 話者分離の実行
    final_segments_for_chunking = aligned_result["segments"]
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        print("⚠️ Hugging Faceトークン (HF_TOKEN) が環境変数に設定されていません。話者分離はスキップされます。")
    else:
        if _whisperx_diarize_pipeline_cache is None:
            print("🗣️ WhisperX Diarization パイプラインをロード中 (pyannote.audio)...")
            from whisperx.diarize import DiarizationPipeline # 変更点
            _whisperx_diarize_pipeline_cache = DiarizationPipeline(use_auth_token=hf_token, device=device)
        
        print("🗣️ WhisperXで話者分離を実行中...")
        # DiarizationPipelineは音声ファイルパスまたはロードされたオーディオを受け取れます
        # audio_path (ノイズ除去後のファイル) を使用するのが一般的
        diarize_segments = _whisperx_diarize_pipeline_cache(audio_path) 
        
        print("🤝 話者情報を文字起こし結果に割り当て中...")
        result_with_speakers = whisperx.assign_word_speakers(diarize_segments, aligned_result)
        final_segments_for_chunking = result_with_speakers["segments"]

    # 6. チャンクの整形
    chunks = []
    print("💬 文字起こし結果をチャンクに整形中...")
    for seg in tqdm(final_segments_for_chunking, desc="チャンク整形"):
        chunks.append({
            "start": seg["start"],
            "end": seg["end"],
            "text": seg["text"].strip(),
            "speaker": seg.get("speaker", "UNKNOWN") # 話者情報がない場合はUNKNOWN
        })
    
    print(f"✅ {len(chunks)}件のチャンクに再構成しました（話者情報 {'あり' if hf_token and _whisperx_diarize_pipeline_cache else 'なし'}）")
    return chunks

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

# ========== 3. 埋め込み（ベクトル化） ==========
def embed_text(text):
    import torch
    # Ensure compatibility: define get_default_device if missing
    if not hasattr(torch, "get_default_device"):
        torch.get_default_device = lambda: torch.device("cpu")
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
    audio_filename = "output.mp3"

    download_audio(url, output_path=audio_filename)
    reduce_noise(audio_path=audio_filename, output_path=audio_filename) # 元のファイルを上書き
    segments = transcribe_audio(audio_path=audio_filename, language_code="ja") # 日本語を指定
    embeddings = [embed_text(seg["text"]).squeeze() for seg in tqdm(segments, desc="📝 テキスト埋め込み中")]
    embeddings = np.stack(embeddings).astype("float32")
    save_to_faiss(embeddings, segments)

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