import cv2
import numpy as np
from paddleocr import PaddleOCR
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import insert
import os
from urllib.parse import parse_qs, urlparse
import logging

# ロギングの設定
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# データベース接続設定
DATABASE_URL = 'postgresql:///youtube_search'  # ローカルホストへの接続を簡略化
engine = sa.create_engine(DATABASE_URL)

# PaddleOCRの初期化（日本語モデル使用）
ocr = PaddleOCR(use_angle_cls=True, lang='japan')

def get_video_id(url):
    """YouTubeのURLからvideo_idを抽出する"""
    parsed_url = urlparse(url)
    if parsed_url.hostname == 'youtu.be':
        return parsed_url.path[1:]
    if parsed_url.hostname in ('www.youtube.com', 'youtube.com'):
        if parsed_url.path == '/watch':
            return parse_qs(parsed_url.query)['v'][0]
    return None

def extract_frames_and_ocr(video_path, video_id):
    """動画から1秒ごとのフレームを抽出してOCR処理を行う"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"動画ファイルを開けませんでした: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = 0
    frames_data = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 1秒ごとのフレームのみ処理
        if frame_count % int(fps) == 0:
            timestamp = frame_count / fps
            logger.info(f"フレーム処理中: {timestamp}秒")

            # OCR処理
            try:
                result = ocr.ocr(frame, cls=True)
                if result and isinstance(result, list):
                    # OCRの結果を結合
                    texts = []
                    for line in result:
                        if not line:
                            continue
                        for word_info in line:
                            if isinstance(word_info, list) and len(word_info) >= 2:
                                text = word_info[1][0]  # 文字列を取得
                                if text and isinstance(text, str):
                                    texts.append(text)
                    
                    if texts:  # テキストが抽出された場合のみデータを追加
                        ocr_text = ' '.join(texts)
                        frames_data.append({
                            'video_id': video_id,
                            'frame_timestamp': timestamp,
                            'ocr_text': ocr_text
                        })
                        logger.debug(f"OCR結果: {ocr_text}")
            except Exception as e:
                logger.error(f"OCR処理エラー: {str(e)}")
                logger.error(f"エラー詳細: {type(e).__name__}")
                continue  # エラーが発生しても次のフレームの処理を続行

        frame_count += 1

    cap.release()
    return frames_data

def save_to_database(frames_data):
    """フレームデータをデータベースに保存"""
    if not frames_data:
        logger.warning("保存するフレームデータがありません")
        return

    logger.info(f"保存するフレームデータ: {len(frames_data)}件")
    logger.debug(f"最初のデータサンプル: {frames_data[0] if frames_data else 'なし'}")

    try:
        with engine.connect() as conn:
            for batch in chunks(frames_data, 100):  # バッチ処理
                logger.info(f"バッチ処理: {len(batch)}件のデータを保存中")
                try:
                    # シンプルなINSERT文を使用
                    stmt = sa.text("""
                        INSERT INTO video_frames (video_id, frame_timestamp, ocr_text)
                        VALUES (:video_id, :frame_timestamp, :ocr_text)
                        ON CONFLICT (video_id, frame_timestamp) DO NOTHING
                    """)
                    
                    # バッチ処理を実行
                    for data in batch:
                        conn.execute(stmt, data)
                    
                    conn.commit()
                    logger.info(f"バッチ処理完了: {len(batch)}件のデータを保存しました")
                except Exception as batch_error:
                    logger.error(f"バッチ処理エラー: {str(batch_error)}")
                    logger.error(f"問題のあるデータ: {batch}")
                    conn.rollback()
                    continue  # 次のバッチを試行
        
        logger.info(f"✅ 合計{len(frames_data)}件のフレームデータを保存しました")
    except Exception as e:
        logger.error(f"データベース保存エラー: {str(e)}")
        logger.error(f"エラーの種類: {type(e).__name__}")
        raise

def chunks(lst, n):
    """リストをn個ずつのチャンクに分割するユーティリティ関数"""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def process_video(video_path, url):
    """動画処理のメインフロー"""
    video_id = get_video_id(url)
    if not video_id:
        logger.error("有効なYouTube URLではありません")
        return

    logger.info(f"処理開始: video_id = {video_id}")
    frames_data = extract_frames_and_ocr(video_path, video_id)
    if frames_data:
        save_to_database(frames_data)
    logger.info("処理完了") 