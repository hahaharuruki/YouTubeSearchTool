CREATE TABLE IF NOT EXISTS video_frames (
    id SERIAL PRIMARY KEY,
    video_id VARCHAR(255) NOT NULL,
    frame_timestamp FLOAT NOT NULL,
    ocr_text TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(video_id, frame_timestamp)
);

CREATE INDEX IF NOT EXISTS idx_video_frames_video_id ON video_frames(video_id);
CREATE INDEX IF NOT EXISTS idx_video_frames_timestamp ON video_frames(frame_timestamp); 