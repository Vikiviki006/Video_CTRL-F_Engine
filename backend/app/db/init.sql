CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS videos (
    id UUID PRIMARY KEY,
    filename VARCHAR(512) NOT NULL,
    original_filename VARCHAR(512) NOT NULL,
    path VARCHAR(1024) NOT NULL,
    duration_seconds FLOAT,
    fps FLOAT,
    frame_count INTEGER,
    status VARCHAR(32) NOT NULL DEFAULT 'queued',
    error_message TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS video_frames (
    id UUID PRIMARY KEY,
    video_id UUID NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
    timestamp_seconds FLOAT NOT NULL,
    frame_number INTEGER NOT NULL,
    thumbnail_path VARCHAR(1024) NOT NULL,
    embedding vector(1152),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_video_frames_video_id ON video_frames(video_id);
CREATE INDEX IF NOT EXISTS idx_video_frames_embedding
    ON video_frames USING hnsw (embedding vector_cosine_ops);
