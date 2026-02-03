-- Create the sentiment analysis history table
-- This schema stores aggregated sentiment scores for each keyword query
CREATE TABLE IF NOT EXISTS sentiment_history (
    id SERIAL PRIMARY KEY,
    
    -- Search context
    keyword VARCHAR(100) NOT NULL,
    source VARCHAR(20),             -- Data origin: 'inference' (fresh) or 'cache' (Redis hit)
    sample_count INT,               -- Total number of posts analyzed (maps to 'count')
    inference_time FLOAT,           -- Duration of the model inference in seconds (maps to 'time')
    
    -- Aggregated Emotion Probabilities (Extracted from 'agg')
    -- Defaulting to 0 for cases where specific emotions are not detected
    joy_score FLOAT DEFAULT 0,
    joy_conf FLOAT DEFAULT 0,
    sadness_score FLOAT DEFAULT 0,
    sadness_conf FLOAT DEFAULT 0,
    anger_score FLOAT DEFAULT 0,
    anger_conf FLOAT DEFAULT 0,
    fear_score FLOAT DEFAULT 0,
    fear_conf FLOAT DEFAULT 0,
    love_score FLOAT DEFAULT 0,
    love_conf FLOAT DEFAULT 0,
    surprise_score FLOAT DEFAULT 0,
    surprise_conf FLOAT DEFAULT 0,
    
);

-- Create index on keyword and timestamp for optimized time-series analysis in Tableau
CREATE INDEX IF NOT EXISTS idx_keyword ON sentiment_history(keyword);
CREATE INDEX IF NOT EXISTS idx_created_at ON sentiment_history(created_at);