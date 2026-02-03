# utils.py — tools for grabbing tweets, plotting, base64 conversion, etc.
from typing import List, Optional
import matplotlib
import json
import time

matplotlib.use("Agg")  # non-GUI matplotlib
import matplotlib.pyplot as plt
import io, base64
import pandas as pd
import praw
import os
import boto3
import redis
from kafka import KafkaProducer
from kafka.errors import NoBrokersAvailable

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
S3_BUCKET = os.getenv("S3_BUCKET_NAME")

def get_redis_client():
    pool = redis.ConnectionPool(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
    return redis.Redis(connection_pool=pool)

def scrape_reddit(keyword: str, limit: Optional[int] = None , lang: str = "en", last = None) -> List[str]:
    reddit = praw.Reddit(
        client_id = os.getenv("REDDIT_CLIENT_ID"),
        client_secret = os.getenv("REDDIT_CLIENT_SECRET"),
        user_agent = 'Sentiment-Analysis'
    )

    posts = []
    for submission in reddit.subreddit("all").search(keyword, sort='new', limit=limit, params={"after": last}):
        posts.append(submission)

    return posts


def summarize_sentiments(predictions: List[dict]) -> pd.DataFrame:
    """
    reformat pipeline predictions to pandas DataFrame
    """
    df = pd.DataFrame(predictions)
    if df.empty:
        return pd.DataFrame()
    agg = df.groupby('label')['score'].agg(['count', 'mean']).reset_index().rename(columns={'mean':'avg_score'})
    total = agg['count'].sum()
    agg['pct'] = 100 * agg['count'] / (total if total > 0 else 1)
    return agg

def plot_sentiment_pie(agg_df) -> str:
    """receive aggregated DataFrame, return base64 PNG"""
    if agg_df is None or agg_df.empty:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "No data", ha="center")
    else:
        labels = agg_df['label'].tolist()
        sizes = agg_df['pct'].tolist()
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
        ax.axis('equal')
        ax.set_title("Sentiment distribution (%)")
    buf = io.BytesIO()  # create a temporary buffer
    fig.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)  # read the buffer from the beginning
    b64 = base64.b64encode(buf.read()).decode('utf-8')  # encode to base64
    return b64

def sync_model_from_s3(s3_prefix="model_raw/", local_path="./model_onnx_quantized"):
    """
    Sync model files from S3 to local path
    """
    if not S3_BUCKET:
        print("S3_BUCKET_NAME not set, skipping S3 sync.")
        return

    if os.path.exists(local_path) and os.listdir(local_path):
        print(f"Model files detected in {local_path}, skipping download.")
        return

    print(f"Syncing model from s3://{S3_BUCKET}/{s3_prefix}...")
    s3 = boto3.client('s3')
    
    # Download all objects under the specified prefix
    paginator = s3.get_paginator('list_objects_v2')
    for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=s3_prefix):
        for obj in page.get('Contents', []):
            rel_path = os.path.relpath(obj['Key'], s3_prefix)
            dest_path = os.path.join(local_path, rel_path)
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            s3.download_file(S3_BUCKET, obj['Key'], dest_path)
    print("Model synchronization complete.")

def create_kafka_producer():
    print("🚀 Attempting to connect to Kafka...")
    while True:
        try:
            producer = KafkaProducer(
                bootstrap_servers=['kafka:9092'],
                value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                api_version=(0, 10, 1), # 
                request_timeout_ms=5000
            )
            print("✅ Kafka Producer connected!")
            return producer
        except NoBrokersAvailable:
            print("⏳ Kafka Broker not available yet, retrying in 5 seconds...")
            time.sleep(5)
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            time.sleep(5)