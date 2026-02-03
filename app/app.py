# app.py — FastAPI service（Reddit sentiment analysis with caching, return JSON + optional plot）
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager
import json
import torch
from typing import Optional
from utils import scrape_reddit, summarize_sentiments, plot_sentiment_pie, sync_model_from_s3, get_redis_client, create_kafka_producer
import time
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os
from optimum.onnxruntime import ORTModelForSequenceClassification
import onnxruntime as ort

# device selection: 0 for GPU, -1 for CPU
device = 0 if torch.cuda.is_available() else -1

# load raw model from S3 to local
MODEL_DIR = os.path.abspath("/app/model_onnx_quantized")
sync_model_from_s3(s3_prefix="model_onnx_quantized/", local_path=MODEL_DIR)
print("Using device:", "cuda" if device==0 else "cpu")
provider = "CUDAExecutionProvider" if device == 0 else "CPUExecutionProvider"

# Configure ONNX session options
session_options = ort.SessionOptions()
session_options.intra_op_num_threads = 1 
session_options.inter_op_num_threads = 1 
session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
print(f"✅ ONNX Session Options: intra_op={session_options.intra_op_num_threads}, inter_op={session_options.inter_op_num_threads}, mode=ORT_SEQUENTIAL")

# Load raw model
# model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)

# Load ONNX quantized model
model = ORTModelForSequenceClassification.from_pretrained(
    MODEL_DIR,
    export=False,            
    file_name="model_quantized.onnx", 
    provider=provider,
    session_options=session_options,
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

# initialize Redis client
redis_client = get_redis_client()

# Initialize Kafka producer
producer = create_kafka_producer()

# Lifespan event handler
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("✅ App starting up...")
    yield
    # Shutdown
    producer.flush()
    print("✅ Kafka producer flushed on shutdown")

app = FastAPI(title="Reddit Sentiment API", lifespan=lifespan)

class Query(BaseModel):
    keyword: str
    limit: Optional[int] = 500
    lang: Optional[str] = "en"
    return_plot: Optional[bool] = True

@app.get("/")
def root():
    return {"message": "Reddit Sentiment API running"}

@app.post("/analyze")
def analyze(q: Query):
    start = time.time()
    
    # Create cache key from query parameters
    cache_key = f"{q.keyword}:{q.lang}"
    
    # Try to retrieve from Redis
    try:
        cached_result = redis_client.get(cache_key)
        if cached_result:
            result = json.loads(cached_result)
            result["source"] = "cache"
            result["time"] = time.time() - start
            print(f"Cache hit for keyword: {q.keyword}")
            # Push the result to Kafka topic
            try:
                producer.send('reddit_sentiment_results', result)
            except Exception as e:
                print(f"Kafka push failed: {e}")
            return result
    except Exception as e:
        print(f"Redis retrieval failed: {e}")
    
    # If not in cache, proceed with inference
    try:
        posts_uptodate = []
        target_count = q.limit
        batch_size = 100
        last = None

        while len(posts_uptodate) < target_count:
            batch_size = min(batch_size, target_count - len(posts_uptodate))
            new_posts = scrape_reddit(keyword=q.keyword, limit=batch_size, lang=q.lang, last=last)

            if not new_posts:
                break

            last = new_posts[-1].fullname
            posts_uptodate.extend([submission.title + " " + submission.selftext for submission in new_posts])

            print(f"Fetched {len(posts_uptodate)} posts so far...")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch reddit posts: {e}")

    if not posts_uptodate:
        return {"keyword": q.keyword, "count": 0, "results": [], "time": time.time()-start}

    # batch inference with mini batching to control RAM usage
    try:
        mini_batch_size = 16  # Control RAM usage with mini batches
        preds1 = []
        
        for batch_idx in range(0, len(posts_uptodate), mini_batch_size):
            batch_texts = posts_uptodate[batch_idx:batch_idx + mini_batch_size]
            
            # Tokenize mini batch
            inputs = tokenizer(batch_texts, truncation=True, padding=True, 
                                return_tensors="pt", max_length=512)
            
            # Move to device if GPU is available
            if device == 0:
                inputs = {k: v.to("cuda") for k, v in inputs.items()}
            
            with torch.no_grad():  # Disable gradient computation
                outputs = model(**inputs)
            
            logits = outputs.logits
            batch_preds = [
                {
                    "label": f"LABEL_{logits[i].argmax().item()}",
                    "score": logits[i].softmax(dim=-1).max().item()
                }
                for i in range(len(batch_texts))
            ]
            preds1.extend(batch_preds)
            
            # Clear CUDA cache after each mini batch
            if device == 0:
                torch.cuda.empty_cache()
                
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model inference failed: {e}")

    # Change the labels
    Label_map = {'LABEL_0': 'Sadness', 'LABEL_1': 'Joy', 'LABEL_2': 'Love',
                 'LABEL_3': 'Anger', 'LABEL_4': 'Fear', 'LABEL_5': 'Surprise'}
    for p in preds1:
        p['label'] = Label_map.get(p['label'], p['label'])

    # Build results list; sort out confident results
    CONF_THRESHOLD = 0.6
    results1 = [{"text": t, "prediction": p} for t, p in zip(posts_uptodate, preds1) if p["score"] >= CONF_THRESHOLD]

    # aggregate (pd dataframe)
    agg1 = summarize_sentiments([r["prediction"] for r in results1])

    # optional plot
    plot_b641 = plot_sentiment_pie(agg1) if q.return_plot else None

    result = {
        "keyword": q.keyword,
        "count": len(posts_uptodate),
        "agg": agg1.to_dict(orient='records'),
        "plot_base64": plot_b641,
        "source": "inference",
        "time": time.time() - start
    }
    
    # Cache the result (with 1-hour expiration)
    try:
        redis_client.setex(cache_key, 3600, json.dumps(result, default=str))
        print(f"Cached result for keyword: {q.keyword}")
    except Exception as e:
        print(f"Redis caching failed: {e}")

    # Push the result to Kafka topic
    try:
        producer.send('reddit_sentiment_results', result)
    except Exception as e:
        print(f"Kafka push failed: {e}")
    
    return result


