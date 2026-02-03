import os
import time
import psutil
import requests
import redis
import json

# --- Configuration ---
KEYWORDS_FILE = "keywords.txt"
CPU_THRESHOLD = 35.0  
CHECK_INTERVAL = 15   
DEFAULT_LANG = "en"
API_URL = "http://sentiment-api:8000/analyze" 

# Initialize Redis
r = redis.Redis(host='redis_cache', port=6379, db=0, decode_responses=True)

def load_keywords():
    """Load keywords from external file."""
    if os.path.exists(KEYWORDS_FILE):
        with open(KEYWORDS_FILE, 'r') as f:
            words = [line.strip() for line in f if line.strip()]
        return words
    print(f"Keywords file '{KEYWORDS_FILE}' not found.")
    return []

def get_cpu_usage():
    return psutil.cpu_percent(interval=1)

def warm_up():
    print(f"🚀 [Cache Warmer] Service started.")
    
    while True:
        # reload to get latest keywords
        keywords = load_keywords()
        cpu = get_cpu_usage()
        print(f"📊 Current CPU Usage: {cpu}%")

        if cpu < CPU_THRESHOLD:
            for word in keywords:
                cache_key = f"{word}:{DEFAULT_LANG}"
                
                # Check if cache exists. 
                # If it exists, it must be < 1 hour old.
                if r.exists(cache_key):
                    ttl = r.ttl(cache_key)
                    print(f"⏭️  Cache for '{word}' is still fresh (Expires in {ttl}s). Skipping.")
                    continue
                
                print(f"🔥 Low CPU detected ({cpu}%). Warming up keyword: {word}")
                try:
                    payload = {
                        "keyword": word,  
                        "limit": 250,   
                        "lang": "en",    
                        "return_plot": False
                  }
                    # Your API should handle the logic to save to Redis with 1-hour TTL
                    response = requests.post(API_URL, json=payload, timeout=180)
                    
                    if response.status_code == 200:
                        print(f"✅ Successfully cached: {word}")
                    else:
                        print(f"⚠️  API Error [{response.status_code}] for: {word}")
                    
                    time.sleep(5) # Cooldown to prevent CPU spikes
                    
                except Exception as e:
                    print(f"❌ Failed to warm '{word}': {e}")
                
                if get_cpu_usage() > CPU_THRESHOLD + 10:
                    print("🛑 CPU usage increased. Pausing current cycle.")
                    break
        else:
            print(f"😴 CPU Busy ({cpu}%). Warmer is sleeping...")
        
        time.sleep(CHECK_INTERVAL)

if __name__ == "__main__":
    warm_up()