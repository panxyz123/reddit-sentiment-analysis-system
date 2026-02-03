import json
import time
import psutil
import psycopg2
from kafka import KafkaConsumer
import os
import threading
from queue import Queue

# 1. initialize KafkaConsumer
consumer = KafkaConsumer(
    'reddit_sentiment_results',
    bootstrap_servers=['localhost:9092'],
    auto_offset_reset='earliest',         # Consume from the earliest info
    group_id='sentiment-sync-group',      # To faciliate future scaling
    value_deserializer=lambda x: json.loads(x.decode('utf-8'))
)

# Message queue for continuous syncing
message_queue = Queue()

def sync_to_postgres(data, current_cpu):
    """Helper function to sync data to PostgreSQL"""
    try:
        # connect postgreSQL
        postgres_user = os.getenv("POSTGRES_USER")
        postgres_password = os.getenv("POSTGRES_PASSWORD")
        conn = psycopg2.connect(f"dbname=your_db user={postgres_user} password={postgres_password} host=localhost")
        cur = conn.cursor()
        
        agg_map = {item['label'].lower(): item for item in data.get('agg', [])}
    
        keyword = data['keyword']

        insert_query = """
            INSERT INTO sentiment_history (
                keyword, source, sample_count, inference_duration,
                anger_pct, anger_conf, joy_pct, joy_conf, 
                sadness_pct, sadness_conf, fear_pct, fear_conf,
                love_pct, love_conf, surprise_pct, surprise_conf
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
    
        # prepare values
        def get_val(label, field):
            return agg_map.get(label, {}).get(field, 0)

        values = (
            keyword, data.get('source',None), data.get('count', 0), data.get('time', 0),
            get_val('anger', 'pct'), get_val('anger', 'avg_score'),
            get_val('joy', 'pct'), get_val('joy', 'avg_score'),
            get_val('sadness', 'pct'), get_val('sadness', 'avg_score'),
            get_val('fear', 'pct'), get_val('fear', 'avg_score'),
            get_val('love', 'pct'), get_val('love', 'avg_score'),
            get_val('surprise', 'pct'), get_val('surprise', 'avg_score')
        )

        cur.execute(insert_query, values)
        
        conn.commit()
        cur.close()
        conn.close()
        print(f"✅ [CPU {current_cpu}%] Synced keyword '{data['keyword']}' to PG (Cache Hit: {data.get('cache_hit')})")
        return True
        
    except Exception as e:
        print(f"❌ DB Sync Error: {e}")
        return False

def kafka_consumer_worker():
    """Worker thread to consume Kafka messages and queue them"""
    print("🎯 Kafka Consumer thread started...")
    for message in consumer:
        data = message.value
        message_queue.put(data)
        print(f"📥 Queued message for keyword: {data['keyword']} (Queue size: {message_queue.qsize()})")

def continuous_sync_worker(cpu_threshold=20.0):
    """Worker thread to continuously process queue when CPU is low"""
    print(f"⚙️  Continuous sync worker started. Monitoring CPU (Threshold: {cpu_threshold}%)...")
    
    while True:
        current_cpu = psutil.cpu_percent(interval=1)
        
        if current_cpu < cpu_threshold and not message_queue.empty():
            # CPU is low and queue has messages - process them
            try:
                data = message_queue.get(block=False)
                sync_to_postgres(data, current_cpu)
                message_queue.task_done()
            except Exception as e:
                print(f"❌ Error processing queue: {e}")
        elif current_cpu >= cpu_threshold and not message_queue.empty():
            print(f"⏳ CPU Busy ({current_cpu}%). Waiting to sync {message_queue.qsize()} queued messages...")
            time.sleep(2)
        else:
            # Queue is empty or CPU is low - just monitor
            time.sleep(1)

if __name__ == "__main__":
    cpu_threshold = 20.0  # Adjust this threshold as needed
    
    # Start Kafka consumer thread
    consumer_thread = threading.Thread(target=kafka_consumer_worker, daemon=True)
    consumer_thread.start()
    
    # Start continuous sync worker thread
    sync_thread = threading.Thread(target=continuous_sync_worker, args=(cpu_threshold,), daemon=True)
    sync_thread.start()
    
    print("🚀 Continuous sync system started (Ctrl+C to stop)")
    
    try:
        # Keep main thread alive
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Shutting down gracefully...")
        # Wait for queue to be processed
        message_queue.join()
        print("✅ All messages processed. Exiting.")