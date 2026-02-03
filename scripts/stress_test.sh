#!/bin/bash

# Define API URL (Localhost since we test from Mac)
API_URL="http://localhost:8000/analyze"

# Define Keywords: Mixed for Cache Hit and Cache Miss
# Assuming NVIDIA and Tesla are already in Redis
KEYWORDS=("AI" "Intel" "Intel" "dog" "midterm" "quantum")

echo "🚀 [STRESS TEST] Starting concurrent burst of ${#KEYWORDS[@]} users..."
echo "--------------------------------------------------------"

start_total=$(date +%s%N)

# Trigger parallel requests
for i in "${!KEYWORDS[@]}"
do
    KW=${KEYWORDS[$i]}
    (
        start_req=$(date +%s%N)
        # Using -s to hide progress bar, -o to discard output body
        status_code=$(curl -s -o /dev/null -w "%{http_code}" -X POST "$API_URL" \
            -H "Content-Type: application/json" \
            -d "{\"keyword\":\"$KW\",\"limit\":250,\"lang\":\"en\"}")
        
        end_req=$(date +%s%N)
        
        # Calculate latency in milliseconds
        latency=$(( (end_req - start_req) / 1000000 ))
        
        if [ "$status_code" -eq 200 ]; then
            echo "👤 User $((i+1)) | Key: $KW | Latency: ${latency}ms | Status: ✅"
        else
            echo "👤 User $((i+1)) | Key: $KW | Latency: ${latency}ms | Status: ❌ ($status_code)"
        fi
    ) &
done

wait # Wait for all background requests to finish
end_total=$(date +%s%N)
total_duration=$(( (end_total - start_total) / 1000000 ))

echo "--------------------------------------------------------"
echo "🏁 Test Finished! Total Duration: ${total_duration}ms"