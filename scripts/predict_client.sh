#!/usr/bin/env bash
KW=${1:-"AI"}
LIMIT=${2:-500}

curl -X POST "http://localhost:8000/analyze" \
  -H "Content-Type: application/json" \
  -d "{\"keyword\":\"$KW\",\"limit\":$LIMIT,\"return_plot\":true}" 