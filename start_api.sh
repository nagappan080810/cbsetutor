#!/bin/bash
# Start CBSE RAG FastAPI server
# Usage: ./start_api.sh [port]

PORT=${1:-8000}

echo "🚀 Starting CBSE RAG API on port $PORT..."
echo "   Docs: http://localhost:$PORT/docs"
echo "   Health: http://localhost:$PORT/health"
echo ""

uvicorn api:app \
  --host 0.0.0.0 \
  --port $PORT \
  --reload \
  --log-level info
