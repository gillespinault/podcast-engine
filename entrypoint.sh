#!/bin/bash
set -e

echo "🚀 Starting Podcast Engine (API + Workers)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Configuration
WORKERS=${PODCAST_ENGINE_WORKERS:-3}
API_HOST=${PODCAST_ENGINE_API_HOST:-0.0.0.0}
API_PORT=${PODCAST_ENGINE_API_PORT:-8000}

echo "📊 Configuration:"
echo "  - API: uvicorn on ${API_HOST}:${API_PORT}"
echo "  - Workers: ${WORKERS} RQ workers"
echo "  - Redis: ${PODCAST_ENGINE_REDIS_HOST:-projects-redis}:${PODCAST_ENGINE_REDIS_PORT:-6379}"
echo ""

# Start API in background
echo "🌐 Starting API server..."
uvicorn app.main:app \
    --host "$API_HOST" \
    --port "$API_PORT" \
    --log-level info \
    --no-access-log &

API_PID=$!
echo "✓ API started (PID: $API_PID)"

# Wait for API to be ready
sleep 2

# Start RQ workers in background
echo ""
echo "👷 Starting $WORKERS RQ workers..."
for i in $(seq 1 $WORKERS); do
    python worker_entrypoint.py &
    WORKER_PID=$!
    echo "✓ Worker $i started (PID: $WORKER_PID)"
    sleep 1
done

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ Podcast Engine running"
echo "   API: http://${API_HOST}:${API_PORT}"
echo "   Workers: ${WORKERS} active"
echo "   Queue: podcast_processing"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "📝 Logs below (Ctrl+C to stop):"
echo ""

# Wait for all background processes
wait
