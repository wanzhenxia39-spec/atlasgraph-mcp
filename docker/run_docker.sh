#!/bin/sh
set -eu
# Prepare config/config.yaml and a compatible local image before running.
TASK_PROJECT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
TASK_RAG_IMAGE=${RAG_IMAGE:-atlasgraph-mcp:local}
mkdir -p "$TASK_PROJECT_DIR/logs" "$TASK_PROJECT_DIR/upload"
docker run --rm \
  --name atlasgraph-mcp \
  -p 21032:21032 \
  -e PYTHONPATH=/root/rag/src \
  -v "$TASK_PROJECT_DIR/config:/root/rag/config" \
  -v "$TASK_PROJECT_DIR/logs:/root/rag/logs" \
  -v "$TASK_PROJECT_DIR/upload:/root/rag/upload" \
  -w /root/rag \
  --entrypoint python \
  "$TASK_RAG_IMAGE" src/rag/service/service.py --rag_config config/config.yaml
