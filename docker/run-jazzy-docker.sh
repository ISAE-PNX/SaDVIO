#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE=sadvio:jazzy
COMPOSE="$SCRIPT_DIR/docker-compose.yml"

# Build image if it doesn't exist or --build is passed
if [ "$1" = "--build" ] || [ -z "$(docker images -q $IMAGE 2>/dev/null)" ]; then
    echo "[sadvio] Building $IMAGE ..."
    docker build -t $IMAGE -f "$SCRIPT_DIR/Dockerfile" "$SCRIPT_DIR/../../.."
fi

# Allow X11 connections from Docker
xhost +local:docker

# Bring up the container (detached)
docker compose -f "$COMPOSE" up -d

echo "[sadvio] Container started. Attaching..."
docker compose -f "$COMPOSE" exec sadvio bash

# Revoke X11 access on exit
xhost -local:docker
