#!/usr/bin/env bash

# Start a nginx server for host ../docs/build/html
# This is useful for testing the docs locally

set -e
docker run --rm \
-it \
-p 8910:80 \
-v $(pwd)/../docs/build/html:/usr/share/nginx/html:ro \
nginx
