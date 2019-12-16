#!/bin/bash
set -e

case "$1" in
    start)
        echo "Running Start"
        exec python main.py
        ;;
    *)
        exec "$@"
esac
