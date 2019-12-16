#!/bin/bash
set -e

case "$1" in
    start)
        echo "Running Start"
        exec python predict.py -i samples/ -o outputs/ -c 4 -m segnet
        ;;
    *)
        exec "$@"
esac
