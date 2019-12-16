#!/bin/bash
set -e

case "$1" in
    start)
        echo "Running Start"
        exec python train.py -i samples/ -o networks/ -c 4 -e 20 -m segnet -a start
        ;;
    *)
        exec "$@"
esac
