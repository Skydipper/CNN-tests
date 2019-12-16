#!/bin/bash

case "$1" in
    generate_data)
        type docker-compose >/dev/null 2>&1 || { echo >&2 "docker-compose is required but it's not installed.  Aborting."; exit 1; }
        docker-compose -f docker-compose.yml build && docker-compose -f docker-compose.yml up gen_data
        ;;
    preprocess)
	type docker-compose >/dev/null 2>&1 || { echo >&2 "docker-compose is required but it's not installed.  Aborting."; exit 1; }
	docker-compose -f docker-compose.yml build && docker-compose -f docker-compose.yml up preprocess
	;;
    train)
	type docker-compose >/dev/null 2>&1 || { echo >&2 "docker-compose is required but it's not installed.  Aborting."; exit 1; }
	docker-compose -f docker-compose.yml build && docker-compose -f docker-compose.yml up train
	;;
    predict)
	type docker-compose >/dev/null 2>&1 || { echo >&2 "docker-compose is required but it's not installed.  Aborting."; exit 1; }
	docker-compose -f docker-compose.yml build && docker-compose -f docker-compose.yml up predict
	;;

    *)
	
	echo "Usage: train_model.sh {generate_data|preprocess|train|predict}" >&2
	exit 1
	;;
esac

exit 0
