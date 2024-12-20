docker-compose run --rm -it dev \
    python components/gbdt_train/main.py \
    gbdt.version=$(basename $0 .sh)
