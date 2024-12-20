docker-compose run --rm -it dev \
    python components/gbdt_train/main.py \
    preprocess.version=002 \
    gbdt.version=$(basename $0 .sh)
