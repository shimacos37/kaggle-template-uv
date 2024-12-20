for i in $(seq 0 4)
do
docker-compose run --rm -it dev \
    python components/train/main.py \
    preprocess.version=002 \
    train.data.n_fold=$i \
    train.data.dataset_class=MotionDataset \
    train.version=$(basename $0 .sh) \
    train.model.model_class=MotionMLPV2 \
    train.base.lr=0.0005 \
    train.trainer.max_epochs=2
done

docker-compose run --rm -it dev \
    python components/train/predict.py \
    preprocess.version=002 \
    train.data.dataset_class=MotionDataset \
    train.data.train_batch_size=8 \
    train.data.test_batch_size=8 \
    train.data.image_width=384 \
    train.data.image_height=384 \
    train.version=$(basename $0 .sh) \
    train.model.model_class=MotionMLPV2 \
    train.base.lr=0.0005
