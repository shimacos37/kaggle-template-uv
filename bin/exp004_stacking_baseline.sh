docker-compose run --rm -it dev python components/stacking/main.py \
    preprocess.version='002' \
    stacking.version=$(basename $0 .sh) \
    stacking.models.nn="['exp003_nn_baseline']" \
    stacking.models.gbdt="['exp002_001_add_scene_feature']" \
    stacking.use_only_preds=False \
    stacking.use_pseudo_label=False
