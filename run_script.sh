# source run_env.sh

rm -rf tmp

# 4096 / 256 tpus = 128 / 8 tpus  
export TFDS_DATA_DIR=gs://kmh-gcp/tensorflow_datasets
python3 main.py \
    --workdir=./tmp \
    --config=configs/cfg_vit_large.py \
    --config.batch_size=128 \
    --config.log_every_steps=10 \
    --config.num_epochs=0.01 \
    --config.profile=False \
    --config.model.patches.size=\(16,16\)

    # --config.model.transformer.num_layers=2 \


# python3 test_profile.py

# pprof -http=localhost:6062 tmp/memory.prof
