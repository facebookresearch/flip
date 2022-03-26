# source run_env.sh

rm -rf tmp

export TFDS_DATA_DIR=gs://kmh-gcp/tensorflow_datasets
python3 main.py \
    --workdir=./tmp \
    --config=configs/cfg_vit_dbg.py \
    --config.batch_size=1024 \
    --config.model.transformer.num_layers=2 \
    --config.log_every_steps=10 \
    --config.num_epochs=0.1

# python3 test_profile.py
