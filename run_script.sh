# source run_env.sh

rm -rf tmp
export TFDS_DATA_DIR=gs://kmh-gcp/tensorflow_datasets

python3 main.py \
    --workdir=./tmp \
    --config=configs/tpu_vit_dbg.py \
    --config.batch_size=1024 \
    --config.model.transformer.num_layers=2 \
    --config.log_every_steps=10