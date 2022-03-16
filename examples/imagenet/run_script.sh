source run_env.sh

export TFDS_DATA_DIR=gs://kmh-gcp/tensorflow_datasets

python3 main.py \
    --workdir=./imagenet_tpu \
    --config=configs/tpu.py \
    --config.batch_size=1024