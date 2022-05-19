rm -rf tmp

# 4096 / 256 tpus = 128 / 8 tpus  

export TFDS_DATA_DIR=gs://kmh-gcp/tensorflow_datasets
python3 main.py \
    --workdir=./tmp \
    --config=configs/cfg_vit_dbg.py \
    --config.batch_size=32 \
    --config.log_every_steps=10 \
    --config.num_epochs=1 \
    --config.profile_memory=True \
    --config.donate=True \
    --config.aug.randerase.on=True \
    --config.aug.randerase.prob=0.25 \
    --config.model.rescale_head_init=0.001 \
    --config.aug.mix.mixup=True \
    --config.aug.mix.cutmix=True \
    --config.aug.autoaug=autoaug \
    --config.model.transformer.torch_qkv=False \
    --config.eval_only=False \
    --config.model.classifier=token \
    --config.learning_rate_decay=1. \
    --config.partitioning.num_partitions=4 \



