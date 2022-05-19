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
    --config.partitioning.num_partitions=1 \

    # --config.pretrain_dir='gs://kmh-gcp/from_pytorch/checkpoint/kaiminghe/converted/2021-10-26-22-16-05-v3-128-mb4096-epo1600-PMAEp16-ViTLarge-lr1e-4-wd5e-2-warm40-mask0.75-pred8d512-exNB-msaLNmlpLNeLNpLNkBN0-1view-NOrelpos-abspos-clstoken-qkv-NOlayerscale-LNtgt-resume3_convert_pt2jax'

    # --config.aug.area_range=0.9,1 \
    # --config.aug.aspect_ratio_range=0.8,1.2 \

    # --config.model.transformer.num_layers=12 \
    # --config.model.hidden_size=768 \
    # --config.model.transformer.num_layers=2 \


# source run_convert_j2p.sh
