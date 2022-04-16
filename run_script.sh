rm -rf tmp

# 4096 / 256 tpus = 128 / 8 tpus  

# export TFDS_DATA_DIR=gs://kmh-gcp/tensorflow_datasets
# python3 main.py \
#     --workdir=./tmp \
#     --config=configs/cfg_vit_large.py \
#     --config.batch_size=128 \
#     --config.log_every_steps=10 \
#     --config.num_epochs=0.005 \
#     --config.profile_memory=True \
#     --config.ema=False \
#     --config.donate=True \
#     --config.aug.randerase.on=False \
#     --config.aug.randerase.prob=0.25 \
#     --config.rescale_head_init=0.001 \
#     --config.aug.mix.mixup=True \
#     --config.aug.mix.cutmix=True \
#     --config.aug.mix.switch_mode=mix_batch \
#     --config.aug.autoaug=randaugv2 \
#     --config.model.transformer.seperate_qkv=True \
#     --config.pretrain_dir='gs://kmh-gcp/checkpoints/flax/20220413_000736_kmh-tpuvm-v3-256-4_cfg_mae_large_1600ep_maeDBG_batch4096_lr1e-4_vmap_normpix_sincos_initmaev2_cropv2_donate_olkNN_NOexClsDBG_masknoise_qkv'

    # --config.model.transformer.num_layers=12 \
    # --config.model.hidden_size=768 \
    # --config.model.transformer.num_layers=2 \


source run_convert_j2p.sh
