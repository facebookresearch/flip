# rm -rf tmp

# 4096 / 256 tpus = 128 / 8 tpus  
# export TFDS_DATA_DIR=gs://kmh-gcp/tensorflow_datasets
# python3 main.py \
#     --workdir=./tmp \
#     --config=configs/cfg_vit_base.py \
#     --config.batch_size=128 \
#     --config.log_every_steps=10 \
#     --config.num_epochs=0.005 \
#     --config.profile_memory=True \
#     --config.model.patches.size=\(16,16\) \
#     --config.ema=True \
#     --config.donate=True \
#     --config.model.classifier='tgap' \
#     --config.pretrain_dir='gs://kmh-gcp/checkpoints/flax/20220331_030004_kmh-tpuvm-v3-128-2_cfg_mae_base_800ep_maeDBG_batch4096_vmap_normpix_sincos_initmaev1' \

    # --config.model.transformer.num_layers=12 \
    # --config.model.hidden_size=768 \
    # --config.model.transformer.num_layers=2 \


source run_convert_pt2jax.sh
