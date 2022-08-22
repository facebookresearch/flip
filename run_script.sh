rm -rf tmp

# 4096 / 256 tpus = 128 / 8 tpus  

export TFDS_DATA_DIR=gs://kmh-gcp/tensorflow_datasets
python3 main.py \
    --workdir=./tmp \
    --config=configs/cfg_mae_dbg.py \
    --config.batch_size=16 \
    --config.log_every_steps=10 \
    --config.num_epochs=1000 \
    --config.profile_memory=True \
    --config.partitioning.num_partitions=1 \
    --config.opt_type=adamw \
    --config.opt_mu_dtype=float32 \
    --config.partitioning.partition_states=True \
    --config.aug.area_range=\(0.1\,1.0\) \
    --config.model.model_img.decoder.cross_attention=False \
    --config.model.model_txt.decoder.cross_attention=True \
    --config.model.model_img.decoder.on_use=False \
    --config.model.model_txt.decoder.on_use=True \
    --config.model.clr.clr_loss=False \
    --config.model.model_img.decoder.pool_x_part=True \
    --config.model.model_txt.decoder.loss_weight=0.01 \

    # --config.resume_dir='gs://kmh-gcp/checkpoints/flax/20220521_221137_scratch_kmh-tpuvm-v3-256-1_cfg_vit_large_50ep_fttl_b1024_wd0.3_lr1e-4_lrd1.0_dp0.2_warm20_s0_beta0.95_p16_dbgp16/checkpoint_62550'
    # --config.pretrain_dir='gs://kmh-gcp/from_pytorch/checkpoint/kaiminghe/converted/2021-10-26-22-16-05-v3-128-mb4096-epo1600-PMAEp16-ViTLarge-lr1e-4-wd5e-2-warm40-mask0.75-pred8d512-exNB-msaLNmlpLNeLNpLNkBN0-1view-NOrelpos-abspos-clstoken-qkv-NOlayerscale-LNtgt-resume3_convert_pt2jax'
    # --config.resume_dir='gs://kmh-gcp/checkpoints/flax/20220520_203852_scratch_kmh-tpuvm-v3-256-1_cfg_vit_large_50ep_fttl_b1024_wd0.3_lr1e-4_lrd1.0_dp0.2_warm20_s0_beta0.95_p1_hwrng_lrd/checkpoint_50040'




