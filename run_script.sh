rm -rf tmp

# 4096 / 256 tpus = 128 / 8 tpus  

export TFDS_DATA_DIR=gs://kmh-gcp/tensorflow_datasets
python3 main.py \
    --workdir=./tmp \
    --config=configs/cfg_mae_base.py \
    --config.batch_size=128 \
    --config.log_every_steps=10 \
    --config.num_epochs=1000 \
    --config.profile_memory=True \
    --config.partitioning.num_partitions=1 \
    --config.opt_type=adamw \
    --config.opt_mu_dtype=float32 \
    --config.partitioning.partition_states=True \
    --config.aug.area_range=\(0.1\,1.0\) \
    --config.model.model_img.decoder.cross_attention=False \
    --config.model.model_txt.decoder.cross_attention=False \
    --config.model.model_img.decoder.on_use=False \
    --config.model.model_txt.decoder.on_use=False \
    --config.model.clr.proj_layers=1 \
    --config.model.clr.proj_dim_out=512 \
    --config.model.clr.clr_loss=True \
    --config.model.model_txt.decoder.loss_weight=1. \
    --config.model.model_img.mask_ratio=0.0 \
    --config.model.model_txt.mask_ratio=0.0 \
    --config.model.clr.tau_learnable=True \
    --config.aug.txt.tokenizer=hf_clip \
    --config.aug.txt.max_len=77 \
    --config.model.model_txt.vocab_size=49408 \
    --config.aug.txt.batch_process=True \
    --config.model.model_txt.use_attention_mask=True \
    --config.eval_only=True \
    --config.aug.eval_pad=0 \
    --config.resume_dir='gs://kmh-gcp/checkpoints/flax/20220914_202349_maet5x_kmh-tpuvm-v3-256-3_cfg_mae_base_10000ep_b32768_lr4e-6_mk0.0txtNO_s100_p1st_re1.0_laion_a0.5_clrtau_eval_512d1mlp_hfclip77b_autoreg_wd0.2_b0.98' \
    
    # --config.aug.txt.tokenizer=hf_clip \
    # --config.aug.txt.max_len=77 \
    # --config.model.model_txt.vocab_size=49408 \

    # --config.resume_dir='gs://kmh-gcp/checkpoints/flax/20220521_221137_scratch_kmh-tpuvm-v3-256-1_cfg_vit_large_50ep_fttl_b1024_wd0.3_lr1e-4_lrd1.0_dp0.2_warm20_s0_beta0.95_p16_dbgp16/checkpoint_62550'
    # --config.pretrain_dir='gs://kmh-gcp/from_pytorch/checkpoint/kaiminghe/converted/2021-10-26-22-16-05-v3-128-mb4096-epo1600-PMAEp16-ViTLarge-lr1e-4-wd5e-2-warm40-mask0.75-pred8d512-exNB-msaLNmlpLNeLNpLNkBN0-1view-NOrelpos-abspos-clstoken-qkv-NOlayerscale-LNtgt-resume3_convert_pt2jax'
    # --config.resume_dir='gs://kmh-gcp/checkpoints/flax/20220520_203852_scratch_kmh-tpuvm-v3-256-1_cfg_vit_large_50ep_fttl_b1024_wd0.3_lr1e-4_lrd1.0_dp0.2_warm20_s0_beta0.95_p1_hwrng_lrd/checkpoint_50040'




