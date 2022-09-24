rm -rf tmp

# 4096 / 256 tpus = 128 / 8 tpus  

export TFDS_DATA_DIR=gs://kmh-gcp/tensorflow_datasets
python3 main.py \
    --workdir=./tmp \
    --config=configs/cfg_mae_dbg.py \
    --config.batch_size=32 \
    --config.log_every_steps=10 \
    --config.num_epochs=1000 \
    --config.profile_memory=True \
    --config.partitioning.num_partitions=1 \
    --config.opt_type=adamw \
    --config.opt_mu_dtype=float32 \
    --config.partitioning.partition_states=True \
    --config.aug.area_range=\(0.9\,1.0\) \
    --config.model.model_img.decoder.cross_attention=False \
    --config.model.model_txt.decoder.cross_attention=False \
    --config.model.model_img.decoder.on_use=False \
    --config.model.model_txt.decoder.on_use=False \
    --config.model.clr.clr_loss=True \
    --config.aug.txt.cls_token=False \
    --config.model.clr.proj_layers=1 \
    --config.model.clr.proj_dim_out=512 \
    --config.model.model_txt.decoder.loss_weight=1. \
    --config.model.model_img.mask_ratio=0.0 \
    --config.model.model_txt.mask_ratio=0.0 \
    --config.model.clr.tau_learnable=True \
    --config.model.clr.proj_out_bias=False \
    --config.model.model_img.ln_pre=True \

    # --config.eval_only=True \
    # --config.aug.eval_pad=0 \
    # --config.resume_dir='gs://kmh-gcp/checkpoints/flax/20220910_212550_maet5x_kmh-tpuvm-v3-512-1_cfg_mae_large_10000ep_b16384_lr4e-6_mk0.0txt0.0_s100_p1st_re1.0_laion_a0.5_clrtau_eval_512d1mlp' \

    # --config.resume_dir='gs://kmh-gcp/checkpoints/flax/20220910_190756_maet5x_kmh-tpuvm-v3-256-3_cfg_mae_base_10000ep_b32768_lr4e-6_mk0.0txt0.0_s100_p1st_re1.0_laion_a0.5_clrtau_eval_512d1mlp'\
    # --config.resume_dir='gs://kmh-gcp/checkpoints/flax/20220914_163137_maet5x_kmh-tpuvm-v3-256-2_cfg_mae_base_10000ep_b32768_lr4e-6_mk0.0txtNO_s100_p1st_re1.0_laion_a0.5_clrtau_eval_512d1mlp' \

    # --config.aug.txt.tokenizer=hf_clip \
    # --config.aug.txt.max_len=77 \
    # --config.model.model_txt.vocab_size=49408 \
    # --config.aug.txt.batch_process=True \
    # --config.model.model_txt.use_attention_mask=True \
    # --config.resume_dir='gs://kmh-gcp/checkpoints/flax/20220914_202349_maet5x_kmh-tpuvm-v3-256-3_cfg_mae_base_10000ep_b32768_lr4e-6_mk0.0txtNO_s100_p1st_re1.0_laion_a0.5_clrtau_eval_512d1mlp_hfclip77b_autoreg_wd0.2_b0.98' \

    




