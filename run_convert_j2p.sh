CONFIG=cfg_vit_large
# CHKPT_DIR='gs://kmh-gcp/checkpoints/flax/20220404_170716_kmh-tpuvm-v3-256-4_cfg_mae_large_1600ep_maeDBG_batch4096_vmap_normpix_sincos_initmaev2_cropv2'
CHKPT_DIR='gs://kmh-gcp/checkpoints/flax/20220415_011305_kmh-tpuvm-v3-256-3_cfg_mae_large_1600ep_maeDBG_batch4096_lr1.0e-4_vmap_normpix_sincos_initmaev2_cropv2ALTER_donate_olkNN_NOexClsDBG_masknoise_qkv_buf16x1024_noavelog_seed'

rm -rf tmp

# 4096 / 256 tpus = 128 / 8 tpus  
python3 main_convert.py \
    --mode=j2p \
    --workdir=./tmp \
    --config=configs/${CONFIG}.py \
    --config.batch_size=128 \
    --config.log_every_steps=10 \
    --config.num_epochs=0.005 \
    --config.model.classifier='token' \
    --config.pretrain_dir=${CHKPT_DIR} \
    --config.model.transformer.seperate_qkv=True \



# these are the template (run it in devfair)
# PYDIR='/checkpoint/kaiminghe/converted/2021-10-26-22-16-05-v3-128-mb4096-epo1600-PMAEp16-ViTLarge-lr1e-4-wd5e-2-warm40-mask0.75-pred8d512-exNB-msaLNmlpLNeLNpLNkBN0-1view-NOrelpos-abspos-clstoken-qkv-NOlayerscale-LNtgt-resume3/pretrained_lastnorm_tf2pt.pth'
# gsu cp ${PYDIR} gs://kmh-gcp/from_pytorch/template_pretrained_lastnorm_tf2pt.large.pth