CONFIG=cfg_vit_large
# CHKPT_DIR='/checkpoint/kaiminghe/converted/2021-10-26-03-09-46-v3-128-mb4096-epo800-PMAEp16-ViTLarge-lr1e-4-wd5e-2-warm40-mask0.75-pred8d512-exNB-msaLNmlpLNeLNpLNkBN0-1view-NOrelpos-abspos-clstoken-qkv-NOlayerscale-LNtgt/pretrained_lastnorm_tf2pt.pth'
CHKPT_DIR='/checkpoint/kaiminghe/converted/2022-04-02-03-06-12-v3-128-mb4096-epo200-PMAEp16-ViTLarge-lr1.5e-4-wd5e-2-warm40-mask0.75-pred8d512-exNB-msaLNmlpLNeLNpLNkBN0-1view-NOrelpos-abspos-clstoken-qkv-NOlayerscale-LNtgteps1e-6/pretrained_lastnorm_tf2pt.pth'



GCP_CHKPT_DIR=gs://kmh-gcp/from_pytorch$CHKPT_DIR

# gsu cp $CHKPT_DIR $GCP_CHKPT_DIR


rm -rf tmp

# 4096 / 256 tpus = 128 / 8 tpus  
export TFDS_DATA_DIR=gs://kmh-gcp/tensorflow_datasets
python3 main_convert.py \
    --workdir=./tmp \
    --config=configs/${CONFIG}.py \
    --config.batch_size=128 \
    --config.log_every_steps=10 \
    --config.num_epochs=0.005 \
    --config.model.classifier='token' \
    --config.pretrain_dir=${GCP_CHKPT_DIR} \
