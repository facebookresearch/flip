echo 'code dir: '$STAGEDIR

# seed=0
batch=4096
lr=1e-4
ep=10000  # 400M * 30 / 1.28M = 9375

mask=0.0
mask_txt=0.0

txtw=0.1

tau=0.1

partitions=1

rescale=1.0

vitsize=base32
CONFIG=cfg_mae_${vitsize}

# _normpix_exwd_NOsplit_fastsave
JOBNAME=flax/$(date +%Y%m%d_%H%M%S)_maet5x_${VM_NAME}_${CONFIG}_${ep}ep_b${batch}_lr${lr}_mk${mask}txt${mask_txt}_s${seed}_p${partitions}st_re${rescale}_laion_a0.5_clr${tau}_eval
# RESUME='gs://kmh-gcp/checkpoints/flax/20220907_051106_maet5x_kmh-tpuvm-v3-512-1_cfg_mae_large_10000ep_b4096_lr1e-4_mk0.0txt0.0_s100_p1st_re1.0_laion_a0.5_NOMAE_NOCross_clr0.1_NOtxtcls_txtw0.1'
RESUME=''

WORKDIR=gs://kmh-gcp/checkpoints/${JOBNAME}
LOGDIR=/kmh_data/logs/${JOBNAME}
mkdir -p ${LOGDIR}
chmod 777 ${LOGDIR}

# source run_init_remote.sh

gcloud alpha compute tpus tpu-vm ssh ${VM_NAME} --zone europe-west4-a \
    --worker=all --command "
cd $STAGEDIR
git config --global --add safe.directory $STAGEDIR

echo Current commit: $(git show -s --format=%h)
echo Current dir: $(pwd)

export GOOGLE_APPLICATION_CREDENTIALS=~/gcp_credential.json
export TCMALLOC_LARGE_ALLOC_REPORT_THRESHOLD=8589934592
export TFDS_DATA_DIR=gs://kmh-gcp/tensorflow_datasets

source run_get_ssh_id.sh

python3 main.py \
    --workdir=${WORKDIR} \
    --config=configs/$CONFIG.py \
    --config.batch_size=${batch} \
    --config.log_every_steps=100 \
    --config.num_epochs=${ep} \
    --config.learning_rate=${lr} \
    --config.profile_memory=True \
    --config.model.model_img.transformer.rescale_init=${rescale} \
    --config.model.model_img.norm_pix_loss=True \
    --config.model.model_img.sincos=True \
    --config.model.model_img.mask_ratio=${mask} \
    --config.model.model_txt.mask_ratio=${mask_txt} \
    --config.seed_tf=${seed} \
    --config.seed_jax=${seed} \
    --config.seed_pt=${seed} \
    --config.partitioning.num_partitions=${partitions} \
    --config.opt_type=adamw \
    --config.opt_mu_dtype=float32 \
    --config.partitioning.partition_states=False \
    --config.resume_dir=${RESUME} \
    --config.aug.area_range=\(0.5\,1.0\) \
    --config.model.clr.tau=${tau} \
    --config.model.model_txt.decoder.cross_attention=False \
    --config.model.model_img.decoder.cross_attention=False \
    --config.model.model_txt.decoder.on_use=False \
    --config.model.model_img.decoder.on_use=False \
    --config.model.clr.clr_loss=True \
    --config.aug.txt.cls_token=False \
    --config.model.model_txt.decoder.loss_weight=${txtw} \
2>&1 | tee -a $LOGDIR/finetune_\$SSH_ID.log
" 2>&1 | tee -a $LOGDIR/finetune.log

echo ${VM_NAME}