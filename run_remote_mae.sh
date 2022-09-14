echo 'code dir: '$STAGEDIR

# seed=0
batch=32768  # 4096, 8192, 16384, 32768
lr=4e-6  # MAE base lr: 1e-4; CLIP base lr: 5e-4/32768*256=3.90625e-06
ep=1600  # 10000  # 400M * 30 / 1.28M = 9375; 400M * 32 / 1.28M = 9375

mask=0.0
mask_txt=0.0

txtw=0

tau=0.01

partitions=1

rescale=1.0

vitsize=basev2
CONFIG=cfg_mae_${vitsize}

# _normpix_exwd_NOsplit_fastsave
JOBNAME=flax/$(date +%Y%m%d_%H%M%S)_maet5x_${VM_NAME}_${CONFIG}_${ep}ep_b${batch}_lr${lr}_mk${mask}txtNO_s${seed}_p${partitions}st_re${rescale}_laion_a0.5_clrtau_eval_512d1mlp_hfclip77b # _autoreg _wd0.2_b0.98
RESUME=''
# RESUME='gs://kmh-gcp/checkpoints/flax/20220907_051106_maet5x_kmh-tpuvm-v3-512-1_cfg_mae_large_10000ep_b4096_lr1e-4_mk0.0txt0.0_s100_p1st_re1.0_laion_a0.5_NOMAE_NOCross_clr0.1_NOtxtcls_txtw0.1'

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
    --config.model.clr.proj_layers=1 \
    --config.model.clr.proj_dim_out=512 \
    --config.model.clr.tau_learnable=True \
    --config.aug.txt.tokenizer=hf_clip \
    --config.aug.txt.max_len=77 \
    --config.model.model_txt.vocab_size=49408 \
    --config.aug.txt.batch_process=True \
    --config.model.model_txt.use_attention_mask=False \
2>&1 | tee -a $LOGDIR/finetune_\$SSH_ID.log
" 2>&1 | tee -a $LOGDIR/finetune.log

    # --config.opt.b2=0.98 \
    # --config.opt.weight_decay=0.2 \


echo ${VM_NAME}