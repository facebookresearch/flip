# select ViT-Large

# converted from TF=>PyTorch
# PRETRAIN_DIR='gs://kmh-gcp/from_pytorch/checkpoint/kaiminghe/converted/2021-10-26-03-09-46-v3-128-mb4096-epo800-PMAEp16-ViTLarge-lr1e-4-wd5e-2-warm40-mask0.75-pred8d512-exNB-msaLNmlpLNeLNpLNkBN0-1view-NOrelpos-abspos-clstoken-qkv-NOlayerscale-LNtgt_convert_pt2jax'
PRETRAIN_DIR='gs://kmh-gcp/from_pytorch/checkpoint/kaiminghe/converted/2021-10-26-22-16-05-v3-128-mb4096-epo1600-PMAEp16-ViTLarge-lr1e-4-wd5e-2-warm40-mask0.75-pred8d512-exNB-msaLNmlpLNeLNpLNkBN0-1view-NOrelpos-abspos-clstoken-qkv-NOlayerscale-LNtgt-resume3_convert_pt2jax'

# debugging, 800ep/1600ep
# PRETRAIN_DIR='gs://kmh-gcp/checkpoints/flax/20220331_020430_kmh-tpuvm-v3-256-3_cfg_mae_large_800ep_maeDBG_batch4096_vmap_normpix_sincos_initmaev1'
# PRETRAIN_DIR='gs://kmh-gcp/checkpoints/flax/20220401_111439_kmh-tpuvm-v3-256-4_cfg_mae_large_800ep_maeDBG_batch4096_vmap_normpix_sincos_initmaev1_randuniform_normimpl_cropv3'
# PRETRAIN_DIR='gs://kmh-gcp/checkpoints/flax/20220402_065729_kmh-tpuvm-v3-256-3_cfg_mae_large_800ep_maeDBG_batch4096_vmap_normpix_sincos_initmaev2_randuniform_normimpl_cropv3_qkvinit_patchinit'  # good
# PRETRAIN_DIR='gs://kmh-gcp/checkpoints/flax/20220401_210908_kmh-tpuvm-v3-128-2_cfg_mae_large_800ep_maeDBG_batch4096_vmap_normpix_sincos_initmaev2_randuniform_normimpl_cropv3_qkvinit_NOpatchinit'
# PRETRAIN_DIR='gs://kmh-gcp/checkpoints/flax/20220402_204718_kmh-tpuvm-v3-256-3_cfg_mae_large_800ep_maeDBG_batch4096_vmap_normpix_sincos_initmaev2'
# PRETRAIN_DIR='gs://kmh-gcp/checkpoints/flax/20220402_203809_kmh-tpuvm-v3-256-4_cfg_mae_large_1600ep_maeDBG_batch4096_vmap_normpix_sincos_initmaev2'  # 1600ep
# PRETRAIN_DIR='gs://kmh-gcp/checkpoints/flax/20220403_202910_kmh-tpuvm-v3-256-3_cfg_mae_large_800ep_maeDBG_batch4096_vmap_normpix_sincos_initmaev2_cropv2'
# PRETRAIN_DIR='gs://kmh-gcp/checkpoints/flax/20220403_224513_kmh-tpuvm-v3-256-4_cfg_mae_large_800ep_maeDBG_batch4096_vmap_normpix_sincos_initmaev2_cropv2_hostseed'
# PRETRAIN_DIR='gs://kmh-gcp/checkpoints/flax/20220403_203139_kmh-tpuvm-v3-128-1_cfg_mae_large_800ep_maeDBG_batch4096_vmap_normpix_sincos_initmaev2_cropv3sanity'
# PRETRAIN_DIR='gs://kmh-gcp/checkpoints/flax/20220404_170716_kmh-tpuvm-v3-256-4_cfg_mae_large_1600ep_maeDBG_batch4096_vmap_normpix_sincos_initmaev2_cropv2'  # 1600ep
# PRETRAIN_DIR='gs://kmh-gcp/checkpoints/flax/20220405_050556_kmh-tpuvm-v3-256-3_cfg_mae_large_800ep_maeDBG_batch4096_vmap_normpix_sincos_initmaev2_cropv2_donate_olkNN_syncbn_fixbug'
# PRETRAIN_DIR='gs://kmh-gcp/checkpoints/flax/20220406_195729_kmh-tpuvm-v3-256-4_cfg_mae_large_1600ep_maeDBG_batch4096_vmap_normpix_sincos_initmaev2_cropv2_donate_olkNN_NOexClsDBG'
# PRETRAIN_DIR='gs://kmh-gcp/checkpoints/flax/20220411_203139_kmh-tpuvm-v3-256-3_cfg_mae_large_1600ep_maeDBG_batch4096_vmap_normpix_sincos_initmaev2_cropv2_donate_olkNN_NOexClsDBG_shf320b'
# PRETRAIN_DIR='gs://kmh-gcp/checkpoints/flax/20220413_000736_kmh-tpuvm-v3-256-4_cfg_mae_large_1600ep_maeDBG_batch4096_lr1e-4_vmap_normpix_sincos_initmaev2_cropv2_donate_olkNN_NOexClsDBG_masknoise_qkv'
# PRETRAIN_DIR='gs://kmh-gcp/checkpoints/flax/20220413_030239_kmh-tpuvm-v3-128-1_cfg_mae_large_1600ep_maeDBG_batch4096_lr1e-4_vmap_normpix_sincos_initmaev2_cropv2_donate_olkNN_NOexClsDBG_masknoise_qkv_buf16x1024'
# PRETRAIN_DIR='gs://kmh-gcp/checkpoints/flax/20220415_011305_kmh-tpuvm-v3-256-3_cfg_mae_large_1600ep_maeDBG_batch4096_lr1.0e-4_vmap_normpix_sincos_initmaev2_cropv2ALTER_donate_olkNN_NOexClsDBG_masknoise_qkv_buf16x1024_noavelog_seed'
# PRETRAIN_DIR='gs://kmh-gcp/checkpoints/flax/20220416_053141_kmh-tpuvm-v3-256-3_cfg_mae_large_1600ep_maeDBG_batch4096_lr1.0e-4_vmap_normpix_sincos_initmaev2_cropv2ALTER_donate_olkNN_NOexClsDBG_masknoise_qkv_buf16x1024_seed'
# PRETRAIN_DIR='gs://kmh-gcp/checkpoints/flax/20220417_063852_kmh-tpuvm-v3-256-3_cfg_mae_large_1600ep_maeDBG_batch4096_lr1.0e-4_vmap_normpix_sincos_initmaev2_cropv2ALTER_donate_olkNN_NOexClsDBG_buf16x1024_seed'
# PRETRAIN_DIR='gs://kmh-gcp/checkpoints/flax/20220421_070850_kmh-tpuvm-v3-256-1_cfg_mae_large_maetf_1600ep_b4096_lr1.0e-4_staging'
# PRETRAIN_DIR='gs://kmh-gcp/checkpoints/flax/20220423_072315_kmh-tpuvm-v3-256-3_cfg_mae_large_maetf_1600ep_b4096_lr1.0e-4_TorchLoader_wseed100_dbg_save50_resume_sanity'
# PRETRAIN_DIR='gs://kmh-gcp/checkpoints/flax/20220423_054117_kmh-tpuvm-v3-256-4_cfg_mae_large_maetf_1600ep_b4096_lr1.0e-4_TorchLoader_wseed100_dbg_save50'
# PRETRAIN_DIR='gs://kmh-gcp/checkpoints/flax/20220422_170904_kmh-tpuvm-v3-256-1_cfg_mae_large_maetf_1600ep_b4096_lr1.0e-4_staging'
# PRETRAIN_DIR='gs://kmh-gcp/checkpoints/flax/20220507_175337_kmh-tpuvm-v3-256-4_cfg_mae_large_maetf_800ep_b4096_lr1.0e-4_mask0.75_TorchLoader_wseed100'  # 800ep

# debugging, 200ep
# PRETRAIN_DIR='gs://kmh-gcp/checkpoints/flax/20220402_015256_kmh-tpuvm-v3-256-4_cfg_mae_large_200ep_maeDBG_batch4096_vmap_normpix_sincos_initmaev2_randuniform_normimpl_cropv3_qkvinit_patchinit'

# explore
# PRETRAIN_DIR='gs://kmh-gcp/checkpoints/flax/20220507_232714_kmh-tpuvm-v3-256-1_cfg_mae_large_maetf_800ep_b4096_lr1.0e-4_mask0.75_TorchLoader_wseed100_cls15_dbgpos'
# PRETRAIN_DIR='gs://kmh-gcp/checkpoints/flax/20220508_184006_kmh-tpuvm-v3-256-1_cfg_mae_large_maetf_800ep_b4096_lr1.0e-4_mask0.75_TorchLoader_wseed100_cls79_dbgpos'
# PRETRAIN_DIR='gs://kmh-gcp/checkpoints/flax/20220509_045043_kmh-tpuvm-v3-256-4_cfg_mae_large_maetf_800ep_b4096_lr1.0e-4_mask0.75_TorchLoader_wseed100_cls147_dbgpos'
