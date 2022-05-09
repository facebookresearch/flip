# select ViT-Huge

# debugging, 800ep/1600ep
# PRETRAIN_DIR='gs://kmh-gcp/checkpoints/flax/20220419_051653_kmh-tpuvm-v3-256-1_cfg_mae_huge_1600ep_maeDBG_batch4096_lr1.0e-4_vmap_normpix_sincos_initmaev2_cropv2ALTER_donate_olkNN_NOexClsDBG_buf16x1024_seed'
# PRETRAIN_DIR='gs://kmh-gcp/checkpoints/flax/20220424_175203_kmh-tpuvm-v3-256-3_cfg_mae_huge_maetf_1600ep_b4096_lr1.0e-4_TorchLoader_wseed100'
# PRETRAIN_DIR='gs://kmh-gcp/checkpoints/flax/20220425_070228_kmh-tpuvm-v3-256-4_cfg_mae_huge_maetf_3200ep_b4096_lr1.0e-4_TorchLoader_wseed100'

# 0.85
# PRETRAIN_DIR='gs://kmh-gcp/checkpoints/flax/20220504_011721_kmh-tpuvm-v3-256-4_cfg_mae_huge_maetf_1600ep_b4096_lr1.0e-4_mask0.85_TorchLoader_wseed100'
PRETRAIN_DIR='gs://kmh-gcp/checkpoints/flax/20220502_060255_kmh-tpuvm-v3-256-3_cfg_mae_huge_maetf_6400ep_b4096_lr1.0e-4_mask0.85_TorchLoader_wseed100'
