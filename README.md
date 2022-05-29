## MAE training in t5x

Written by Kaiming.

This branch: `t5x.dev.mae`. Support up to 10B ~ 15B MAE pre-training.

The fine-tuning branch is `t5x.dev.vit`.

### Getting Started
- **Warning**: This repo is under development and not well documented yet
- Check https://github.com/google/flax/tree/main/examples/imagenet for ImageNet R50 training in JAX and TPU VM setup.
- Check https://github.com/google-research/vision_transformer for the official ViT code (in JAX).
- Check https://github.com/google-research/t5x for the official t5x code (in JAX).
- See `run_script.sh` for an example command line to debug in "local" TPU VM (v3-8).
- See `run_remote.sh` for an example script to run in "remote" TPU VMs (like v3-256).
