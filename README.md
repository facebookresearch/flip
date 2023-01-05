## Scaling Language-Image Pre-training via Masking

This repository contains the official JAX implementation of **FLIP**, as described in the paper [Scaling Language-Image Pre-training via Masking](https://arxiv.org/abs/2212.00794)

```
@article{li2022scaling,
  title={Scaling Language-Image Pre-training via Masking},
  author={Li, Yanghao and Fan, Haoqi and Hu, Ronghang and Feichtenhofer, Christoph and He, Kaiming},
  journal={arXiv preprint arXiv:2212.00794},
  year={2022}
}
```

* The implementation is based on JAX and the models are trained on TPUs.
* FLIP models are trained on [LAION datasets](https://laion.ai/) including LAION-400M and LAION-2B.



### Results and Pre-trained FLIP models

The following table provides zero-shot results on ImageNet-1K and links to pre-trained weights for the LAION datasets:
<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom"></th>
<th valign="bottom">data</th>
<th valign="bottom">sampled</th>
<th valign="bottom">zero-shot IN-1K</th>
<th valign="bottom">model</th>
<!-- TABLE BODY -->
<tr><td align="left">ViT-B</td>
<td align="center">LAION-400M</td>
<td align="center">12.8B</td>
<td align="center">68.0</td>
<td align="center">download</td>
</tr>
<tr><td align="left">ViT-L</td>
<td align="center">LAION-400M</td>
<td align="center">12.8B</td>
<td align="center">74.3</td>
<td align="center">download</td>
</tr>
<tr><td align="left">ViT-H/14</td>
<td align="center">LAION-400M</td>
<td align="center">12.8B</td>
<td align="center">75.5</td>
<td align="center">download</td>
</tr>
<tr><td align="left">ViT-L</td>
<td align="center">LAION-2B</td>
<td align="center">25.6B</td>
<td align="center">76.6</td>
<td align="center">download</td>
<tr><td align="left">ViT-H/14</td>
<td align="center">LAION-2B</td>
<td align="center">25.6B</td>
<td align="center">78.8</td>
<td align="center">download</td>
</tbody></table>

### Installation and data preperation

Please check [INSTALL.md](INSTALL.md) for installation instructions and data prepraration.

### Training

Our FLIP models are trained on Google Cloud TPU To set up Google Cloud TPU, please refer to the their docs for
[single VM setup](https://cloud.google.com/tpu/docs/jax-quickstart-tpu-vm)
and [pod slice setup](https://cloud.google.com/tpu/docs/jax-quickstart-tpu-vm).

By default, we train ViT-B/L models using v3-256 TPUs and ViT-H models with v3-512 TPUs. 

#### 1. Pretraining FLIP models via masking
##### Running locally

```
export TFDS_DATA_DIR=gs://$GCS_TFDS_BUCKET/datasets
python3 main.py \
    --workdir=${workdir} \
    --config=$1 \
    --config.batch_size=256 \
```

##### Running on cloud

```
gcloud alpha compute tpus tpu-vm ssh $VM_NAME --zone $ZONE \
--worker=all --command "
export TFDS_DATA_DIR=gs://$GCS_TFDS_BUCKET/datasets &&
python3 main.py --workdir=$WORKDIR --config=configs/cfg_flip_large.py
```

#### 2. Unmasked tuning 

For unmasked tuning, we use the same configs except the following parameters: 
```
python3 main.py --workdir=$WORKDIR --config=configs/cfg_flip_large.py \
--config.model.model_img.mask_ratio=0.0 --config.learning_rate=4e-8
--config.num_epochs=100 --config.warmup_epochs=20 \
```
To avoid out of memory issue, we may need to optionally turn on `config.partitioning.partition_states` and activation checkpointing by `config.model.model_img.transformer.remat_policy=actcp`, and reduce batch size `config.batch_size`.


### Evaluation

To evaluation the pre-trained models for zero-shot ImageNet-1K:

```
export TFDS_DATA_DIR=gs://$GCS_TFDS_BUCKET/datasets
python3 main.py \
    --workdir=${workdir} \
    --config=configs/cfg_flip_large.py \
    --config.pretrain_dir=$PRETRAIN_MODEL_PATH \
    --config.eval_only=True \
```


### Acknowledgement
* This repo is built based on [flax](https://github.com/google/flax/tree/main/examples/imagenet) and [t5x](https://github.com/google-research/t5x).

### License

This project is under the CC-BY-NC 4.0 license. See [LICENSE](LICENSE) for details.