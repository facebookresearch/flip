## Scaling Language-Image Pre-training via Masking

<p align="center">
  <img src="https://user-images.githubusercontent.com/5648235/210933226-b157339a-7aa8-47e9-864a-75cc65e52c99.png" width="480">
</p>

This repository contains the official JAX implementation of **FLIP**, as described in the paper [Scaling Language-Image Pre-training via Masking](https://arxiv.org/abs/2212.00794).

```
@inproceedings{li2022scaling,
  title={Scaling Language-Image Pre-training via Masking},
  author={Li, Yanghao and Fan, Haoqi and Hu, Ronghang and Feichtenhofer, Christoph and He, Kaiming},
  booktitle={CVPR},
  year={2023}
}
```

* The implementation is based on JAX and the models are trained on TPUs.
* FLIP models are trained on [LAION datasets](https://laion.ai/) including LAION-400M and LAION-2B.
* Other links
  * For PyTorch and GPU implementation, OpenCLIP has incorporated FLIP into their [repo](https://github.com/mlfoundations/open_clip#patch-dropout) and trained a ViT-G/14 FLIP model with 80.1% ImageNet zero-shot accuray (79.4% before model soups). See their [blog](https://laion.ai/blog/giant-openclip/) for more information.



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
<tr><td align="left">ViT-B/16</td>
<td align="center">LAION-400M</td>
<td align="center">12.8B</td>
<td align="center">68.0</td>
<td align="center">-</td>
</tr>
<tr><td align="left">ViT-L/16</td>
<td align="center">LAION-400M</td>
<td align="center">12.8B</td>
<td align="center">74.3</td>
<td align="center">-</td>
</tr>
<tr><td align="left">ViT-H/14</td>
<td align="center">LAION-400M</td>
<td align="center">12.8B</td>
<td align="center">75.5</td>
<td align="center">-</td>
</tr>
<tr><td align="left">ViT-L/16</td>
<td align="center">LAION-2B</td>
<td align="center">25.6B</td>
<td align="center">76.6</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/flip/flip_large16_laion2b_v2.zip">download†</td>
</tr>
<tr>
<td align="left">ViT-H/14</td>
<td align="center">LAION-2B</td>
<td align="center">25.6B</td>
<td align="center">78.8</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/flip/flip_huge14_laion2b_v2.zip">download†</td>
</tbody></table>

† The released ViT-L/16 and ViT-H/14 models were trained on LAION datasets where faces were blurred as a legal requirement, resulting in a slight performance drop by 0.2-0.3% to achieve accuracies of 76.4% and 78.5%, respectively.

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
    --config.laion_path=LAION_PATH \
```

##### Running on cloud

```
gcloud alpha compute tpus tpu-vm ssh $VM_NAME --zone $ZONE \
--worker=all --command "
export TFDS_DATA_DIR=gs://$GCS_TFDS_BUCKET/datasets &&
python3 main.py --workdir=$WORKDIR --config=configs/cfg_flip_large.py --config.laion_path=LAION_PATH 
```

#### 2. Unmasked tuning 

For unmasked tuning, we use the same configs except the following parameters: 
```
python3 main.py --workdir=$WORKDIR --config=configs/cfg_flip_large.py \
--config.laion_path=LAION_PATH \
--config.model.model_img.mask_ratio=0.0 --config.learning_rate=4e-8
--config.num_epochs=100 --config.warmup_epochs=20 \
 --config.pretrain_dir=${PRETRAIN} \
```
To avoid out of memory issue, we may need to optionally turn on activation checkpointing by `config.model.model_img.transformer.remat_policy=actcp` and reduce batch size `config.batch_size`.


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