from typing import Tuple, Any

from absl import logging

import tree as nest

import jax
import flax
from flax.training import checkpoints
import jax.numpy as jnp

import jax.tree_util as tu

import torch
import os



def convert_from_pytorch(state, pretrain_dir):
  from IPython import embed; embed();
  if (0 == 0): raise NotImplementedError

  names = filter_parameters(state.params, get_name)

  params_list, params_tree = tu.tree_flatten(state.params)
  names_list, names_tree = tu.tree_flatten(names)
  named_params = dict((x, y) for x, y in zip(names_list, params_list))

  a = tu.tree_unflatten(flattened[1], flattened[0])

  os.system('gsutil cp {} ./tmp/.'.format(pretrain_dir))

  os.path.join('./tmp', os.path.basename(pretrain_dir))

  checkpoint = torch.load(os.path.join('./tmp', os.path.basename(pretrain_dir)), map_location='cpu')
  checkpoint = checkpoint['model']

  keys_jax = state.params.keys()

  for name_pt in checkpoint:
      name_jx = convert_name(name_pt)
      print('{:32s}: {:60s}, {}'.format(name_pt, name_jx, (name_jx in named_params)))


def convert_name(name):
    # convert:
    if name == 'cls_token':
        name_jx = 'cls'
    elif name == 'pos_embed':
        name_jx = 'posembed_encoder.pos_embedding'
    elif name == 'patch_embed.proj.weight':
        name_jx = 'embedding.kernel'
    elif name == 'patch_embed.proj.bias':
        name_jx = 'embedding.bias'
    elif name.startswith('norm.weight'):
        name_jx = 'Transformer.encoder_norm.bias'
    elif name.startswith('norm.bias'):
        name_jx = 'Transformer.encoder_norm.scale'
    elif name == 'head.weight':
        name_jx = 'head.kernel'
    elif name == 'head.bias':
        name_jx = 'head.bias'
    elif name.startswith('blocks.'):
        name_suffix = name[len('blocks.'):]  # remove 'block.'
        blk_idx = name_suffix[:name_suffix.find('.')]  # get 11 from '11.mlp.fc2.weight'
        blk_idx = int(blk_idx)
        
        name_jx = 'Transformer.encoderblock_{:02d}.'.format(blk_idx)

        # MSA
        if 'norm1.weight' in name:
            name_jx += 'LayerNorm_0.scale'
        elif 'norm1.bias' in name:
            name_jx += 'LayerNorm_0.bias'
        elif 'attn.proj.weight' in name:
            name_jx += 'MultiHeadDotProductAttention_0.out.kernel'
        elif 'attn.proj.bias' in name:
            name_jx += 'MultiHeadDotProductAttention_0.out.bias'
        elif 'attn.qkv.weight' in name:
            name_jx += 'MultiHeadDotProductAttention_0.qkv.kernel'
        elif 'attn.q_bias' in name:
            name_jx += 'MultiHeadDotProductAttention_0.q.bias'
        elif 'attn.v_bias' in name:
            name_jx += 'MultiHeadDotProductAttention_0.v.bias'
        # MLP
        elif 'norm2.weight' in name:
            name_jx += 'LayerNorm_1.scale'
        elif 'norm2.bias' in name:
            name_jx += 'LayerNorm_1.bias'
        elif 'mlp.fc1.weight' in name:
            name_jx += 'MlpBlock_0.Dense_0.kernel'
        elif 'mlp.fc1.bias' in name:
            name_jx += 'MlpBlock_0.Dense_0.bias'
        elif 'mlp.fc2.weight' in name:
            name_jx += 'MlpBlock_0.Dense_1.kernel'
        elif 'mlp.fc2.bias' in name:
            name_jx += 'MlpBlock_0.Dense_1.bias'
    else:
        return None
        # raise NotImplementedError
    return name_jx

for name_pt in checkpoint:
    name_jx = convert_name(name_pt)
    print('{:32s}: {:100s}, {}'.format(name_pt, name_jx, (name_jx in named_params)))



def get_name(path: Tuple[Any], val: jnp.ndarray):
    del val
    layer_name = '.'.join(path)
    return layer_name

# ---------------------------------------------------------
# the entrance function:
# ---------------------------------------------------------
def filter_parameters(params, filter_fn):
    """Filter the params based on filter_fn."""
    params_to_filter = nest.map_structure_with_path(filter_fn, params)
    return params_to_filter

Transformer.encoderblock_04.MultiHeadDotProductAttention_0.LayerNorm_0.scale