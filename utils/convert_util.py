from typing import Tuple, Any

from absl import logging
from termcolor import colored

import tree as nest

import jax
import flax
from flax.training import checkpoints
import jax.numpy as jnp

import jax.tree_util as tu

import torch
import os


def convert_from_pytorch(state, pretrain_dir):
  state_params = flax.core.frozen_dict.unfreeze(state.params)
  state_params.pop('head')  # remove head
  state_params = flax.core.frozen_dict.freeze(state_params)

  names = filter_parameters(state_params, get_name)

  params_list, params_tree = tu.tree_flatten(state_params)
  names_list, names_tree = tu.tree_flatten(names)
  named_params = dict((x, y) for x, y in zip(names_list, params_list))

  os.system('gsutil cp {} ./tmp/.'.format(pretrain_dir))
  os.path.join('./tmp', os.path.basename(pretrain_dir))

  checkpoint = torch.load(os.path.join('./tmp', os.path.basename(pretrain_dir)), map_location='cpu')
  checkpoint = checkpoint['model']
  checkpoint = revise_qkv(checkpoint)

  converted_named_params = convert_names_and_shapes(checkpoint, named_params)

  converted_params_list = []
  for name in names_list:
    converted_params_list.append(jnp.asarray(converted_named_params[name]))

  converted_params = tu.tree_unflatten(params_tree, converted_params_list)
  converted_params = flax.core.frozen_dict.freeze(converted_params)
  
  # sanity
  verify = tu.tree_multimap(lambda x, y: (x.shape == y.shape),
      converted_params, state_params)
  verify = tu.tree_leaves(verify)
  assert jnp.all(jnp.array(verify)).item()

  return state.replace(params=converted_params)
  

def convert_names_and_shapes(checkpoint, named_params):
  converted_named_params = {}
  for name_pt in checkpoint:
    p_pt = checkpoint[name_pt].clone()
    shape_pt = tuple(p_pt.shape)

    name_jx = convert_name(name_pt)
    if name_jx in named_params:
      shape_jx = tuple(named_params[name_jx].shape)
      if len(shape_pt) == 1 and len(shape_jx) == 1:  # 1-d tensors
          assert shape_pt == shape_jx
      elif len(shape_pt) == 4 and len(shape_jx) == 4:  # patch_embed
          p_pt = torch.einsum('nchw->hwcn', p_pt)
          assert tuple(p_pt.shape) == shape_jx
      elif len(shape_pt) == 3 and len(shape_jx) == 3:  # pos_embed
          assert shape_pt == shape_jx
      elif len(shape_pt) == 2 and len(shape_jx) == 2:  # mlp
          p_pt = torch.einsum('nc->cn', p_pt)
          assert tuple(p_pt.shape) == shape_jx
      elif len(shape_pt) == 2 and len(shape_jx) == 3 and '.out.kernel' in name_jx:  # msa, out.kernel
          p_pt = torch.einsum('nc->cn', p_pt)  # n: out_filters; c: in_filters
          assert shape_jx[-1] == p_pt.shape[-1]
          p_pt = p_pt.view(shape_jx)
      elif len(shape_pt) == 2 and len(shape_jx) == 3 and '.out.kernel' not in name_jx:  # msa, q, k, v
          p_pt = torch.einsum('nc->cn', p_pt)  # n: out_filters; c: in_filters
          assert shape_jx[0] == p_pt.shape[0]
          p_pt = p_pt.view(shape_jx)
      elif len(shape_pt) == 1 and len(shape_jx) == 2:  # q, k, v bias
          p_pt = p_pt.view(shape_jx)

      # now, p_pt is the expected tensor
      assert tuple(p_pt.shape) == shape_jx
      print('{:32s}:{:20s} => {:80s}:{:20s}'.format(name_pt, str(shape_pt), name_jx, str(shape_jx)))
      converted_named_params[name_jx] = p_pt
    else:
      print(colored('Not converted: {} => {}'.format(name_pt, name_jx), 'red'))

  return converted_named_params


def revise_qkv(checkpoint):
  # handle q, k, v
  checkpoint_revised = {}
  for name_pt in checkpoint:
    p = checkpoint[name_pt].clone()
    if 'attn.qkv.weight' in name_pt:
      q, k, v = torch.split(p, p.shape[0] // 3, dim=0)
      checkpoint_revised[name_pt.replace('.qkv.', '.q.')] = q
      checkpoint_revised[name_pt.replace('.qkv.', '.k.')] = k
      checkpoint_revised[name_pt.replace('.qkv.', '.v.')] = v
    else:
      checkpoint_revised[name_pt] = p

    if 'attn.q_bias' in name_pt:  # add k_bias
      checkpoint_revised[name_pt.replace('.q_bias', '.k_bias')] = torch.zeros_like(p)

  del checkpoint
  return checkpoint_revised


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
    elif 'attn.q.weight' in name:
      name_jx += 'MultiHeadDotProductAttention_0.query.kernel'
    elif 'attn.k.weight' in name:
      name_jx += 'MultiHeadDotProductAttention_0.key.kernel'
    elif 'attn.v.weight' in name:
      name_jx += 'MultiHeadDotProductAttention_0.value.kernel'
    elif 'attn.q_bias' in name:
      name_jx += 'MultiHeadDotProductAttention_0.query.bias'
    elif 'attn.k_bias' in name:
      name_jx += 'MultiHeadDotProductAttention_0.key.bias'
    elif 'attn.v_bias' in name:
      name_jx += 'MultiHeadDotProductAttention_0.value.bias'
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


# ---------------------------------------------------------
# the tree functions:
# ---------------------------------------------------------
def get_name(path: Tuple[Any], val: jnp.ndarray):
  del val
  layer_name = '.'.join(path)
  return layer_name


def filter_parameters(params, filter_fn):
  """Filter the params based on filter_fn."""
  params_to_filter = nest.map_structure_with_path(filter_fn, params)
  return params_to_filter