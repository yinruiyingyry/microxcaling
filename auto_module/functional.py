import torch
from torch import nn
import torch.nn.functional as F
from auto_module.utils import get_op_id, _context, disable_replacement
import functools

def get_parameter(param):
  if isinstance(param, nn.Parameter):
    return param
  elif isinstance(param, torch.Tensor):
    return nn.Parameter(param, requires_grad=False)
  else:
    raise Exception("param has to be instance of nn.Parameter or torch.Tensor!")

F_linear = F.linear

@functools.wraps(F_linear)
def wrapped_F_linear(inpt, weight, bias=None, *args, **kwargs):
  if not _context.is_replacement_enabled():
    return F_linear(inpt, weight, bias, *args, **kwargs)
  with disable_replacement():
    op_name = "wrapped_F_linear"
    module, op_id = get_op_id(op_name=op_name)
    if isinstance(module, nn.Linear):
      return F_linear(inpt, weight, bias, *args, **kwargs)
    if not hasattr(module, op_id):
      linear = nn.Linear(weight.shape[1], weight.shape[0], bias=(bias is not None), device=inpt.device, dtype=inpt.dtype)
      linear.weight = get_parameter(weight)
      linear.bias = None if bias is None else get_parameter(bias)
      setattr(module, op_id, linear)

    return getattr(module, op_id)(inpt)

F.linear = wrapped_F_linear

F_relu = F.relu

@functools.wraps(F_relu)
def wrapped_F_relu(inpt: torch.Tensor, inplace: bool = False):
  if not _context.is_replacement_enabled():
    return F_relu(inpt, inplace=inplace)
  with disable_replacement():
    op_name = "wrapped_F_relu"
    module, op_id = get_op_id(op_name=op_name)
    if isinstance(module, nn.ReLU):
      return F_relu(inpt, inplace=inplace)
    if not hasattr(module, op_id):
      setattr(module, op_id, nn.ReLU(inplace=inplace))

    return getattr(module, op_id)(inpt)

F.relu = wrapped_F_relu

F_max_pool2d = F.max_pool2d

@functools.wraps(F_max_pool2d)
def wrapped_F_max_pool2d(inpt, kernel_size, stride = None,
                 padding = 0, dilation = 1,
                 return_indices = False, ceil_mode = False):
  if not _context.is_replacement_enabled():
    return F_max_pool2d(
      inpt, kernel_size=kernel_size, stride=stride, padding=padding,
      dilation=dilation, return_indices=return_indices, ceil_mode=ceil_mode)
  with disable_replacement():
    op_name = "wrapped_F_max_pool2d"
    module, op_id = get_op_id(op_name=op_name)
    if isinstance(module, nn.MaxPool2d):
      return F_max_pool2d(
        inpt, kernel_size=kernel_size, stride=stride, padding=padding,
        dilation=dilation, return_indices=return_indices, ceil_mode=ceil_mode)
    if not hasattr(module, op_id):
      setattr(module, op_id, nn.MaxPool2d(kernel_size=kernel_size, stride=stride,
                  padding=padding, dilation=dilation,
                  return_indices=return_indices, ceil_mode=ceil_mode))

    return getattr(module, op_id)(inpt)

F.max_pool2d = wrapped_F_max_pool2d

F_conv1d = F.conv1d

@functools.wraps(F_conv1d)
def wrapped_F_conv1d(inpt, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
  if not _context.is_replacement_enabled():
    return F_conv1d(inpt, weight, bias, stride=stride, padding=padding, dilation=dilation, groups=groups)
  with disable_replacement():
    op_name = "wrapped_F_conv1d"
    module, op_id = get_op_id(op_name=op_name)
    if isinstance(module, nn.Conv1d):
      return F_conv1d(inpt, weight, bias, stride=stride, padding=padding, dilation=dilation, groups=groups)
    if not hasattr(module, op_id):
      conv1d = nn.Conv1d(
        weight.shape[1], weight.shape[0], (weight.shape[2], weight.shape[3]), bias=(bias is not None),
        stride=stride, padding=padding, dilation=dilation, groups=groups,
        device=weight.device, dtype=weight.dtype)
      conv1d.weight = get_parameter(weight)
      conv1d.bias = None if bias is None else get_parameter(bias)
      setattr(module, op_id, conv1d)

    return getattr(module, op_id)(inpt)

F.conv1d = wrapped_F_conv1d

F_conv2d = F.conv2d

@functools.wraps(F_conv2d)
def wrapped_F_conv2d(inpt, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
  if not _context.is_replacement_enabled():
    return F_conv2d(inpt, weight, bias, stride=stride, padding=padding, dilation=dilation, groups=groups)
  with disable_replacement():
    op_name = "wrapped_F_conv2d"
    module, op_id = get_op_id(op_name=op_name)
    if isinstance(module, nn.Conv2d):
      return F_conv2d(inpt, weight, bias, stride=stride, padding=padding, dilation=dilation, groups=groups)
    if not hasattr(module, op_id):
      conv2d = nn.Conv2d(
        weight.shape[1], weight.shape[0], (weight.shape[2], weight.shape[3]), bias=(bias is not None),
        stride=stride, padding=padding, dilation=dilation, groups=groups,
        device=weight.device, dtype=weight.dtype)
      conv2d.weight = get_parameter(weight)
      conv2d.bias = None if bias is None else get_parameter(bias)
      setattr(module, op_id, conv2d)

    return getattr(module, op_id)(inpt)

F.conv2d = wrapped_F_conv2d

F_conv3d = F.conv3d

@functools.wraps(F_conv3d)
def wrapped_F_conv3d(inpt, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
  if not _context.is_replacement_enabled():
    return F_conv3d(inpt, weight, bias, stride=stride, padding=padding, dilation=dilation, groups=groups)
  with disable_replacement():
    op_name = "wrapped_F_conv3d"
    module, op_id = get_op_id(op_name=op_name)
    if isinstance(module, nn.Conv3d):
      return F_conv3d(inpt, weight, bias, stride=stride, padding=padding, dilation=dilation, groups=groups)
    if not hasattr(module, op_id):
      conv3d = nn.Conv3d(
        weight.shape[1], weight.shape[0], (weight.shape[2], weight.shape[3]), bias=(bias is not None),
        stride=stride, padding=padding, dilation=dilation, groups=groups,
        device=weight.device, dtype=weight.dtype)
      conv3d.weight = get_parameter(weight)
      conv3d.bias = None if bias is None else get_parameter(bias)
      setattr(module, op_id, conv3d)

    return getattr(module, op_id)(inpt)

F.conv3d = wrapped_F_conv3d
