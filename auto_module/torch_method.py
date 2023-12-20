import torch
from torch import nn
from auto_module.utils import wrap_stateless_function
from auto_module.modules import *

#wrap_stateless_function(torch, "where", Where)
wrap_stateless_function(torch, "add", TorchAdd)
#wrap_stateless_function(torch, "mul", TorchMul)
#wrap_stateless_function(torch, "matmul", TorchMatMul)
#wrap_stateless_function(torch, "sqrt", TorchSqrt)
