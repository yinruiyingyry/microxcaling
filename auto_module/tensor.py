import torch
from torch import nn
from torch import Tensor
from typing import Any
from auto_module.utils import wrap_stateless_function
from auto_module.modules import *

wrap_stateless_function(Tensor, "__add__", Add)
wrap_stateless_function(Tensor, "__radd__", Add)
wrap_stateless_function(Tensor, "__iadd__", Add)
wrap_stateless_function(Tensor, "__mul__", Mul)
wrap_stateless_function(Tensor, "__rmul__", Mul)
wrap_stateless_function(Tensor, "__imul__", Mul)
wrap_stateless_function(Tensor, "__truediv__", Div)
wrap_stateless_function(Tensor, "__itruediv__", Div)
wrap_stateless_function(Tensor, "__rtruediv__", Rdiv)
wrap_stateless_function(Tensor, "__sub__", Sub)
wrap_stateless_function(Tensor, "__isub__", Sub)
wrap_stateless_function(Tensor, "__rsub__", Rsub)
wrap_stateless_function(Tensor, "__lt__", LessThan)
wrap_stateless_function(Tensor, "to", To)
wrap_stateless_function(Tensor, "type", Type)
