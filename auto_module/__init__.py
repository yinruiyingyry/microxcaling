from torch import nn

from auto_module.utils import OP_COUNTS, enable_replacement

class WrappedModel(nn.Module):
  def __init__(self, model: nn.Module) -> None:
    super().__init__()
    self._inner_model = model

  def forward(self, *args, **kwargs):
    with enable_replacement():
      return self._inner_model(*args, **kwargs)


def wrap_module(model: nn.Module) -> nn.Module:
  def clear_op_counts(module, inputs):
    setattr(module, OP_COUNTS, {})
    return inputs

  model.register_forward_pre_hook(clear_op_counts)
  for child in model.children():
    wrap_module(child)
  return model

def wrap(model: nn.Module, *args, **kwargs) -> nn.Module:
  """
  Args:
    model (nn.Module): The model to be processed by auto_module
    args, kwargs: The inputs of model. There is model inference in this function.
      `args` and `kwargs` will be fed to model directly during model inference.
  """
  import auto_module.functional
  import auto_module.tensor
  import auto_module.torch_method

  training = model.training

  model.eval()
  model = WrappedModel(model)
  model = wrap_module(model)
  model(*args, **kwargs)

  model.train(training)
  return model
