
import torch

_original_load = torch.load


def _patched_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)   
    return _original_load(*args, **kwargs)


torch.load = _patched_load
