# loss.py
import torch
import torch.nn.functional as F

# def sincos_loss(pred, target):
#     return F.mse_loss(torch.sin(pred), torch.sin(target)) + F.mse_loss(torch.cos(pred), torch.cos(target))

def sincos_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_sin = torch.sin(pred)
    pred_cos = torch.cos(pred)
    true_sin = torch.sin(target)
    true_cos = torch.cos(target)
    return torch.mean((pred_sin - true_sin)**2 + (pred_cos - true_cos)**2)
