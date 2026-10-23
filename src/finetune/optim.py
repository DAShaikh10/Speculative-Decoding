"""
This module contains Adafactor for FSDP-sharded parameters, which `transformers.Adafactor` can't update.

FSDP2 shards every parameter along dimension 0, and the parameters and gradients are DTensors. The update of HF Adafactor
mixes them with plain tensors and fails. This one works on the local shards, and all-reduces the statistics that span the
whole tensor: the parameter and update RMS, the column statistics of factored parameters and the mean of the row
statistics. It is the T5 Adafactor of the recipe in PLAN.md: factored second moment, no momentum, no weight decay,
update clipping, and a learning rate scaled by the parameter RMS. On unsharded tensors it equals `transformers.Adafactor`.
"""

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor


def _reduced(value: torch.Tensor, sharded: bool) -> torch.Tensor:
    if sharded:
        dist.all_reduce(value)
    return value


class ShardedAdafactor(torch.optim.Optimizer):
    """
    Adafactor with beta1 = None, as `transformers.Adafactor(scale_parameter=True, relative_step=False)`.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        clip_threshold: float = 1.0,
        decay_rate: float = -0.8,
        eps: tuple[float, float] = (1e-30, 1e-3),
    ) -> None:
        super().__init__(params, {"lr": lr, "clip_threshold": clip_threshold, "decay_rate": decay_rate, "eps": eps})

    @torch.no_grad()
    def step(self, closure=None) -> None:
        for group in self.param_groups:
            for param in group["params"]:
                if param.grad is None:
                    continue
                sharded: bool = isinstance(param, DTensor)
                weight: torch.Tensor = param.to_local() if sharded else param
                grad: torch.Tensor = (param.grad.to_local() if sharded else param.grad).float()
                rows: int = param.shape[0]  # The global number of rows. Local shards hold some of them.

                state: dict = self.state[param]
                if not state:
                    state["step"] = 0
                    if grad.ndim >= 2:
                        state["row"] = grad.new_zeros(grad.shape[:-1])
                        state["col"] = grad.new_zeros(grad.shape[:-2] + grad.shape[-1:])
                    else:
                        state["square"] = torch.zeros_like(grad)
                state["step"] += 1

                eps_square, eps_scale = group["eps"]
                beta: float = 1.0 - state["step"] ** group["decay_rate"]
                rms: torch.Tensor = (_reduced(weight.float().square().sum(), sharded) / param.numel()).sqrt()
                lr: torch.Tensor = group["lr"] * rms.clamp_min(eps_scale)

                square: torch.Tensor = grad.square() + eps_square
                if grad.ndim >= 2:
                    state["row"].mul_(beta).add_(square.mean(dim=-1), alpha=1.0 - beta)
                    column: torch.Tensor = _reduced(square.sum(dim=-2), sharded) / rows
                    state["col"].mul_(beta).add_(column, alpha=1.0 - beta)
                    row_mean: torch.Tensor = _reduced(state["row"].sum(dim=-1, keepdim=True), sharded) / rows
                    r_factor: torch.Tensor = (state["row"] / row_mean).rsqrt().unsqueeze(-1)
                    c_factor: torch.Tensor = state["col"].unsqueeze(-2).rsqrt()
                    update: torch.Tensor = r_factor * c_factor * grad
                else:
                    state["square"].mul_(beta).add_(square, alpha=1.0 - beta)
                    update = state["square"].rsqrt() * grad

                update_rms: torch.Tensor = (_reduced(update.square().sum(), sharded) / param.numel()).sqrt()
                update = update / (update_rms / group["clip_threshold"]).clamp_min(1.0)
                weight.add_((-lr * update).to(weight.dtype))
