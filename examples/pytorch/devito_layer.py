"""A differentiable fixed diffusion layer using an explicit discrete adjoint.

Run from the repository root with:
    OMP_NUM_THREADS=2 python -m examples.pytorch.devito_layer

This CPU, eager-mode example implements first-order input gradients for a
constant-coefficient, uniform-grid, zero-ghost diffusion stencil. It does not
make arbitrary Devito Operators differentiable.
"""

from functools import lru_cache
from itertools import count
from math import isfinite
from threading import Lock

import torch
from torch import nn
from torch.autograd.function import once_differentiable

from devito import (
    Dimension, Eq, Grid, Operator, TimeFunction, TorchAllocator, configuration
)
from devito.tools import dtype_from_torch

_names = count()


class _DiffusionPlan:

    def __init__(self, planes, height, width, dtype, steps, coefficient):
        self.grid = Grid(shape=(height, width), extent=(height-1., width-1.))
        self.plane = Dimension(name='plane')
        self.shape = (2, planes, height, width)
        self.dtype = dtype_from_torch(dtype).type
        self.steps = steps
        self.lock = Lock()
        self.symbol = self._state()
        self.operator = Operator(
            Eq(self.symbol.forward, self.symbol + coefficient*self.symbol.laplace),
            language='torch',
            opt=('advanced', {'openmp': True})
        )

    def _state(self):
        return TimeFunction(
            name=f'diffusion_{next(_names)}', grid=self.grid,
            dimensions=(self.grid.stepping_dim, self.plane, *self.grid.dimensions),
            shape=self.shape, time_order=1, space_order=2, dtype=self.dtype,
            allocator=TorchAllocator()
        )

    def run(self, value):
        # Cache the compiled operator, never mutable activations. A fresh
        # workspace makes two forwards before either backward safe.
        with self.lock:
            state = self._state()
            state.data[0].copy_(value.detach().reshape(self.shape[1:]))
            self.operator.apply(**{self.symbol.name: state}, time_M=self.steps-1,
                                autotune='off')
            output = torch.empty(value.shape, dtype=value.dtype, device=value.device)
            output.copy_(state.data[self.steps].reshape(value.shape))
            return output


@lru_cache(maxsize=16)
def _plan(planes, height, width, dtype, steps, coefficient):
    return _DiffusionPlan(planes, height, width, dtype, steps, coefficient)


class _DiffusionFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, value, plan):
        ctx.plan = plan
        return plan.run(value)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        # A = I + coefficient*L is symmetric for this exact five-point
        # discretization and zero-ghost boundary treatment. Thus (A^steps)^T
        # is A^steps too. This is an explicit adjoint, not autodiff of C++.
        return ctx.plan.run(grad_output), None


class DevitoDiffusion(nn.Module):
    """Fixed diffusion on CPU float32/float64 tensors shaped [N, C, H, W].

    `coefficient` is alpha*dt on a unit-spaced 2D grid and lies in [0, 1/4].
    Batch and channel planes evolve independently. First-order gradients pass
    through to the input; steps/coefficient are fixed Python configuration.
    CUDA, distributed execution and higher-order gradients are not supported
    by this example layer.
    """

    def __init__(self, steps=3, coefficient=0.1):
        super().__init__()
        if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
            raise ValueError('steps must be a positive integer')
        if isinstance(coefficient, torch.Tensor):
            raise TypeError('coefficient is fixed configuration, not a tensor parameter')
        if not isfinite(coefficient) or not 0 <= coefficient <= 0.25:
            raise ValueError('coefficient must lie in [0, 0.25]')
        self.steps = steps
        self.coefficient = float(coefficient)

    def forward(self, value):
        if configuration['mpi']:
            raise NotImplementedError('This example layer does not support MPI')
        if value.device.type != 'cpu':
            raise ValueError('This example layer supports CPU tensors only')
        if value.dtype not in (torch.float32, torch.float64):
            raise TypeError('Expected float32 or float64 tensor storage')
        if value.ndim != 4 or min(value.shape) < 1 or min(value.shape[-2:]) < 2:
            raise ValueError('Expected [N, C, H, W] with H and W at least 2')
        batch, channels, height, width = value.shape
        plan = _plan(batch*channels, height, width, value.dtype,
                     self.steps, self.coefficient)
        return _DiffusionFunction.apply(value, plan)

    def extra_repr(self):
        return f'steps={self.steps}, coefficient={self.coefficient:g}'


def train_demo(iterations=40):
    """Train learned convolutions on both sides of a Devito physics layer."""
    torch.manual_seed(7)
    inputs = torch.randn(8, 1, 16, 16)
    target = nn.functional.avg_pool2d(inputs, kernel_size=3, stride=1, padding=1)
    network = nn.Sequential(
        nn.Conv2d(1, 4, kernel_size=3, padding=1),
        nn.Tanh(),
        DevitoDiffusion(steps=2, coefficient=0.1),
        nn.Conv2d(4, 1, kernel_size=1),
    )
    optimizer = torch.optim.Adam(network.parameters(), lr=0.03)
    losses = []
    for _ in range(iterations):
        optimizer.zero_grad(set_to_none=True)
        loss = nn.functional.mse_loss(network(inputs), target)
        loss.backward()
        if not torch.isfinite(network[0].weight.grad).all():
            raise RuntimeError('Nonfinite upstream gradient')
        losses.append(loss.item())
        optimizer.step()
    with torch.no_grad():
        losses.append(nn.functional.mse_loss(network(inputs), target).item())
    return network, losses


if __name__ == '__main__':
    model, history = train_demo()
    print(model)
    print(f'Training MSE: {history[0]:.6f} -> {history[-1]:.6f}')
