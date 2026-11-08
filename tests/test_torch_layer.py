"""Validate the example layer's explicit discrete adjoint and network use."""

import pytest

torch = pytest.importorskip('torch')

from examples.pytorch.devito_layer import DevitoDiffusion, train_demo  # noqa: E402


def reference(value, steps, coefficient):
    channels = value.shape[1]
    kernel = value.new_tensor([[0, coefficient, 0],
                               [coefficient, 1-4*coefficient, coefficient],
                               [0, coefficient, 0]])
    weights = kernel.expand(channels, 1, 3, 3)
    for _ in range(steps):
        value = torch.nn.functional.conv2d(value, weights, padding=1, groups=channels)
    return value


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_forward_and_backward_match_pytorch(dtype):
    torch.manual_seed(3)
    value = torch.randn(2, 2, 5, 6, dtype=dtype, requires_grad=True)
    expected_input = value.detach().clone().requires_grad_()
    layer = DevitoDiffusion(steps=3, coefficient=0.1)
    output = layer(value)
    expected = reference(expected_input, 3, 0.1)
    cotangent = torch.randn_like(output)
    actual_gradient, = torch.autograd.grad(output, value, cotangent)
    expected_gradient, = torch.autograd.grad(expected, expected_input, cotangent)
    torch.testing.assert_close(output, expected)
    torch.testing.assert_close(actual_gradient, expected_gradient)


def test_finite_difference_gradcheck():
    torch.manual_seed(4)
    layer = DevitoDiffusion(steps=2, coefficient=0.12)
    value = torch.randn(1, 1, 3, 4, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(layer, (value,), eps=1e-6, atol=1e-5, rtol=1e-4)


def test_multiple_forwards_and_noncontiguous_input():
    layer = DevitoDiffusion(steps=3, coefficient=0.1)
    left = torch.randn(2, 2, 6, 5, dtype=torch.float64).transpose(-1, -2)
    left.requires_grad_()
    right = torch.randn_like(left, requires_grad=True)
    left_output, right_output = layer(left), layer(right)
    saved = left_output.detach().clone()
    layer(torch.zeros_like(left))
    torch.testing.assert_close(left_output, saved)
    left_output.sum().backward()
    right_output.square().sum().backward()
    expected_left = left.detach().clone().requires_grad_()
    expected_right = right.detach().clone().requires_grad_()
    reference(expected_left, 3, 0.1).sum().backward()
    reference(expected_right, 3, 0.1).square().sum().backward()
    torch.testing.assert_close(left.grad, expected_left.grad)
    torch.testing.assert_close(right.grad, expected_right.grad)


def test_network_trains_through_physics():
    network, losses = train_demo()
    assert losses[-1] < 0.5*losses[0]
    assert network[0].weight.grad.abs().sum() > 0
    assert network[3].weight.grad.abs().sum() > 0


@pytest.mark.parametrize('steps,coefficient', [(0, .1), (2, -.1), (2, .3)])
def test_invalid_configuration(steps, coefficient):
    with pytest.raises(ValueError):
        DevitoDiffusion(steps, coefficient)


def test_invalid_input():
    layer = DevitoDiffusion()
    with pytest.raises(ValueError):
        layer(torch.zeros(8, 8))
    with pytest.raises(TypeError):
        layer(torch.zeros((1, 1, 8, 8), dtype=torch.int64))
