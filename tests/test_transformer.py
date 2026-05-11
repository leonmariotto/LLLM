import pytest

torch = pytest.importorskip("torch")

from torch import nn

from ..LLLM.transformer import (
    FeedForward,
    GELU,
    LayerNorm,
    TransformerBlock,
)


class AddConstant(nn.Module):
    def __init__(self, value: float) -> None:
        """Store the constant added to every tensor element."""
        super().__init__()
        self.value = value

    def forward(self, x: torch.Tensor, pos: int | None = None) -> torch.Tensor:
        """Return the input tensor with the configured constant added."""
        return x + self.value


class ScaleBy(nn.Module):
    def __init__(self, factor: float) -> None:
        """Store the multiplicative factor applied in the forward pass."""
        super().__init__()
        self.factor = factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the input tensor scaled by the configured factor."""
        return x * self.factor


def test_gelu_matches_torch_tanh_approximation() -> None:
    """Verify GELU matches PyTorch's tanh-approximate GELU implementation."""
    activation = GELU()
    inputs = torch.tensor([-2.0, -0.5, 0.0, 0.5, 2.0])

    outputs = activation(inputs)
    expected_outputs = torch.nn.functional.gelu(inputs, approximate="tanh")

    torch.testing.assert_close(outputs, expected_outputs)


def test_feed_forward_matches_manual_two_layer_computation() -> None:
    """Check FeedForward matches an explicit two-linear-layer GELU computation."""
    feed_forward = FeedForward(embedded_dimension=2, expansion_factor=2)

    # Pull out the internal linear layers so the expected result can be computed step by step.
    first_linear = feed_forward.layers[0]
    second_linear = feed_forward.layers[2]

    with torch.no_grad():
        first_linear.weight.copy_(
            torch.tensor(
                [
                    [1.0, 0.0],
                    [0.0, 1.0],
                    [1.0, 1.0],
                    [-1.0, 1.0],
                ]
            )
        )
        first_linear.bias.copy_(torch.tensor([0.1, -0.2, 0.3, 0.0]))
        second_linear.weight.copy_(
            torch.tensor(
                [
                    [1.0, 0.0, 0.5, -0.5],
                    [0.0, 1.0, -0.5, 0.5],
                ]
            )
        )
        second_linear.bias.copy_(torch.tensor([0.2, -0.1]))

    inputs = torch.tensor([[[0.5, -1.0], [1.5, 0.25]]])

    outputs = feed_forward(inputs)

    # Rebuild the forward pass explicitly: linear -> GELU -> linear.
    hidden = inputs @ first_linear.weight.T + first_linear.bias
    activated = torch.nn.functional.gelu(hidden, approximate="tanh")
    expected_outputs = activated @ second_linear.weight.T + second_linear.bias

    torch.testing.assert_close(outputs, expected_outputs)


def test_layer_norm_matches_manual_normalization_with_scale_and_shift() -> None:
    """Confirm LayerNorm applies normalization, scaling, and shifting correctly."""
    layer_norm = LayerNorm(emb_dim=3, eps=1e-5)
    inputs = torch.tensor([[[1.0, 3.0, 5.0], [2.0, 4.0, 8.0]]])

    with torch.no_grad():
        layer_norm.scale.copy_(torch.tensor([1.5, -2.0, 0.5]))
        layer_norm.shift.copy_(torch.tensor([0.2, 0.3, -0.4]))

    outputs = layer_norm(inputs)

    # Match the implementation details, including population variance.
    mean = inputs.mean(dim=-1, keepdim=True)
    var = inputs.var(dim=-1, keepdim=True, unbiased=False)
    normalized = (inputs - mean) / torch.sqrt(var + layer_norm.eps)
    expected_outputs = normalized * layer_norm.scale + layer_norm.shift

    torch.testing.assert_close(outputs, expected_outputs)


def test_transformer_block_applies_both_residual_paths() -> None:
    """Ensure TransformerBlock applies attention and feed-forward residual paths."""
    cfg = {
        "emb_dim": 2,
        "context_length": 3,
        "n_heads": 1,
        "drop_rate": 0.0,
        "qkv_bias": False,
        "positional_encoding": "gpt2",
    }
    block = TransformerBlock(cfg)
    # Replace submodules with deterministic operations so the residual math is easy to verify.
    block.norm1 = nn.Identity()
    block.norm2 = nn.Identity()
    block.att = AddConstant(1.0)
    block.ff = ScaleBy(2.0)
    block.drop_shortcut = nn.Identity()

    inputs = torch.tensor(
        [
            [
                [1.0, 2.0],
                [3.0, 4.0],
                [5.0, 6.0],
            ]
        ]
    )

    outputs = block(inputs)
    # First residual path: x + (x + 1) = 2x + 1. Second path: (2x + 1) + 2(2x + 1) = 6x + 3.
    expected_outputs = 6 * inputs + 3

    torch.testing.assert_close(outputs, expected_outputs)
