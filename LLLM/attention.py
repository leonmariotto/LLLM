"""
Implement a simple self attention.
Attention mechanism involve 3 trainable matrix : query, key, values.
"""

import torch
from torch import nn

from .rope import apply_rope, precompute_rope_cache


class MultiHeadAttention(nn.Module):
    """
    Being inherited of nn.Module this class act as a neural network.
    In torch.nn.Module there is a __call__ implementation that call forward method
    (which is defined here).
    No custom pre-forward hook or post-forward hook is implemented here.

    Implement Causal mask and dropout.

    Multi-headed attention: d_out is splited in num_head parts. Each head
    produce a part of d_out (head_dim, calculated at init), and at the end
    context_vec is reshaped to the correct size.
    So things can be parallel.

    """

    # Need to tell pyright that the "mask" registered by register_buffer method
    # is an tensor, to avoid typing errors.
    mask: torch.Tensor
    rope_cos: torch.Tensor
    rope_sin: torch.Tensor

    def __init__(
        self,
        d_in: int,
        d_out: int,
        context_length: int,
        dropout: float,
        num_heads: int,
        qkv_bias: bool = False,
        use_rope: bool = False,
    ) -> None:
        """
        - d_in: embedding size (size of embedded vector, 1 embedded vector per token)
        - d_out context vector size.
        - context_lenght: correspond to the number token used to compute a context
        vector. In the case of a DataSet/DataLoader setup, it will correspond
        to the window_size.
        - droput: for training purpose, it is possible to hide randomly some attention
        weight before computing the context vector. dropout value is the probability
        for a weight to be zeroed.
        - num_head: number of head.
        """
        super().__init__()

        assert num_heads != 0, "num_head shall not be 0"
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.d_in = d_in
        self.num_heads = num_heads
        self.head_dim = (
            d_out // num_heads
        )  # Reduce the projection dim to match desired output dim
        self.context_length = context_length
        self.use_rope = use_rope

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)  # Linear layer to combine head outputs
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask", torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )
        if self.use_rope:
            cos, sin = precompute_rope_cache(
                seq_len=self.context_length,
                head_dim=self.head_dim,
            )
            self.register_buffer("rope_cos", cos, persistent=False)
            self.register_buffer("rope_sin", sin, persistent=False)

    def forward(self, x: torch.Tensor, pos: int | None = None) -> torch.Tensor:
        """
        forward method is called by the nn.Module __call__ method.
        x is expected to be a batch of tensor of d_in size.
        (number of batch, number of token, embedding size)
        return a tensor of d_out size.

        pos: contains the (context-wide relative) starting index of the sequence.
        """
        b, num_tokens, d_in = x.shape

        assert self.d_in == d_in, "invalid d_in (embedding size)"

        # As in `CausalAttention`, for inputs where `num_tokens` exceeds
        # `context_length`, this will result in errors in the mask creation further
        # below.
        # In practice, this is not a problem since the LLM (chapters 4-7) ensures that
        # inputs do not exceed `context_length` before reaching this forward method.

        keys_new = self.W_key(x)  # Shape: (b, num_tokens, d_out)
        values_new = self.W_value(x)
        queries = self.W_query(x)

        # NOTE on tensor.view and tensor.transpose methods.
        # Tensor view method reshape a tensor, without moving elements in memory.
        # Whereas transpose change how dimensions are indexed.
        # So, for example :
        #     tensor([[0, 1, 2],
        #         [3, 4, 5]])
        # y.view(3,2)
        #     tensor([[0, 1],
        #         [2, 3],
        #         [4, 5]])
        # y.transpose(0, 1)
        #     tensor([[0, 3],
        #             [1, 4],
        #             [2, 5]])

        # We implicitly split the matrix by adding a `num_heads` dimension
        # Unroll last dim: (b, num_tokens, d_out) -> (b, num_tokens, num_heads, head_dim)
        keys_new = keys_new.view(b, num_tokens, self.num_heads, self.head_dim)
        values_new = values_new.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        keys, values = keys_new, values_new

        # Transpose: (b, num_tokens, num_heads, head_dim) -> (b, num_heads, num_tokens, head_dim)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        if self.use_rope:
            # Add RoPE after Q/K projection and head reshaping and before computing
            # attention scores.
            if pos is None:
                position_ids = torch.arange(num_tokens, device=x.device)
            else:
                position_ids = torch.arange(
                    pos,
                    pos + num_tokens,
                    device=x.device,
                )
            assert int(position_ids[-1]) < self.context_length, (
                "RoPE position exceeds precomputed context length"
            )
            cos = self.rope_cos[position_ids]
            sin = self.rope_sin[position_ids]
            queries = apply_rope(queries, cos, sin)
            keys = apply_rope(keys, cos, sin)

        # Compute scaled dot-product attention (aka self-attention) with a causal mask
        attn_scores = queries @ keys.transpose(2, 3)  # Dot product for each head

        # `queries` has shape (batch, num_heads, num_tokens, head_dim).
        # So shape[-2] is the query-token dimension.
        num_tokens_Q = queries.shape[-2]
        num_tokens_K = keys.shape[-2]
        # Original mask truncated to the number of tokens and converted to boolean.
        mask_bool = self.mask.bool()[:num_tokens_Q, :num_tokens_K]

        # Use the mask to fill attention scores
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Shape: (b, num_tokens, num_heads, head_dim)
        context_vec = (attn_weights @ values).transpose(1, 2)

        # Combine heads, where self.d_out = self.num_heads * self.head_dim
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)  # optional projection

        return context_vec
