"""
Generative Pretrained Tranformer
"""

import torch
from torch import nn
from .transformer import LayerNorm
from .transformer import TransformerBlock

# TODO this should be pydantic dataclass

GPT_CONFIG_124M = {
    "vocab_size": 50257,  # Vocabulary size
    "context_length": 1024,  # Context length
    "emb_dim": 768,  # Embedding dimension
    "n_heads": 12,  # Number of attention heads
    "n_layers": 12,  # Number of layers
    "drop_rate": 0.1,  # Dropout rate
    "qkv_bias": True,  # Query-Key-Value bias: set to true as pre-trained use it
}

GPT_CONFIG_355M = {
    "vocab_size": 50257,  # Vocabulary size
    "context_length": 1024,  # Context length
    "emb_dim": 1024,  # Embedding dimension
    "n_heads": 16,  # Number of attention heads
    "n_layers": 24,  # Number of layers
    "drop_rate": 0.0,  # Dropout rate
    "qkv_bias": True,  # Query-Key-Value bias: set to true as pre-trained use it
}


class GPTModel(nn.Module):
    """
    Being inherited of nn.Module this class act as a neural network.
    In torch.nn.Module there is a __call__ implementation that call forward method
    (which is defined here).
    No custom pre-forward hook or post-forward hook is implemented here.

    Initialization is done with a config dict.

    - Input: batch of tokenized sentence.
    - Output: a forward pass produce logits for each positions in each batch
    Logits are a dictionary-wide scores vector. With softmax function applied on it it
    become a probability distribution of the token.
    For input tokens [x1, x2, x3] it predict x2 from x1, x3 for x1 and x2, and finally
    x4 from all.
    The prediction is applied of all token in the input for training purpose.
    At inference only the next token matter.
    """

    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )

        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx):
        _, seq_len = in_idx.shape  # batch_size, seq_len
        tok_embeds = self.tok_emb(in_idx)
        pos_ids = torch.arange(0, seq_len, device=in_idx.device, dtype=torch.long)
        pos_embeds = self.pos_emb(pos_ids).unsqueeze(0)
        x = tok_embeds + pos_embeds  # Shape [batch_size, num_tokens, emb_size]
        x = self.drop_emb(x)
        # x is the hidden state between transformers block. Output of transformer 1
        # is the input of transformer 2 and so on.
        for blk in self.trf_blocks:
            x = blk(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits
