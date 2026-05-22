# Leon's LLM

Based on *Build a Large Language Model from scratch* book by Sebastian Raschka.
Then on *Build a reasoning model* book from the same author.

## Wishlist

- cache dataset download !! It will speedup a lot the tests.
- tiny-aya models
- optimized quantization computation.

The following 2 components are usefull for DeepSeek architecture but low priority:
- ?: Mixture of Experts (MoE): replace each feed-forward module in a transformer block with multiple
expert layers. This increase models parameters size but the tricks is that we don't use all *expert*
for every tokens. This is a memory optimization technics that lead to model with increased learning
capacity while keeping the runtime memory low.
- 4: Multi-Head Latent Attention (MLA): compresses the key and value tensors into a lower-dimensional
space before storing them in the KV cache.

- Use PyTorch SDPA in Attention class.
- Improve output metrics. Use sklearn/evaluate to add metrics.
- Add an optional cache to detect specialized neural regions activated for tasks.
- bitsandbytes for memory pressure ?

Tokenizer encoding scheme is part of the pre-trained base model contract.
Some special character can be part of post-training: instructions, reasoning

## Inventory

- RoPE.
- KV cache with sliding window.
- Evaluation setup : WikiText-2, BoolQ, gsm8k and squad.
- Quantization support.
- GGUF loading.
- HF's hub loading.

