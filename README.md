# Leon's LLM

Based on *Build a Large Language Model from scratch* book by Sebastian Raschka.

## Wishlist

As of now, top priority is RoPe, with it I could create a efficient KV cache. Then, SWA and GQA.

- 1: A good positional encoding scheme : Rotary Postional Encoding (RoPE). This would be used
to implement KV cache optimization at inference time and would stay compatible with
incremental decoding. Designed for, musch cleaner and standard approach.
RoPE is better (than absolute positioning) because cached attention can keep old keys and
values without running into the positional inconsistency that learned absolute embeddings
create when the context grows or slides.
- 2: Sliding Window Attention (SWA): build a *local* attention window by focusing on the N token before
the query. Then we can use only the local window or a mix between local and global (the usual one).
- 3: Grouped-Query Attention (GQA): a technics that reduce KV memory size by using shared KV
between attention heads.


The following 2 components are usefull for DeepSeek architecture but low priority:
- ?: Mixture of Experts (MoE): replace each feed-forward module in a transformer block with multiple
expert layers. This increase models parameters size but the tricks is that we don't use all *expert*
for every tokens. This is a memory optimization technics that lead to model with increased learning
capacity while keeping the runtime memory low.
- 4: Multi-Head Latent Attention (MLA): compresses the key and value tensors into a lower-dimensional
space before storing them in the KV cache.

Note that the goal of all of theses technics are to optimize KV cache size.

- A uniform way to download various model from HuggingFace, shape the model accordingly and load the
weight. Use **safetensors** for saving/loading weight?

- Use PyTorch SDPA in Attention class.
- Improve output metrics. Use sklearn/evaluate to add metrics.
- Add an optional cache to detect specialized neural regions activated for tasks.
- bitsandbytes for memory pressure ?
- !!! Use HuggingFace `datasets` to download and manipulate datasets. !!!!

Tokenizer encoding scheme is part of the pre-trained base model contract.
Some special character can be part of post-training: instructions, reasoning
