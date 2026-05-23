# Leon's LLM

Based on *Build a Large Language Model from scratch* book by Sebastian Raschka.
Then on *Build a reasoning model* book from the same author.

## Wishlist

- Mixture of Experts (MoE): replace each feed-forward module in a transformer block with multiple
expert layers. This increase models parameters size but the tricks is that we don't use all *expert*
for every tokens. This is a memory optimization technics that lead to model with increased learning
capacity while keeping the runtime memory low.
- Multi-Head Latent Attention (MLA): compresses the key and value tensors into a lower-dimensional
space before storing them in the KV cache.
- tiny-aya models
- optimized quantization computation.

- Use PyTorch SDPA in Attention class.
- Improve output metrics. Use sklearn/evaluate to add metrics.
- Add an optional cache to detect specialized neural regions activated for tasks.
- bitsandbytes for memory pressure

- Top-P filtering.

- Do a pass of documentation on docstring models because some has been dropped by refactor.

### Role-local self-consistency:

Part of inference-time reasoning technics: the model output N answer, then it choose one, or create one from the Ns.
Self-consistency criterium depends highly on the task.
Self-consistency for a coding task is quite different than self consistency for a summary/plan task.
It should be part of the role class: Coder, Planer, ...
- Coder: we may need to restrict what the agent can do. If it can produce only self-contained, compilable code
it greatly simplify the verification/evaluation design. Or we could work in a controlled environment, with a
known way to compile and execute. That sound better but is a lot more complex because the model need to be able to
read and modify existing files, that introduce tools in the loop.

## Inventory

- RoPE.
- KV cache with sliding window.
- Evaluation setup : WikiText-2, BoolQ, gsm8k and squad.
- Quantization support.
- GGUF loading.
- HF's hub loading.

Tokenizer encoding scheme is part of the pre-trained base model contract.
Some special character can be part of post-training: instructions, reasoning
