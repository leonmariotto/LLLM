# Leon's LLM

Based on *Build a Large Language Model from scratch* book by Sebastian Raschka.
Then on *Build a reasoning model* book from the same author.

## Coder

Coder is a tool used to generate C code. 
It use (by default) Qwen2.5 0.6B for generation and Qwen3 0.6B for judging. 
Implement code-spcific self-consistency:
For a given task, X candidates are generated, compilation is checked, then among 
the successfull (compiled) candidates we choose the best by doing a judge tournament. 
After self-consistency, a loop of self-refinment is launch to remove compiler warning 
if any (note that it barely manage to remove simple warning with Qwen-2.5 0.5B ...).

Run:
```
$> uv run coder --help
$> echo "task" | uv run coder
```

Even with very small models, it manage to produce compilable and functionaly correct 
C code for simple tasks.

## Chat

Chat can be used to ask a question to a model.
If called without piping, start a fancy terminal prompt.
The chat history is re-injected at every new request.

Run:
```
$> uv run chat --help
$> echo "question" | uv run chat
$> uv run chat
```

<img width="1920" height="1080" alt="Screenshot from 2026-05-24 16-23-11" src="https://github.com/user-attachments/assets/9df05958-152d-4df4-999e-9a78c1b218cb" />

## Planner

Planner contain a specialized self-refinment loop with human feedback.
First the user request is expanded in X version. Then the X versions are agregated
into a bullet-point list.
The user is then asked to `revise`, `cancel` or `accept`.
If `revise` is used, the user can write a review, and Planner will regenerate the
bullet-point list with this input, and the user will be re-prompted.
If `accept` is used the bullet point list is used to generate a markdown plan.
If `cancel` is used the whole process is canceled.

Run:
```
$> uv run planner --help
$> uv run planner "design a caching layer"
$> uv run planner
```

WIP: it's not working very well... I should add more discipline by doing a standardized process
(SPICE). I could try to transform SYS.3 requirements in SWE.1 requirements, and then SWE.1 into
SWE.2 architecture decision. It could be adapted for other SPICE-defined process.


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

- Do a pass of documentation on docstring models because some has been dropped by refactor.

## Inventory

- RoPE.
- KV cache with sliding window.
- Evaluation setup : WikiText-2, BoolQ, gsm8k and squad.
- Quantization support.
- GGUF loading.
- HF's hub loading.

Tokenizer encoding scheme is part of the pre-trained base model contract.
Some special character can be part of post-training: instructions, reasoning
