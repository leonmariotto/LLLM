# Leon's LLM

A from scratch implementation of LLM inference using **pytorch**. Currently 
supporting **GPT2**, **Gemma3**, **Llama3**, **Qwen3** models inferences. Implement
**sliding KV caching** and **quantization** optimizations.

To easily integrate new model there is an evaluation system using dataset like
**BoolQ**, **GSM8K** and **Squad** for instructions-tuned models and **Wikitext-2** for
base models. Each integrated models is covered by unit-tests and functional tests.

Implement LLM reasoning technics such as specialized **self-refinment** and
**self-consistency** for specific tasks.

Provide user application : **coder** code-production oriented generation, **chat** simple
chat application, **planner** plan-production oriented generation.

Provide different loader (**hf_loader**, **gguf** and **native**) that translate into an internal
representation which is then used by models. This keep a clear separation between models
format and internal logic.

Full coverage test suite including **unit-tests** and **functional tests**.

Based on :
- *Build a Large Language Model from scratch* book by Sebastian Raschka.
- *Build a reasoning model* book from the same author.

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

## Inventory

- Applications: exported via uv tool, run `uv run <app> --help` to launch it.
    - LLLM/chat.py: Interactive chat application with terminal UI, **conversation history**, context/memory reporting, and one-shot stdin mode.
    - LLLM/coder.py: C code generation application: generates **multiple candidates**, compiles them, **refines warnings**, and uses an **LLM judge** to select the best result.
    - LLLM/planner.py: Interactive planning application: **expands a request**, synthesizes a reviewable summary, accepts **user refinement**, then produces an approved task plan.
- Model Implementations: Tokenizer encoding scheme is part of the pre-trained base model contract. Some special character can be part of post-training: instructions, reasoning.
Models accept only intermediate representation, keeping external models format outside this code.
    - LLLM/gpt2.py: GPT-2 model, tokenizer, transformer blocks, attention layers, and its generation adapter.
    - LLLM/llama2.py: Llama 2 implementation using multi-head attention, RoPE, RMSNorm, tokenizer support, and KV caching.
    - LLLM/llama3.py: Llama 3/3.1/3.2 implementation with grouped-query attention, extended RoPE support, quantization, and tokenizer loading.
    - LLLM/qwen2.py: Qwen2/Qwen2.5 decoder implementation with grouped-query attention, tokenizer support, KV caching, and quantized weights.
    - LLLM/qwen3.py: Dense Qwen3 decoder implementation, adding Q/K normalization to grouped-query attention; used by chat and planner.
    - LLLM/gemma3.py: Gemma 3 implementation with grouped-query attention, sliding-window attention, specialized normalization, quantization, and tokenizer support.
- Inference Building Blocks:
    - LLLM/generator.py: Shared autoregressive text-generation engine with **KV-cache handling**, greedy or sampled decoding, and **throughput metrics**.
    - LLLM/kv_cache.py: Reusable key/value cache for incremental decoding, including **bounded sliding-window retention**.
    - LLLM/rope.py: Rotary positional encoding utilities supporting **interleaved and split-half layouts** plus **scaled frequencies**.
    - LLLM/norm.py: Shared LayerNorm and RMSNorm implementations used by transformer architectures.
    - LLLM/quantization.py: Quantized-weight representation and linear-layer execution support, primarily for **GGUF-loaded** models.
- Model Loading And Representation:
    - LLLM/model_ir.py: Canonical **intermediate representation** that separates model implementations from Hugging Face and GGUF source formats.
    - LLLM/fetch.py: Entry point for locating or downloading model artifacts and loading them into the internal representation.
    - LLLM/hf_loader.py: Converts Hugging Face configs and **safetensor weights** into the project’s canonical model representation.
    - LLLM/gguf.py: Reads **GGUF files**, maps metadata and tensors into the internal representation, and extracts tokenizer information.
- Evaluation And Utilities:
    - LLLM/eval.py: Evaluation framework for **base-model perplexity** and instruction-task benchmarks such as **BoolQ, GSM8K, and SQuAD**.
    - LLLM/utils.py: Earlier/general training and evaluation helpers for token conversion, device selection, datasets, dataloaders, and accuracy.
    - LLLM/yaml_parser.py: Small strict YAML configuration parser with project-specific error handling.

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
- Add a training app that exports native IR checkpoints and persists optimizer/scheduler state for training resumption.
- Add Hugging Face and GGUF exporters for distributing fine-tuned native IR checkpoints outside this project.

- create EmbeddingDB: function to produce, load and search into embedding database. + an app to use it.

- litellm completion have a response_format variable that define the expected
structured output. This must be a prompt formatting technics, see with current
Tokenizer what we can do. Using pydantic it's handy because we can generate a JSon schema from a pydantic type easily.
Note that it's not litellm that implement constrained decoding it's the "provider": the
inference service/server.
Here's some technics to do that :
    - Prompt-only formating: no hard garantee.
    - Some model are trained for schema, so we can output them a JSON schema directly.
    - Constrained decoding, we tweaks the generation so that invalid token are masked
    out in token probability. That's the hard-lock. It's fun, I should do that. Not so
    simple to implement though.

## Context management

- an agent need enough **effective context management**
- a reasonably large cache_length
- truncation of noisy old messages
- summaries of old history
- retrieval from files/vector DB/search
- selective tool output inclusion
- compact state like “current goal”, “known facts”, “open issues”
- keeping the current task and constraints near the end of the prompt
- "thinking" block not preserved.
- tool executor can have state and memory.

- system prompt must:
    - define boundaries, which request is accepted, which is rejected.
    - output format and style
    - clarify knowledge/capacity limits
    - "When the user's intent is clear, execute immediately without confirmation.
        Only when intent is unclear, ask minimal questions to clarify"
    - "use tool proactively, without asking permission"
    - Clearly define when to use tools: use trigger pattern.
    - Define when not use tool : "Do not search for timeless information,
        fundamental concepts, definitions, or well- established technical facts."
    - Provide concrete examples.

## Embedded models inference

A causal LLM answer :
```
Given this prefix, what would be the next tokens ?
```
An embedding model answer:
```
Given this whole text, what vector represent it's meaning ?
```
Causal LLM are **unidirectional** whereas embedding models are **bi-drectional**, 
both previous and nexts tokens are used, no causal mask is applied. 
An embedding model produce hidden_state, and then pools them into one vector :
```
Without pool: in -> model -> embedding [batch, seq_len, hidden_size]
With pool: in -> model -> embedding [batch, embedding_size]
```
**Pooling step is central**, can be `mean` or `cls`, depending on the model.
Optionaly a projection head can be used at this point.
Embedding normalization is often required.

**Hidden states** are the transformers blocks output. Embedding models don't 
include final normalization, but use directly the last hidden state returned by
the transformer forward pass. This hidden states is then pooled, and normalized.
