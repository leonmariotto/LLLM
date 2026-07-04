# Leon's LLM

A from scratch implementation of LLM inference using **pytorch**. Currently 
supporting **GPT2**, **Gemma3**, **Llama3**, **Qwen3** models inferences. Implement
**sliding KV caching** and **quantization** optimizations. Implement **structured 
output** at generation level for a very small subset of JSON-schema.

To easily integrate new model provide an evaluation system using dataset like
**BoolQ**, **GSM8K** and **Squad** for instructions-tuned models and **Wikitext-2** for
base models. Each integrated models is covered by unit-tests and functional tests.

Implement embedding models inference and tooling to produce **embedding vector database**. Provide a tool (`vector_db`) to build, save, load and use theses database.
Provide options to use this DB with **retrieval augmented generation**.

Provide user application : `coder` code-production oriented generation (use reasoning technics suck as **self-refinment** and **self-consistency**), `chat` simple
chat application with fancy TUI.

Internally use different loader (**hf_loader** and **gguf**) that translate into an internal
representation (**model_ir** or **model_ir**) which is then used by models. This keep a clear separation between models
format and internal logic.

Provide a `server` application that act as an inference server provising OpenAI-compatible HTTP endpoints.

Full coverage test suite including **unit-tests** and **functional tests**.

Based on :
- *Build a Large Language Model from scratch* book by Sebastian Raschka.
- *Build a reasoning model* book by Sebastian Raschka.

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

## Inference server

Run the OpenAI-compatible chat completions server:

```sh
uv run server --served-model-name lllm
```

The server detects the model architecture from its artifacts. Qwen3 and Gemma3
models are currently supported, for example:

```sh
uv run server --model google/gemma-3-1b-it --served-model-name gemma3
```

Then send a non-streaming request:

```sh
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "lllm",
    "messages": [{"role": "user", "content": "Say hello"}],
    "max_tokens": 64,
    "enable_thinking": false
  }'
```

Thinking is disabled in this short example so the token budget is used for the
visible answer. When thinking is enabled, use a larger `max_tokens` value.

Function tools use the OpenAI chat-completions loop: the server returns requested
tool calls, while the client executes them and sends their results back as `tool`
messages. The server does not execute tools.

## vector_db

vector_db can be used to build an embedding vector database, which then can be used in chat application 
for RAG (Retrieval-Augmented Generation).
```
# Build the VectorDB database.
uv run vector_db --yaml tests/vectordb/rfc/rfc.yml --out rfc.vectordb
# Call chat app in non-terminal mode with debug enabled to see the augmented prompt.
echo "what's ARP ?" | uv run chat --rag-vector-db-path rfc.vectordb  --rag-max-entries 3 --rag-score-cutoff 0.4 --verbosity debug
```

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
This is mainly a demonstration of self-refinment and self-consistency technics for 
reasoning models.


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

- add suport for tool_choice="required"
- add a strong hard constrained generation for tool calls.
- add typed output in generator (now that we have typed generation), not only str.

### Embedded models inference

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
