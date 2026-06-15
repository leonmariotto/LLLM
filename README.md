# Leon's LLM

A from scratch implementation of LLM inference using **pytorch**. Currently 
supporting **GPT2**, **Gemma3**, **Llama3**, **Qwen3** models inferences. Implement
**sliding KV caching** and **quantization** optimizations.

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

Full coverage test suite including **unit-tests** and **functional tests**.

Based on :
- *Build a Large Language Model from scratch* book by Sebastian Raschka.
- *Build a reasoning model* book by Sebastian Raschka.
- *Build an AI agent from scratch* book by Younghee Song & Jungjun Hur.

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

- add support for optional field in constrained decoder
- add suport for tool_choice="required"
- add a strong hard constrined generation for tool calls.

## Development note.

### Context management

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


Different tools response/requests must have differents compaction.
Each tool can declare a ToolContextPolicy class.

History summarization should use layers:
    - recent_message: exact last N turn
    - session summary: compressed older conversation
    - task_state: current goals, decision and constraint
    - facts_memory: durable facts discovered.
    - open_thread: unresolved questions.

Distinguish memory:
    - Instruction memory: stable behavior rules
    - Task memory, current goal, plan, constraint, decision, TODOs.
    - Conversation memory: user/assistant turns, summarizable.
    - Evidence memory: RAG snippet, search result, citation.
    - Tool memory: calls, output
Each category have different compaction rules.

Context invalidation is important, when files changes, or something happen. This
can be done using some context items metadata.

Deduplication.

So there is a general context management part that handle which messages are pinned, 
which can be summarized, etc
And there is a per-tool context management that handle tool call tool response in context.
Some LLM call can be done to extract information of tool result in order to summarize them.
Don't do callback, I don't like this.

task_state should be made from
    - deterministic parsing of all events.
    - LLM produced task_state_patch, validated deterministically.

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
