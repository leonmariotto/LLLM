from typing import Any, Protocol, cast

import torch


class TensorModel(Protocol):
    def eval(self) -> Any: ...

    def __call__(self, idx: torch.Tensor) -> torch.Tensor: ...


class Tokenizer(Protocol):
    def encode(self, input: str) -> list[int]: ...

    def decode(self, tok: list[int]) -> str: ...


class Generator:
    def __init__(
        self,
        model: TensorModel,
        tokenizer: Tokenizer,
        context_size: int = 1024,
    ) -> None:
        """
        Create a text generator for an autoregressive token model.

        Args:
            model: Callable model that accepts a tensor of token ids shaped
                ``[batch, tokens]`` and returns logits shaped
                ``[batch, tokens, vocab_size]``.
            tokenizer: Tokenizer used to encode input text into token ids and
                decode generated token ids back into text.
            context_size: Default number of latest tokens to feed back into the
                model at each generation step. Older tokens are cropped from the
                model input but remain in the returned text.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.context_size = context_size

    def generate(
        self,
        prompt: str,
        *,
        stop_at_eos: bool = True,
        max_generated_token: int = 20,
        eos: int | None = None,
        context_size: int | None = None,
        temperature: float = 0.0,
        top_k: int | None = None,
        include_prompt: bool = True,
    ) -> str:
        """
        Generate text from a prompt.

        Args:
            prompt: Input text used as the initial generation context.
            stop_at_eos: When ``True``, stop generation before appending ``eos``
                if the next predicted token is the EOS token.
            max_generated_token: Maximum number of new tokens to generate.
            eos: Token id treated as end-of-sequence when ``stop_at_eos`` is
                enabled. If ``None``, no EOS stopping is applied.
            context_size: Optional per-call context window override. When not
                provided, the generator default from ``__init__`` is used.
            temperature: Sampling temperature. ``0.0`` uses deterministic greedy
                argmax decoding; values above zero sample from the scaled
                probability distribution.
            top_k: If set, restrict each next-token choice to the ``top_k``
                highest-logit tokens before decoding.
            include_prompt: When ``True``, return prompt plus generated text.
                When ``False``, return only the generated completion.

        Returns:
            Decoded text from the selected output tokens. The returned string is
            full prompt plus completion when ``include_prompt`` is true, otherwise
            completion text only.
        """
        prompt_tokens = self.tokenizer.encode(prompt)
        generated_tokens = self._generate_tokens(
            prompt_tokens,
            stop_at_eos=stop_at_eos,
            max_generated_token=max_generated_token,
            eos=eos,
            context_size=self.context_size if context_size is None else context_size,
            temperature=temperature,
            top_k=top_k,
        )

        output_tokens = (
            generated_tokens
            if include_prompt
            else generated_tokens[len(prompt_tokens) :]
        )
        return self.tokenizer.decode(output_tokens)

    def _generate_tokens(
        self,
        input_tokens: list[int],
        *,
        stop_at_eos: bool,
        max_generated_token: int,
        eos: int | None,
        context_size: int,
        temperature: float,
        top_k: int | None,
    ) -> list[int]:
        self.model.eval()
        idx = torch.tensor(
            [input_tokens],
            dtype=torch.long,
            device=self._model_device(),
        )

        for _ in range(max_generated_token):
            idx_cond = idx[:, -context_size:]
            with torch.no_grad():
                logits = self.model(idx_cond)

            logits = logits[:, -1, :]
            logits = self._filter_logits(logits, top_k)
            idx_next = self._select_next_token(logits, temperature)
            if stop_at_eos and eos is not None and bool((idx_next == eos).all().item()):
                break
            idx = torch.cat((idx, idx_next), dim=1)

        return cast(list[int], cast(Any, idx.squeeze(0)).tolist())

    def _filter_logits(self, logits: torch.Tensor, top_k: int | None) -> torch.Tensor:
        if top_k is None:
            return logits

        top_logits, _ = torch.topk(logits, top_k)
        min_val = top_logits[:, -1]
        return torch.where(
            logits < min_val,
            torch.tensor(float("-inf"), device=logits.device),
            logits,
        )

    def _select_next_token(
        self, logits: torch.Tensor, temperature: float
    ) -> torch.Tensor:
        if temperature > 0.0:
            logits = logits / temperature
            logits = logits - logits.max(dim=-1, keepdim=True).values
            probs = torch.softmax(logits, dim=-1)
            return torch.multinomial(probs, num_samples=1)

        return torch.argmax(logits, dim=-1, keepdim=True)

    def _model_device(self) -> torch.device:
        if not isinstance(self.model, torch.nn.Module):
            return torch.device("cpu")

        try:
            return next(self.model.parameters()).device
        except StopIteration:
            return torch.device("cpu")
