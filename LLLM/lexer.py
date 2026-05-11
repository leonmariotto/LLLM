"""
Use tiktoken library to transform input text into tokens.
"""

import logging
from importlib.metadata import version
from typing import Literal, TypeAlias, TypedDict

import tiktoken
from tiktoken.core import Encoding

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)

TokenId: TypeAlias = int


class LexerDescription(TypedDict):
    encoding: str
    tiktoken_version: str
    vocabulary_size: int
    end_of_text_token: TokenId
    special_tokens: list[str]


EncodingName = Literal["gpt2", "r50k_base", "p50k_base", "cl100k_base", "o200k_base"]


class Lexer:
    """
    Default algorithm is BPE (byte paire encoding, gpt2).
    """

    def __init__(
        self,
        encoding: EncodingName = "gpt2",
    ) -> None:
        self.encoding_name = encoding
        self.tiktok: Encoding = tiktoken.get_encoding(encoding)
        self.logger = logging.getLogger(__name__)
        self.logger.debug(
            "Initialize TikToken tokenizer version %s , encoding %s",
            version("tiktoken"),
            encoding,
        )

    @property
    def vocabulary_size(self) -> int:
        # Surface encoder capacity directly from tiktoken for quick inspection.
        return self.tiktok.n_vocab

    @property
    def end_of_text_token(self) -> TokenId:
        # Keep the canonical special token id accessible to callers and tests.
        return self.tiktok.eot_token

    @property
    def special_tokens(self) -> set[str]:
        return set(self.tiktok.special_tokens_set)

    def describe(self) -> LexerDescription:
        # Centralize encoder metadata so debug output stays consistent.
        return {
            "encoding": self.encoding_name,
            "tiktoken_version": version("tiktoken"),
            "vocabulary_size": self.vocabulary_size,
            "end_of_text_token": self.end_of_text_token,
            "special_tokens": sorted(self.special_tokens),
        }

    def encode(self, in_str: str) -> list[TokenId]:
        self.logger.debug("encode %d char", len(in_str))
        return self.tiktok.encode(in_str, allowed_special={"<|endoftext|>"})
        # TODO should return tensor here ??
        # encoded = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
        # encoded_tensor = torch.tensor(encoded).unsqueeze(0)  # add batch dimension
        # return encoded_tensor

    def decode(self, in_tok: list[TokenId]) -> str:
        self.logger.debug("decode %d token", len(in_tok))
        return self.tiktok.decode(in_tok)

    def token_count(self, in_str: str) -> int:
        return len(self.encode(in_str))
