"""
Encoding notes used in these tests:

- `gpt2`: classic GPT-2 BPE; widely documented and stable, but less efficient on
  multilingual and modern chat-oriented text.
- `r50k_base`: close to `gpt2`; useful for compatibility checks, but offers few
  practical gains over it.
- `p50k_base`: tuned for Codex/code use cases; often good on source code, but not
  necessarily better than newer vocabularies on natural language.
- `cl100k_base`: newer general-purpose encoding; typically produces fewer tokens
  on contemporary text, with broad model compatibility.
- `o200k_base`: larger and more recent vocabulary; can compress text further, at
  the cost of targeting a narrower set of newer models.
"""

from typing import List
from ..LLLM.lexer import Lexer, EncodingName


LA_CIGALE_ET_LA_FOURMI = """La Cigale et la Fourmi

La Cigale, ayant chanté
Tout l'été,
Se trouva fort dépourvue
Quand la bise fut venue :
Pas un seul petit morceau
De mouche ou de vermisseau.
Elle alla crier famine
Chez la Fourmi sa voisine,
La priant de lui prêter
Quelque grain pour subsister
Jusqu'à la saison nouvelle.
"""


def test_gpt2_round_trip_on_poem():
    # The baseline contract: encoding followed by decoding must preserve the text.
    lexer = Lexer()

    tokens = lexer.encode(LA_CIGALE_ET_LA_FOURMI)

    assert lexer.decode(tokens) == LA_CIGALE_ET_LA_FOURMI
    assert lexer.token_count(LA_CIGALE_ET_LA_FOURMI) == len(tokens)


def test_describe_exposes_tiktoken_metadata():
    # Metadata should expose the key encoder facts needed for debugging and reporting.
    lexer = Lexer("gpt2")

    description = lexer.describe()

    assert description["encoding"] == "gpt2"
    assert description["vocabulary_size"] == lexer.vocabulary_size
    assert description["vocabulary_size"] > 0
    assert description["end_of_text_token"] == lexer.end_of_text_token
    assert "<|endoftext|>" in description["special_tokens"]


def test_end_of_text_special_token_is_supported():
    # The lexer explicitly allows the GPT end-of-text special token during encoding.
    lexer = Lexer("gpt2")
    text = "La cigale<|endoftext|>La fourmi"

    encoded = lexer.encode(text)

    assert lexer.end_of_text_token in encoded
    assert lexer.decode(encoded) == text


def test_multiple_tiktoken_encodings_round_trip_on_poem():
    # Different BPE vocabularies should all round-trip the poem, even if token counts differ.
    encodings: List[EncodingName] = ["gpt2", "r50k_base", "p50k_base", "cl100k_base", "o200k_base"]

    token_counts = {}
    for encoding in encodings:
        lexer = Lexer(encoding)
        tokens = lexer.encode(LA_CIGALE_ET_LA_FOURMI)
        token_counts[encoding] = len(tokens)
        assert lexer.decode(tokens) == LA_CIGALE_ET_LA_FOURMI

    # Older GPT-2 derived vocabularies align here, while newer vocabularies compress better.
    assert token_counts["gpt2"] == token_counts["r50k_base"]
    assert token_counts["gpt2"] == token_counts["p50k_base"]
    assert token_counts["cl100k_base"] <= token_counts["gpt2"]
    assert token_counts["o200k_base"] <= token_counts["cl100k_base"]
