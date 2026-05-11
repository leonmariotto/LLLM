"""
DataLoader does:
- asks the dataset for items dataset[i]
- groups them into batches
- optionally shuffles indices
- optionally drops the last incomplete batch

DataSet does:
- define sampling window size.
- apply stride between dataset items: how far the next sample start from the previous one.
- define relationship between sources and target samples.
- define how many samples exist with respect to stride and window_sample.
DataSet class is meant to be implemented by user, torch's DataSet define an interface, that
we want to respect to use our DataSet in DataLoader.
"""
import pytest

torch = pytest.importorskip("torch")

from torch.utils.data import DataLoader, Dataset

FABLE = """
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
""".strip()


def _build_long_text(repetitions: int = 8) -> str:
    return "\n".join(FABLE for _ in range(repetitions))


def _encode(text: str) -> torch.Tensor:
    return torch.tensor([ord(char) for char in text], dtype=torch.long)


def _decode(tokens: torch.Tensor) -> str:
    return "".join(chr(token) for token in tokens.tolist())


class TextWindowDataset(Dataset):
    """
    Educational dataset returning overlapping next-token prediction samples.
    Dataset in PyTorch is just an interface:
        - __len__ tells how many samples exist
        - __getitem__(index) tells how to build sample index
    So the relationship between source and target is entirely your responsibility.
    It is here defined by :
        source = self.tokens[start:stop]
        target = self.tokens[start + 1 : stop + 1]
    stride define the number of token shift between sample.
    window_size define the size of the sample.
    if stride == window: no overlapping :
        Sample 1: "abcd" (source)
        Sample 2: "efgh" (source)
    else if stride == 2 and window == 4:
        Sample 1: "abcd" (source)
        Sample 2: "cdef" (source)
    """

    def __init__(self, text: str, *, window_size: int, stride: int) -> None:
        self.text = text
        self.tokens = _encode(text)
        self.window_size = window_size
        self.stride = stride

    def __len__(self) -> int:
        # Remove incomplete window, and divide by stride size.
        return (len(self.tokens) - self.window_size - 1) // self.stride + 1

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = index * self.stride
        stop = start + self.window_size
        source = self.tokens[start:stop]
        # target sample is shifted of 1 to the left.
        target = self.tokens[start + 1 : stop + 1]
        return source, target


def test_torch_dataset_dataloader_sequential_sampling() -> None:
    dataset = TextWindowDataset(_build_long_text(), window_size=24, stride=8)
    loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0)

    # Get first batch
    sources, targets = next(iter(loader))

    # The default collate function stacks per-sample tensors into batch tensors.
    assert sources.shape == (2, 24)
    assert targets.shape == (2, 24)

    # The first sample starts at the beginning of the text and the target is shifted by one token.
    assert _decode(sources[0]) == FABLE[:24]
    assert _decode(targets[0]) == FABLE[1:25]

    # The second sample illustrates overlapping windows controlled by the stride.
    assert _decode(sources[1]) == _build_long_text()[8:32]
    assert _decode(targets[1]) == _build_long_text()[9:33]


def test_torch_dataset_dataloader_drop_last() -> None:
    """
    DataLoader create a as many as possible batch of 5 DataSet entries.
    So in our case :
        - batch 1: 5 samples (each sample with window size 18 and stride 11)
        - batch 2: 5 samples (each sample with window size 18 and stride 11)
        - batch 3: 2 samples (each sample with window size 18 and stride 11)
    Because DataSet length in not dividable by 5 (batch_size) we end up having an incomplete last batch.
    DataLoader option drop_last remove it.
    """
    dataset = TextWindowDataset(_build_long_text(), window_size=18, stride=11)
    batch_size = 5

    keep_all_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )
    drop_last_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=0,
    )

    keep_all_batches = list(keep_all_loader)
    drop_last_batches = list(drop_last_loader)

    # We picked parameters that produce an incomplete final batch.
    assert len(dataset) % batch_size != 0
    # Without drop_last, iterating over the loader returns every sample in the dataset.
    assert sum(batch[0].shape[0] for batch in keep_all_batches) == len(dataset)
    # With drop_last, only full batches remain.
    assert sum(batch[0].shape[0] for batch in drop_last_batches) == len(drop_last_batches) * batch_size
    assert sum(batch[0].shape[0] for batch in drop_last_batches) < len(dataset)


def test_torch_dataset_dataloader_shuffle_is_seeded_and_lossless() -> None:
    dataset = TextWindowDataset(_build_long_text(), window_size=16, stride=5)
    expected_samples = [_decode(dataset[index][0]) for index in range(len(dataset))]

    def collect_sources(seed: int) -> list[str]:
        loader = DataLoader(
            dataset,
            batch_size=4,
            shuffle=True,
            num_workers=0,
            generator=torch.Generator().manual_seed(seed),
        )
        return [_decode(sample) for batch, _ in loader for sample in batch]

    shuffled_once = collect_sources(1234)
    shuffled_twice = collect_sources(1234)
    shuffled_other_seed = collect_sources(5678)

    # Seeding the generator makes the shuffled order reproducible.
    assert shuffled_once == shuffled_twice
    # Shuffling changes the order, and changing the seed changes the shuffled order again.
    assert shuffled_once != expected_samples
    assert shuffled_once != shuffled_other_seed
    # Even when shuffled, the loader still visits each sample exactly once.
    assert sorted(shuffled_once) == sorted(expected_samples)
