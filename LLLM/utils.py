from typing import Any, Protocol, cast

import torch
from torch.utils.data import DataLoader, Dataset
import tiktoken
from tiktoken.core import Encoding


class TensorModel(Protocol):
    def eval(self) -> Any: ...

    def __call__(self, idx: torch.Tensor) -> torch.Tensor: ...


def text_to_token_ids(text: str, tokenizer: Encoding) -> torch.Tensor:
    encoded = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    return torch.tensor(encoded).unsqueeze(0)  # add batch dimension


def token_ids_to_text(token_ids: torch.Tensor, tokenizer: Encoding) -> str:
    flat = token_ids.squeeze(0)  # remove batch dimension
    flat_any = cast(Any, flat)
    return tokenizer.decode(cast(list[int], flat_any.tolist()))


def calc_accuracy_loader(
    data_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    model: TensorModel,
    device: torch.device,
    num_batches: int | None = None,
) -> float:
    model.eval()
    correct_predictions, num_examples = 0, 0

    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            input_batch, target_batch = input_batch.to(device), target_batch.to(device)

            with torch.no_grad():
                logits = model(input_batch)[:, -1, :]  # Logits of last output token
            predicted_labels = torch.argmax(logits, dim=-1)

            num_examples += predicted_labels.shape[0]
            correct_predictions += (predicted_labels == target_batch).sum().item()
        else:
            break
    return correct_predictions / num_examples


def get_device() -> torch.device:
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        # Use PyTorch 2.9 or newer for stable mps results
        major, minor = map(int, torch.__version__.split(".")[:2])
        if (major, minor) >= (2, 9):
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device("cpu")

    return device


def get_device_str() -> str:
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        # Use PyTorch 2.9 or newer for stable mps results
        major, minor = map(int, torch.__version__.split(".")[:2])
        if (major, minor) >= (2, 9):
            device = "mps"
        else:
            device = "cpu"
    else:
        device = "cpu"

    return device


class GPTDatasetV1(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(
        self, txt: str, tokenizer: Encoding, max_length: int, stride: int
    ) -> None:
        self.input_ids: list[torch.Tensor] = []
        self.target_ids: list[torch.Tensor] = []

        # Tokenize the entire text
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})

        # Use a sliding window to chunk the book into overlapping sequences of max_length
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i : i + max_length]
            target_chunk = token_ids[i + 1 : i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self) -> int:
        """
        Returns:
            Number of samples in the dataset.
        """
        return len(self.input_ids)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            Input and target tensors for the requested sample.
        """
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader_v1(
    txt: str,
    batch_size: int = 4,
    max_length: int = 256,
    stride: int = 128,
    shuffle: bool = True,
    drop_last: bool = True,
    num_workers: int = 0,
    encoding: str = "gpt2",
) -> DataLoader[tuple[torch.Tensor, torch.Tensor]]:
    # Initialize the tokenizer
    tokenizer = tiktoken.get_encoding(encoding)

    # Create dataset
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)

    # Create dataloader
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
    )
