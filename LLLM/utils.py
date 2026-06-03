import torch


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
