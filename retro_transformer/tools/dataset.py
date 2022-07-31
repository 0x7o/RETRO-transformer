import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset as PyTorchDataset
from retro_transformer.tools.database import TextFileDataset, RetroIndex, TextDataset

from rich.progress import track
from rich.console import Console
console = Console()


def build_dataset(workspace: str = './workspace', file_name: str = 'text.txt', 
                  chunk_len: int = 16, chunks_per_sample: int = 2,
                  skip_range: int = 8):
    """
    ## Build the dataset
    * `chunk_len` is the chunk length
    * `chunks_per_sample` is the number of chunks per training sample
    * `skip_range` is the maximum number of characters to skip between two samples.
        We skip a few characters between samples to make sure the samples
        aren't aligned perfectly with the chunks in the [database](database.html)
    """

    # Load the text file
    dataset = TextFileDataset(
        f'{workspace}/{file_name}',
        list)

    # Training portion of it
    text = dataset.train

    # Load the index for retrieving neighbors
    index = RetroIndex()

    # The input sample offsets
    sample_offsets = []
    # Cursor for the text
    i = 0
    while i < len(text):
        # Skip a few characters to make sure it's not aligned with the neighbors
        skip = np.random.randint(skip_range)
        i += skip

        # Stop if we've reached the end of the text
        if i + chunks_per_sample * chunk_len > len(text):
            break

        # Collect the offset
        sample_offsets.append(i)

        # Increment the cursor
        i += chunks_per_sample * chunk_len

    # For samples
    samples = []
    # Iterate through sample offsets
    for i in track(sample_offsets, 'Gather Neighbors'):
        # Get the sample including an extra character (for prediction)
        sample = text[i: i + chunks_per_sample * chunk_len + 1]
        # The input
        src = sample[:-1]
        # Break it into chunks
        chunks = [src[j:j + chunk_len] for j in range(0, len(src), chunk_len)]
        # The chunk offsets
        chunk_offsets = [j + i for j in range(0, len(src), chunk_len)]

        # Retrieve nearest neighbors
        neighbor_offsets = index(chunks, chunk_offsets)

        # Get neighbor texts. The neighbor length is twice the `chunk_len`
        neighbors = [[text[j: j + chunk_len * 2] for j in n_off] for n_off in neighbor_offsets]

        # Add to list of samples
        samples.append((sample[:-1], sample[1:], neighbors))

    # Save the samples in JSON.
    # We don't need to use complex dataset storage mechanisms or pre-tokenize
    # since our dataset is small.
    with open(f'{workspace}/retro_train_dataset.json', 'w') as f:
        f.write(json.dumps(samples))
    
    return dataset.n_tokens


class Dataset(PyTorchDataset):
    """
    ## Dataset
    This is the PyTorch dataset that loads the dataset created
    by `build_dataset`.
    """
    def __init__(self, file_path: Path, tds: TextDataset):
        """
        * `file_path` is the path of the saved JSON file
        * `tds` is the `TextDataset`
        """

        self.tds = tds
        # Load the samples
        with open(str(file_path), 'r') as f:
            self.samples = json.loads(f.read())

    def __len__(self):
        """
        Number of samples
        """
        return len(self.samples)

    def __getitem__(self, idx: int):
        """
        Get a sample
        """
        # Get the sample
        s = self.samples[idx]
        # Tokenize
        src = self.tds.text_to_i(s[0])
        tgt = self.tds.text_to_i(s[1])
        neighbors = torch.stack([torch.stack([self.tds.text_to_i(n) for n in chunks]) for chunks in s[2]])
        #
        return src, tgt, neighbors