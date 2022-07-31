from pathlib import PurePath, Path
from typing import List, Callable, Dict, Optional
from retro_transformer.bert import BERTForChunkEmbeddings

import faiss
import numpy as np
import torch

from rich.progress import track
from rich.console import Console

console = Console()

class TextDataset:
    itos: List[str]
    stoi: Dict[str, int]
    n_tokens: int
    train: str
    valid: str
    standard_tokens: List[str] = []

    @staticmethod
    def load(path: PurePath):
        with open(str(path), 'r') as f:
            return f.read()

    def __init__(self, path: PurePath, tokenizer: Callable, train: str, valid: str, test: str, *,
                 n_tokens: Optional[int] = None,
                 stoi: Optional[Dict[str, int]] = None,
                 itos: Optional[List[str]] = None):
        self.test = test
        self.valid = valid
        self.train = train
        self.tokenizer = tokenizer
        self.path = path

        if n_tokens or stoi or itos:
            assert stoi and itos and n_tokens
            self.n_tokens = n_tokens
            self.stoi = stoi
            self.itos = itos
        else:
            self.n_tokens = len(self.standard_tokens)
            self.stoi = {t: i for i, t in enumerate(self.standard_tokens)}

            with console.status("Tokenize"):
                tokens = self.tokenizer(self.train) + self.tokenizer(self.valid)
                tokens = sorted(list(set(tokens)))
                console.log("Tokenize")

            for t in track(tokens, description='Build vocabulary'):
                self.stoi[t] = self.n_tokens
                self.n_tokens += 1

            self.itos = [''] * self.n_tokens
            for t, n in self.stoi.items():
                self.itos[n] = t

    def text_to_i(self, text: str) -> torch.Tensor:
        tokens = self.tokenizer(text)
        return torch.tensor([self.stoi[s] for s in tokens if s in self.stoi], dtype=torch.long)

    def __repr__(self):
        return f'{len(self.train) / 1_000_000 :,.2f}M, {len(self.valid) / 1_000_000 :,.2f}M - {str(self.path)}'


class TextFileDataset(TextDataset):
    standard_tokens = []

    def __init__(self, path: PurePath, tokenizer: Callable, *,
                 url: Optional[str] = None,
                 filter_subset: Optional[int] = None):
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(str(path))

        with console.status("Load data"):
            text = self.load(path)
            if filter_subset:
                text = text[:filter_subset]
            split = int(len(text) * .9)
            train = text[:split]
            valid = text[split:]
            console.log("Load data")

        super().__init__(path, tokenizer, train, valid, '')


def build_database(workspace: str = './workspace', file_name: str = 'text.txt', chunk_len: int = 16,
                   batch_size: int = 64, d_emb: int = 768, n_centeroids: int = 256, code_size: int = 64,
                   n_probe: int = 8, n_train: int = 50_000, bert: BERTForChunkEmbeddings = None):

    # Load the dataset text file
    dataset = TextFileDataset(
        f'{workspace}/{file_name}',
        list)

    # Get training data (a string)
    text = dataset.train

    # Split the text into chunks of `chunk_length`
    chunks = [text[i:i + chunk_len] for i in range(0, len(text), chunk_len) if i + chunk_len * 2 < len(text)]
    # Get the offsets of each of the chunks
    chunk_offsets = np.array([i for i in range(0, len(text), chunk_len) if i + chunk_len * 2 < len(text)])
    # Number of chunks
    n_chunks = len(chunks)

    # Get chunk embeddings by processing `batch_size` number of chunks on each iteration
    chunk_emb = []
    for i in track(range(0, n_chunks, batch_size), description='Get embeddings'):
        chunk_emb.append(bert(chunks[i: i + batch_size]).cpu())
    # Merge them into a single tensor
    chunk_emb = torch.cat(chunk_emb, dim=0).numpy()

    # Create the [FAISS index](https://faiss.ai/cpp_api/struct/structfaiss_1_1IndexIVFPQ.html)
    quantizer = faiss.IndexFlatL2(d_emb)
    index = faiss.IndexIVFPQ(quantizer, d_emb, n_centeroids, code_size, 8)
    index.nprobe = n_probe

    # Get a random sample of the the chunk indexes
    random_sample = np.random.choice(np.arange(n_chunks), size=[min(n_train, n_chunks)], replace=False)

    # Train the index to store the keys
    with console.status("Train index"):
        index.train(chunk_emb[random_sample])
        console.log("Train index")

    # Add the chunks to the index in batches of size `1024`
    for s in track(range(0, n_chunks, 1024), description='Index'):
        e = min(s + 1024, n_chunks)
        # Add to index
        index.add_with_ids(chunk_emb[s:e], chunk_offsets[s: e])

    # Save the index
    with console.status("Save"):
        faiss.write_index(index, f'{workspace}/retro.index')
        console.log("Save index")
    
    return dataset.n_tokens


class RetroIndex:
    """
    ## Index for retrieving nearest neighbors
    """

    def __init__(self, workspace: str = './workspace', chunk_len: int = 16, n_probe: int = 8,
                 n_neighbors: int = 2, n_extra: int = 2,
                 exclude_neighbor_span: int = 8, bert: BERTForChunkEmbeddings = None):
        """
        * `chunk_len` is the chunk length
        * `n_probe` is the number of lists to probe
        * `n_neighbors` is the number of neighbors to retrieve
        * `n_extra` is the number of extra neighbors to retrieve since we will be
            removing neighbors overlapping with the query chunk
        * `exclude_neighbor_span` is the extra text length to avoid when checking for overlaps
        """
        self.n_neighbors = n_neighbors
        self.chunk_len = chunk_len
        self.exclude_neighbor_span = exclude_neighbor_span
        self.n_extra = n_extra

        # Initialize BERT to get $\text{B\small{ERT}}(N)$
        self.bert = bert
        # Load the database
        with console.status("Load index"):
            self.index = faiss.read_index(f'{workspace}/retro.index')
            self.index.nprobe = n_probe
            console.log("Load index")

    def filter_neighbors(self, offset: int, neighbor_offsets: List[int]):
        """
        #### Filter neighbors that overlap with the query
        
        The positions of the neighbors are given by `neighbor_offsets` and the position
        of the query chunk is `offset`.
        """
        return [n for n in neighbor_offsets
                if n < offset - (self.chunk_len + self.exclude_neighbor_span)
                or n > offset + (self.chunk_len + self.exclude_neighbor_span)]

    def __call__(self, query_chunks: List[str], offsets: Optional[List[int]]):
        """
        ### Retrieve nearest neighbors
        """

        # Get $\text{B\small{ERT}}(N)$ of query chunks
        emb = self.bert(query_chunks).cpu()

        # Get `n_neighbors + n_extra` nearest neighbors from the database
        distance, neighbor_offsets = self.index.search(emb.numpy(), self.n_neighbors + self.n_extra)

        # If the query chunk offsets are given filter out overlapping chunks
        if offsets is not None:
            neighbor_offsets = [self.filter_neighbors(off, n_off)
                                for off, n_off in zip(offsets, neighbor_offsets)]

        # Get the closest `n_neighbors` after filtering
        neighbor_offsets = [n_off[:self.n_neighbors] for n_off in neighbor_offsets]

        #
        return neighbor_offsets