import torch
from torch import nn
from torch.utils.data import DataLoader, RandomSampler

from retro_transformer.tools.optimizer import Noam
from retro_transformer.model import RetroModel
from retro_transformer.tools.database import TextFileDataset, RetroIndex
from retro_transformer.tools.dataset import Dataset

import wandb

from rich.progress import track
from rich.console import Console

import os

from accelerate import Accelerator

accelerator = Accelerator()
device = accelerator.device
console = Console()

class Sampler:
    """
    ## Sampler
    This class greedily samples from a model.
    """

    def __init__(self, model: RetroModel, tds: TextFileDataset, chunk_len: int):
        """
        * `device` is the device of the model
        * `model` is the [Retro mode](retro.html)
        * `tds` is the text dataset (used to get neighbor chunks)
        * `chunk_len` is the length of a chunk
        """
        self.chunk_len = chunk_len
        self.tds = tds
        self.model = model

        # [Retro index](database.html)
        self.index = RetroIndex()

    def retrieve_nearest_neighbours(self, chunk: str):
        """
        ### Retrieve nearest neighbors of a given chunk
        """

        # Retrieve the offsets of the nearest neighbors
        neighbor_offsets = self.index([chunk], None)

        # Get the neighbors (with neighbor length equal to `chunk_len * 2`)
        text = self.tds.train
        neighbors = [text[j: j + self.chunk_len * 2] for j in neighbor_offsets[0]]

        #
        return neighbors

    def sample(self, prompt: str, sample_len: int):
        """
        ### Sample text from the given prompt
        """

        # To store nearest neighbors as strings
        neighbors_str = []

        # Sampled text
        sampled = ''

        # Sample `sample_len` tokens
        for _ in range(sample_len):
            # We need to retrieve neighbors,
            # if there are more sampled chunks than we have already retrieved for
            while len(neighbors_str) < len(prompt) // self.chunk_len:
                # Get the last chunk for which we haven't retrieved neighbors
                off = len(neighbors_str) * self.chunk_len
                chunk = prompt[off: off + self.chunk_len]
                # Retrieve nearest neighbors
                neighbors_str.append(self.retrieve_nearest_neighbours(chunk))

            # Tokenize the input
            src = self.tds.text_to_i(prompt)
            # Tokenize the retrieved neighbors
            neighbors = torch.stack([torch.stack([self.tds.text_to_i(n) for n in chunk]) for chunk in neighbors_str])

            # Move them to the same device as the model
            src = src.to(device)
            neighbors = neighbors.to(device)

            # Get model output
            res = self.model(src[None, :], neighbors[None, :, :, :])

            # Greedily sample the last token
            token = res[0, -1, :].argmax(dim=-1)

            # Add the sampled token text to the prompt and sample text
            prompt += self.tds.itos[token.item()]
            sampled += self.tds.itos[token.item()]

        #
        return sampled


class Trainer:
    """
    ## Retro trainer
    """

    def __init__(self, model: RetroModel, dataloader: DataLoader, optimizer: torch.optim.Optimizer):
        """
        * `device` is the device of the model
        * `model` is the [Retro mode](retro.html)
        * `dataloader` is the dataloader for the [dataset with pre-retrieved neighbors](dataset.html)
        * `optimizer` is the optimizer
        """
        self.optimizer = optimizer
        self.dataloader = dataloader
        self.model = model
        self.loss_func = nn.CrossEntropyLoss()

    def __call__(self):
        """
        ### Train the model for an epoch
        """

        # Iterate through training data
        for (src, tgt, neighbors) in track(self.dataloader, 'Train'):
            # Forward pass
            res = self.model(src, neighbors)
            # Calculate loss
            loss = self.loss_func(res.view(-1, res.shape[-1]), tgt.view(-1))

            # Clear the gradients
            self.optimizer.zero_grad()
            # Backward pass
            accelerator.backward(loss)
            # Optimize the model
            self.optimizer.step()

            # Save training statistics and increment the global step counter
            wandb.log({'loss': loss})


def train(model, workspace: str = './workspace', file_name: str = 'text.txt', wandb_name: str = 'retro-transformer',
          batch_size: int = 4, d_model: int = 512, chunk_len: int = 16, epoch: int = 32, save_dir: str = './workspace'):
    """
    ## Create and train a small model
    """

    # Create an experiment
    wandb.init(project=wandb_name)
    
    if os.path.exists(f'{workspace}/checkpoints/') is False:
        os.mkdir(f'{workspace}/checkpoints/')

    # Load dataset
    tds = TextFileDataset(
        f'{workspace}/{file_name}',
        list)

    # Load [Retro dataset](dataset.html)
    train_dataset = Dataset(f'{workspace}/retro_train_dataset.json', tds)

    # Create dataloader
    train_dl = DataLoader(train_dataset,
                          batch_size=batch_size,
                          sampler=RandomSampler(train_dataset, replacement=True))

    # Create the optimizer
    optimizer = Noam(model.parameters(), lr=1., d_model=d_model, warmup=2_000)
    
    model, optimizer, train_dl = accelerator.prepare(model, optimizer, train_dl)
    
    # Create the `Trainer`
    trainer = Trainer(model, train_dl, optimizer)
    # Create the `Sampler`
    sampler = Sampler(model, tds, chunk_len)
    #

    for _ in range(epoch):
        # Train
        trainer()
        
        # Save model and optimizer
        os.mkdir(f'{save_dir}/checkpoints/{_}')
        torch.save(model.state_dict(), f'{save_dir}/checkpoints/{_}/model.pt')
        torch.save(optimizer.state_dict(), f'{save_dir}/checkpoints/{_}/optimizer.pt')

        # Sample
        sample = sampler.sample('Second Citizen:\nOne word, good citizens.\n\nFirst Citizen:', 128)
        
        # Save the sample
        wandb.log({'sample': sample})

        # Print the sample
        console.print(sample)