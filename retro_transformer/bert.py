from typing import List
from rich.console import Console

import torch
from transformers import BertTokenizer, BertModel

from transformers import logging
logging.set_verbosity_error()

class BERTForChunkEmbeddings:
    """
    ## BERT Embeddings
    For a given chunk of text $N$ this class generates BERT embeddings $\text{B\small{ERT}}(N)$.
    $\text{B\small{ERT}}(N)$ is the average of BERT embeddings of all the tokens in $N$.
    """

    def __init__(self, model_name, device: torch.device):
        self.device = device
        console = Console()

        with console.status(f"Loading {model_name} tokenizer") as status:
            # Load the BERT tokenizer from [HuggingFace](https://huggingface.co/bert-base-uncased)
            self.tokenizer = BertTokenizer.from_pretrained(model_name)
            console.log(f"{model_name} tokenizer loaded")

        with console.status(f"Loading {model_name} model") as status:
            # Load the BERT model from [HuggingFace](https://huggingface.co/bert-base-uncased)
            self.model = BertModel.from_pretrained(model_name)

            # Move the model to `device`
            self.model.to(device)
            console.log(f"{model_name} loaded")

    @staticmethod
    def _trim_chunk(chunk: str):
        """
        In this implementation, we do not make chunks with a fixed number of tokens.
        One of the reasons is that this implementation uses character-level tokens and BERT
        uses its sub-word tokenizer.
        So this method will truncate the text to make sure there are no partial tokens.
        For instance, a chunk could be like `s a popular programming la`, with partial
        words (partial sub-word tokens) on the ends.
        We strip them off to get better BERT embeddings.
        As mentioned earlier this is not necessary if we broke chunks after tokenizing.
        """
        # Strip whitespace
        stripped = chunk.strip()
        # Break words
        parts = stripped.split()
        # Remove first and last pieces
        stripped = stripped[len(parts[0]):-len(parts[-1])]

        # Remove whitespace
        stripped = stripped.strip()

        # If empty return original string
        if not stripped:
            return chunk
        # Otherwise, return the stripped string
        else:
            return stripped

    def __call__(self, chunks: List[str]):
        """
        ### Get $\text{B\small{ERT}}(N)$ for a list of chunks.
        """

        with torch.no_grad():
            trimmed_chunks = [self._trim_chunk(c) for c in chunks]

            tokens = self.tokenizer(trimmed_chunks, return_tensors='pt', add_special_tokens=False, padding=True)

            input_ids = tokens['input_ids'].to(self.device)
            attention_mask = tokens['attention_mask'].to(self.device)
            token_type_ids = tokens['token_type_ids'].to(self.device)
            output = self.model(input_ids=input_ids,
                                attention_mask=attention_mask,
                                token_type_ids=token_type_ids)

            state = output['last_hidden_state']
            emb = (state * attention_mask[:, :, None]).sum(dim=1) / attention_mask[:, :, None].sum(dim=1)

            return emb
