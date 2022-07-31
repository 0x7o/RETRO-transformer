import setuptools

setuptools.setup(
	name="retro_transformer",
	version="1.0.3",
	author="Danil Kononyuk",
	author_email="me@0x7o.link",
	description="Easy-to-use Retrieval-Enhanced Transformer (Retro) implementation",
	long_description="""![RETRO](data/RETRO.png)
# Retrieval-Enhanced Transformer (WIP)

Easy-to-use [Retro](https://arxiv.org/abs/2112.04426) implementation in PyTorch.

This code based on [labml.ai](https://nn.labml.ai/transformers/retro/index.html) and [accelerate](https://github.com/huggingface/accelerate) for light inference and training on CPUs, GPUs, TPUs.

```python
from retro_transformer.bert import BERTForChunkEmbeddings
from retro_transformer.tools.database import build_database, RetroIndex
from retro_transformer.tools.dataset import build_dataset
from retro_transformer.model import RetroModel, NearestNeighborEncoder
from retro_transformer.tools.train import train

chunk_len = 16
d_model = 128
d_ff = 512
n_heads = 16
d_k = 16
n_layers = 16
workspace = './workspace'
text_file = 'text.txt'

bert = BERTForChunkEmbeddings('bert-base-uncased', 'cuda')
index = RetroIndex(workspace, chunk_len, bert=bert)

build_database(workspace, text_file, bert=bert, chunk_len=chunk_len)
num_tokens = build_dataset(workspace, text_file, chunk_len=chunk_len, index=index)

nearest_neighbor_encoder = NearestNeighborEncoder(chunk_len=chunk_len, n_layers=n_layers,
                                                  d_model=d_model, d_ff=d_ff, n_heads=n_heads,
                                                  d_k=d_k, ca_layers={3})

model = RetroModel(n_vocab=num_tokens, d_model=d_model, n_layers=n_layers, chunk_len=chunk_len,
                   n_heads=n_heads, d_k=d_k, d_ff=d_ff, encoder=nearest_neighbor_encoder, ca_layers={3, 5})

train(model, workspace, text_file, chunk_len=chunk_len, d_model=d_model)
```
""",
	long_description_content_type="text/markdown",
	url="https://github.com/0x7o/RETRO-transformer",
	packages=setuptools.find_packages(),
	classifiers=[],
	python_requires='>=3.6',
 	install_requires=[
       "torch",
"accelerate",
"wandb",
"rich",
"numpy",
"autofaiss",
"transformers",
"labml_nn"],
)