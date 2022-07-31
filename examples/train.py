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

build_database(workspace, text_file, bert=bert, chunk_len=chunk_len)
index = RetroIndex(workspace, chunk_len, bert=bert)
num_tokens = build_dataset(workspace, text_file, chunk_len=chunk_len, index=index)

nearest_neighbor_encoder = NearestNeighborEncoder(chunk_len=chunk_len, n_layers=n_layers,
                                                  d_model=d_model, d_ff=d_ff, n_heads=n_heads,
                                                  d_k=d_k, ca_layers={3})

model = RetroModel(n_vocab=num_tokens, d_model=d_model, n_layers=n_layers, chunk_len=chunk_len,
                   n_heads=n_heads, d_k=d_k, d_ff=d_ff, encoder=nearest_neighbor_encoder, ca_layers={3, 5})

train(model, workspace, text_file, chunk_len=chunk_len, d_model=d_model)