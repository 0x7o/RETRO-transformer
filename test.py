from retro_transformer.bert import BERTForChunkEmbeddings
#from retro_transformer.tools.database import build_database
from retro_transformer.model import RetroModel

bert = BERTForChunkEmbeddings('cointegrated/rubert-tiny', 'cuda')
model = RetroModel()
#build_database('data', 'test.txt', bert=bert)